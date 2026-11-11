# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The Delaunay tessellation under the surface mesh and the dislocation
analysis: a valid tessellation of the box for perfect lattices too, whose
atoms, moved at random by 1e-4 Å as the analyses do, lie by many nearly on
one sphere."""

import numpy as np
import pytest
from mdapy._core import analysis as _native

from mdapy import Box
from mdapy.analysis import DislocationAnalysis, SurfaceMesh
from mdapy.build import build_crystal, build_polycrystal, dislocation_quadrupole
from mdapy.transform import affine

from dislocation_models import CU

JITTER = 1e-4  # as the analyses move the atoms before the tessellation


def tessellate(system, *, jitter=JITTER, noise=0.0, window=1e-10, seed=1):
    """The tetrahedra of the system, with the positions they refer to."""
    rng = np.random.default_rng(seed)
    positions = system["positions"] + rng.normal(0.0, noise, (len(system), 3))
    positions = positions + rng.uniform(-jitter, jitter, (len(system), 3))
    box = system.box
    fractional = (positions - box.origin) @ box.inverse
    fractional[:, box.pbc] -= np.floor(fractional[:, box.pbc])
    positions = np.ascontiguousarray(fractional @ box.matrix + box.origin)
    atoms, shifts, neighbors, sides, repeated = _native.delaunay_tetrahedra(
        positions,
        np.ascontiguousarray(box.matrix),
        np.ascontiguousarray(box.origin),
        np.ascontiguousarray(box.pbc),
        window,
    )
    assert not repeated
    return (
        positions,
        atoms.reshape(-1, 4),
        shifts.reshape(-1, 4, 3).astype(np.int64),
        neighbors.reshape(-1, 4),
        sides.reshape(-1, 4).astype(np.int64),
    )


def check(system, periodic=True, **options):
    """Every face shared by two tetrahedra that agree on it, no overlap, the
    volumes filling the box, and no atom inside the sphere of a neighbor."""
    positions, atoms, shifts, neighbors, sides = tessellate(system, **options)
    matrix = system.box.matrix
    corners = positions[atoms] + shifts @ matrix  # (T, 4, 3)
    count = len(atoms)
    assert count > 0
    # no atom twice in a tetrahedron
    keyed = atoms[:, :, None] * 1000 + (shifts + 8) @ np.array([1, 16, 256])[:, None]
    keyed = np.sort(keyed[:, :, 0], axis=1)
    assert np.all(np.diff(keyed, axis=1) != 0)
    edges = corners[:, 1:] - corners[:, :1]
    volumes = np.abs(np.linalg.det(edges)) / 6
    assert volumes.min() > 0
    linked = neighbors >= 0
    if periodic:
        assert linked.all()
        np.testing.assert_allclose(volumes.sum(), abs(np.linalg.det(matrix)), rtol=1e-9)
    t, side = np.nonzero(linked)
    other, back = neighbors[t, side], sides[t, side]
    # the neighbor across a face has this one across the same face
    np.testing.assert_array_equal(neighbors[other, back], t)
    np.testing.assert_array_equal(sides[other, back], side)
    # the face itself: the same three atoms at the same relative places
    mine = np.ones((len(t), 4), bool)
    mine[np.arange(len(t)), side] = False
    theirs = np.ones((len(t), 4), bool)
    theirs[np.arange(len(t)), back] = False
    face = corners[t][mine].reshape(-1, 3, 3)
    seen = corners[other][theirs].reshape(-1, 3, 3)
    face_atoms = atoms[t][mine].reshape(-1, 3)
    seen_atoms = atoms[other][theirs].reshape(-1, 3)
    order_a, order_b = np.argsort(face_atoms, axis=1), np.argsort(seen_atoms, axis=1)
    rows = np.arange(len(t))[:, None]
    np.testing.assert_array_equal(face_atoms[rows, order_a], seen_atoms[rows, order_b])
    face, seen = face[rows, order_a], seen[rows, order_b]
    np.testing.assert_allclose(face - face[:, :1], seen - seen[:, :1], atol=1e-9)
    # the atom of the neighbor off the face lies outside the sphere of this one
    center, radius = circumspheres(corners[t])
    beyond = corners[other, back] + (face[:, 0] - seen[:, 0])
    distance = np.linalg.norm(beyond - center, axis=1)
    assert np.all(distance >= radius - 1e-6 * radius)
    return count


def circumspheres(corners):
    a = corners[:, 1:] - corners[:, :1]
    rhs = 0.5 * np.einsum("tij,tij->ti", a, a)
    local = np.linalg.solve(a, rhs[..., None])[..., 0]
    return corners[:, 0] + local, np.linalg.norm(local, axis=1)


PERFECT = [
    ("fcc", 3.615, None, (16, 16, 16)),
    ("hcp", 3.21, 5.21, (12, 12, 8)),
    ("diamond", 5.43, None, (12, 12, 12)),
    ("lonsdaleite", 3.84, 6.28, (10, 10, 6)),
    ("bcc", 2.87, None, (16, 16, 16)),
    ("sc", 2.5, None, (12, 12, 12)),
]


@pytest.mark.parametrize(
    "structure, a, c, repeat", PERFECT, ids=[p[0] for p in PERFECT]
)
def test_perfect_lattices_moved_as_the_analyses_do(structure, a, c, repeat):
    check(build_crystal(structure, a, c=c, repeat=repeat))


@pytest.mark.parametrize("noise", [1e-5, 1e-3, 0.05])
@pytest.mark.parametrize(
    "structure, a, c",
    [("fcc", 3.615, None), ("hcp", 3.21, 5.21), ("diamond", 5.43, None)],
)
def test_a_wide_window_still_gives_a_tessellation(structure, a, c, noise):
    """With points within 1e-6 of the reach of every plane settled by the
    atoms, thousands of times per box."""
    crystal = build_crystal(
        structure, a, c=c, repeat=(8, 8, 8) if c is None else (8, 8, 5)
    )
    check(crystal, noise=noise, window=1e-6, seed=7)


def test_triclinic_boxes():
    crystal = build_crystal("fcc", 3.615, repeat=(8, 8, 8))
    sheared = affine(crystal, [[1.0, 0.3, 0.2], [0.0, 1.0, -0.25], [0.0, 0.0, 1.0]])
    check(sheared)


@pytest.mark.parametrize(
    "pbc", [(True, True, False), (True, False, False), (False, False, False)]
)
def test_open_boundaries(pbc):
    crystal = build_crystal("fcc", 3.615, repeat=(8, 8, 8))
    opened = crystal.with_box(Box(crystal.box.matrix, pbc=pbc))
    check(opened, periodic=False)


def test_built_systems_are_analyzed():
    """A polycrystal as built and a quadrupole of split dislocations, whose
    atoms lie by many nearly on spheres."""
    polycrystal = build_polycrystal(
        build_crystal("fcc", 3.615, elements="Cu"), (60, 60, 60), 5, seed=1
    )
    assert len(polycrystal) > 15000
    check(polycrystal)
    DislocationAnalysis("fcc").compute(polycrystal)
    SurfaceMesh().compute(polycrystal)
    quadrupole = dislocation_quadrupole(
        "fcc edge", 3.615, CU, (28, 16, 4), elements="Cu", dissociation=10
    )
    check(quadrupole)
    found = DislocationAnalysis("fcc").compute(quadrupole)
    # a dislocation and its opposite in the box, each split into two partials
    assert [line.family for line in found.lines] == ["1/6<112>"] * 4
    total = sum(line.spatial_burgers for line in found.lines)
    np.testing.assert_allclose(total, 0.0, atol=0.02)


@pytest.mark.parametrize("n", [16, 20])
def test_perfect_fcc_is_analyzed(n):
    crystal = build_crystal("fcc", 3.615, elements="Cu", repeat=(n, n, n))
    assert DislocationAnalysis("fcc").compute(crystal).lines == ()
    assert SurfaceMesh().compute(crystal).area == 0.0
