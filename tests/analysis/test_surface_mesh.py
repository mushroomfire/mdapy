# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import pytest

import mdapy
from mdapy.analysis import SurfaceMesh
from mdapy.build import add_vacuum, build_crystal
from mdapy.errors import InvalidParameter

from lattices import crystal


def with_void(radius=8.0, jitter=0.05, repeat=10, seed=1):
    """fcc copper without the atoms in a sphere at the center of the box,
    the others moved at random."""
    system = crystal("fcc", 3.615, (repeat,) * 3, jitter=jitter, seed=seed)
    center = system.box.matrix.sum(axis=0) / 2
    return system.filter(np.linalg.norm(system["positions"] - center, axis=1) > radius)


def edges_of(result):
    """Each edge of the mesh as (vertex, vertex, image of the second from
    the first), once per triangle it borders, in its direction there."""
    faces, shifts = result.mesh.faces, result.mesh.shifts.astype(int)
    edges = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        edges.append(
            np.column_stack([faces[:, a], faces[:, b], shifts[:, b] - shifts[:, a]])
        )
    return np.concatenate(edges)


def assert_closed(result):
    """Every edge borders two triangles, run in opposite directions."""
    edges = edges_of(result)
    reverse = np.column_stack([edges[:, 1], edges[:, 0], -edges[:, 2:]])
    forward = {tuple(e) for e in edges}
    assert len(forward) == len(edges)  # no edge twice in one direction
    assert forward == {tuple(e) for e in reverse}


def test_a_perfect_crystal_is_solid():
    system = build_crystal("fcc", 3.615, elements="Cu", repeat=(5, 5, 5))
    result = SurfaceMesh(regions=True, surface=True).compute(system)
    assert len(result.mesh) == 0 and result.area == 0
    assert result.solid_volume == pytest.approx(system.box.volume, rel=1e-12)
    regions = result.regions
    np.testing.assert_array_equal(regions.solid, [True])
    np.testing.assert_array_equal(regions.periodic, [True])
    assert len(regions.voids) == 0
    assert not result.surface.any()


@pytest.mark.parametrize(
    "orientation",
    [None, [(1, -1, 0), (1, 1, -2), (1, 1, 1)], [(0, 0, 1), (1, -1, 0), (1, 1, 0)]],
)
@pytest.mark.parametrize("smoothing", [0, 8])
def test_flat_surfaces_of_a_slab(orientation, smoothing):
    cell = build_crystal("fcc", 3.615, elements="Cu", orientation=orientation)
    repeat = np.ceil([10, 10, 14] / cell.box.matrix.diagonal()).astype(int)
    slab = build_crystal(
        "fcc", 3.615, elements="Cu", orientation=orientation, repeat=repeat
    )
    z = slab["positions"][:, 2]
    slab = add_vacuum(slab, 12.0)
    result = SurfaceMesh(smoothing=smoothing, regions=True, surface=True).compute(slab)
    face = np.linalg.norm(np.cross(slab.box.matrix[0], slab.box.matrix[1]))
    # the planes of the outer atoms
    assert result.area == pytest.approx(2 * face, rel=1e-9)
    assert result.solid_volume == pytest.approx(face * np.ptp(z), rel=1e-9)
    on_surface = np.isclose(z, z.min()) | np.isclose(z, z.max())
    np.testing.assert_array_equal(result.surface, on_surface)
    np.testing.assert_allclose(
        result.mesh.vertices[:, 2] % slab.box.matrix[2, 2],
        slab["positions"][result.atoms, 2],
        atol=1e-9,
    )
    # the solid and the vacuum each run along the slab without end
    regions = result.regions
    np.testing.assert_array_equal(regions.solid, [True, False])
    np.testing.assert_array_equal(regions.periodic, [True, True])
    assert len(regions.voids) == 0
    assert regions.volume.sum() == pytest.approx(slab.box.volume, rel=1e-12)
    assert_closed(result)


def test_a_void():
    system = with_void()
    result = SurfaceMesh(smoothing=0, regions=True, surface=True).compute(system)
    regions = result.regions
    void = regions.voids
    assert len(void) == 1 and not regions.solid[void[0]]
    # the void and the solid fill the box
    assert regions.volume.sum() == pytest.approx(system.box.volume, rel=1e-12)
    assert result.solid_volume == pytest.approx(regions.volume[regions.solid].sum())
    np.testing.assert_allclose(regions.area, result.area)
    # a closed surface like a sphere, through atoms just outside the hole
    assert_closed(result)
    edges = len(edges_of(result)) // 2
    assert len(result.atoms) - edges + len(result.mesh) == 2
    center = system.box.matrix.sum(axis=0) / 2
    distance = np.linalg.norm(system["positions"][result.atoms] - center, axis=1)
    assert distance.min() > 8 and distance.max() < 8 + 3.615
    np.testing.assert_array_equal(np.flatnonzero(result.surface), result.atoms)
    # the normals point into the void
    corners = result.mesh.corners()
    toward = center - corners.mean(axis=1)
    assert ((toward * result.mesh.normals).sum(axis=1) > 0).all()
    # the solid side of each triangle, then the void
    np.testing.assert_array_equal(
        result.regions.faces,
        np.tile([np.flatnonzero(regions.solid)[0], void[0]], (len(result.mesh), 1)),
    )


def test_smoothing_moves_the_vertices_only():
    system = with_void()
    rough = SurfaceMesh(smoothing=0, regions=True).compute(system)
    smooth = SurfaceMesh(smoothing=8, regions=True).compute(system)
    np.testing.assert_array_equal(smooth.mesh.faces, rough.mesh.faces)
    np.testing.assert_array_equal(smooth.atoms, rough.atoms)
    assert smooth.solid_volume == rough.solid_volume
    np.testing.assert_array_equal(smooth.regions.volume, rough.regions.volume)
    assert not np.allclose(smooth.mesh.vertices, rough.mesh.vertices)
    # rounder: a smaller area around nearly the same volume
    assert smooth.area < rough.area
    np.testing.assert_allclose(rough.mesh.vertices, system["positions"][rough.atoms])


def test_a_particle_in_an_open_box():
    big = crystal("fcc", 3.615, (8, 8, 8), jitter=0.05, seed=2)
    center = big.box.matrix.sum(axis=0) / 2
    ball = big.filter(np.linalg.norm(big["positions"] - center, axis=1) < 11)
    # the box as tight as the atoms: no tetrahedron that may be solid is lost
    low, high = ball["positions"].min(axis=0), ball["positions"].max(axis=0)
    tight = ball.with_box(mdapy.Box(high - low, origin=low, pbc=False))
    roomy = ball.with_box(mdapy.Box(big.box.matrix * 2, origin=-center, pbc=False))
    result = SurfaceMesh(regions=True, surface=True).compute(tight)
    other = SurfaceMesh(regions=True, surface=True).compute(roomy)
    assert result.area == pytest.approx(other.area, rel=1e-12)
    assert result.solid_volume == pytest.approx(other.solid_volume, rel=1e-12)
    regions = result.regions
    np.testing.assert_array_equal(regions.solid, [False, True])
    np.testing.assert_array_equal(regions.exterior, [True, False])
    assert regions.volume[0] == np.inf
    assert regions.volume[1] == pytest.approx(result.solid_volume)
    assert len(regions.voids) == 0
    assert_closed(result)
    # convex: the surface atoms are the outer ones
    distance = np.linalg.norm(ball["positions"] - center, axis=1)
    assert distance[result.surface].min() > distance[~result.surface].max() - 3.0


def test_invariance():
    system = with_void(seed=3)
    result = SurfaceMesh(regions=True).compute(system)
    # moved by a vector, the atoms wrapped or not
    moved = system.with_columns(positions=system["positions"] + [5.0, -40.0, 17.0])
    other = SurfaceMesh(regions=True).compute(moved)
    assert other.area == pytest.approx(result.area, rel=1e-10)
    assert other.solid_volume == pytest.approx(result.solid_volume, rel=1e-12)
    # turned with its box
    q, _ = np.linalg.qr(np.random.default_rng(5).normal(size=(3, 3)))
    q[:, 0] *= np.sign(np.linalg.det(q))
    turned = mdapy.System(
        system["positions"] @ q.T + 3.0,
        mdapy.Box(system.box.matrix @ q.T, origin=[3.0, 3.0, 3.0]),
        elements="Cu",
    )
    other = SurfaceMesh(regions=True).compute(turned)
    assert other.area == pytest.approx(result.area, rel=1e-10)
    np.testing.assert_allclose(
        np.sort(other.regions.volume), np.sort(result.regions.volume), rtol=1e-10
    )
    with mdapy.num_threads(3):
        again = SurfaceMesh(regions=True).compute(system)
    np.testing.assert_array_equal(again.mesh.vertices, result.mesh.vertices)
    np.testing.assert_array_equal(again.mesh.faces, result.mesh.faces)
    np.testing.assert_array_equal(again.regions.faces, result.regions.faces)


def test_voids_across_the_boundary_and_a_triclinic_box():
    system = crystal("hcp", 3.2, (12, 12, 8), jitter=0.05, seed=4)
    box = system.box
    keep = np.ones(len(system), dtype=bool)
    # one void across a corner of the box, one inside
    for center, radius in [((0.0, 0.0, 0.0), 6.0), ((0.5, 0.5, 0.5), 5.0)]:
        d = box.minimum_image(system["positions"] - np.array(center) @ box.matrix)
        keep &= np.linalg.norm(d, axis=1) > radius
    system = system.filter(keep)
    result = SurfaceMesh(regions=True).compute(system)
    regions = result.regions
    assert len(regions.voids) == 2
    assert regions.volume.sum() == pytest.approx(box.volume, rel=1e-12)
    assert regions.volume[regions.voids].max() > regions.volume[regions.voids].min()
    assert_closed(result)


def test_columns_and_options():
    system = with_void()
    plain = SurfaceMesh().compute(system)
    assert plain.regions is None and plain.surface is None
    assert plain.columns() == {}
    marked = SurfaceMesh(surface=True).compute(system)
    table = system.with_columns(marked).table
    assert table["surface"].sum() == len(marked.atoms)
    assert repr(SurfaceMesh()) == (
        "SurfaceMesh(radius=4.0, smoothing=8, regions=False, surface=False)"
    )


def test_a_smaller_probe_sees_less_solid():
    system = with_void(radius=5.0, jitter=0.1)
    large = SurfaceMesh(radius=4.0).compute(system)
    small = SurfaceMesh(radius=1.6).compute(system)
    assert small.solid_volume < large.solid_volume
    assert small.area > large.area


def test_no_atoms():
    system = build_crystal("fcc", 3.615, elements="Cu", repeat=(3, 3, 3))
    empty = system.filter(np.zeros(len(system), dtype=bool))
    result = SurfaceMesh(regions=True, surface=True).compute(empty)
    assert len(result.mesh) == 0 and result.solid_volume == 0
    assert len(result.regions.solid) == 0 and result.regions.area.dtype == float
    assert result.surface.shape == (0,)


def test_progress():
    reports = []
    SurfaceMesh().compute(
        with_void(), progress=lambda done, total: reports.append((done, total))
    )
    assert reports and reports[-1][0] == reports[-1][1]


def test_errors():
    with pytest.raises(InvalidParameter, match="replicate"):
        SurfaceMesh().compute(build_crystal("fcc", 3.615, elements="Cu"))
    for radius in (0, -1, np.inf):
        with pytest.raises(InvalidParameter, match="radius"):
            SurfaceMesh(radius)
    for smoothing in (-1, 1.5, True):
        with pytest.raises(InvalidParameter, match="smoothing"):
            SurfaceMesh(smoothing=smoothing)
