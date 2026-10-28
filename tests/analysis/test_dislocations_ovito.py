# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The dislocation analysis against that of OVITO 3.17.1, on dislocations
built from elastic fields or relaxed to one state. Skipped where OVITO is
not installed."""

import numpy as np
import pytest

import mdapy
from mdapy.analysis import DislocationAnalysis
from mdapy.build import (
    build_crystal,
    dislocation_cylinder,
    dislocation_quadrupole,
    insert_dislocation,
    twin_lamellae,
)
from mdapy.calculators import EAM
from mdapy.simulation import minimize
from mdapy.transform import replicate

from dislocation_models import shear_loop

ovito = pytest.importorskip("ovito")
from ovito.modifiers import DislocationAnalysisModifier  # noqa: E402

from ovito_reference import ovito_compute  # noqa: E402

LATTICE = {
    "fcc": DislocationAnalysisModifier.Lattice.FCC,
    "hcp": DislocationAnalysisModifier.Lattice.HCP,
    "bcc": DislocationAnalysisModifier.Lattice.BCC,
    "cubic_diamond": DislocationAnalysisModifier.Lattice.CubicDiamond,
    "hexagonal_diamond": DislocationAnalysisModifier.Lattice.HexagonalDiamond,
}
# OVITO's names of the hexagonal families in its attributes
SPELLING = {
    "1/3<11-20>": "1/3<11-20>",
    "<1-100>": "<-1100>",
    "1/3<1-100>": "1/3<-1100>",
}

CU = (168.4, 121.4, 75.4)
FE = (243.3, 145.0, 116.1)
SI = (165.7, 63.9, 79.6)
PLANE_FCC = [(1, -1, 0), (1, 1, -2), (1, 1, 1)]


def assert_same_vectors(ours, theirs, atol):
    """The same vectors in some order."""
    left = [np.asarray(v, dtype=float) for v in theirs]
    assert len(ours) == len(left)
    for v in ours:
        match = [k for k, w in enumerate(left) if np.allclose(v, w, atol=atol)]
        assert match, f"{v} not among {left}"
        left.pop(match[0])


def oriented(burgers, points, loop, infinite):
    """The Burgers vector of a line, of a closed loop along the loop run
    counterclockwise about the axis it encloses the most area across, which
    the analysis does not fix."""
    if loop and not infinite:
        area = np.cross(points[:-1] - points[0], points[1:] - points[0]).sum(axis=0)
        if area[np.argmax(np.abs(area))] < 0:
            return -np.asarray(burgers)
    return np.asarray(burgers)


def compare(system, crystal, rel, lines=True):
    """Our analysis and OVITO's of a system: the same structures atom by
    atom, lengths of each family within rel, and, with lines, lines of the
    same Burgers vectors and the same loops."""
    ours = DislocationAnalysis(crystal).compute(system)
    data = ovito_compute(
        system, DislocationAnalysisModifier(input_crystal_structure=LATTICE[crystal])
    )
    np.testing.assert_array_equal(
        ours.structure, np.asarray(data.particles["Structure Type"])
    )
    theirs = list(data.dislocations.lines)
    if lines:
        assert_same_vectors(
            [
                oriented(line.spatial_burgers, line.points, line.loop, line.infinite)
                for line in ours.lines
            ],
            [
                oriented(
                    line.spatial_burgers_vector,
                    np.asarray(line.points),
                    line.is_loop,
                    line.is_infinite_line,
                )
                for line in theirs
            ],
            # the frame of the Burgers vector of a line is that of a cluster
            # its first circuit touches, which the circuits' choices pick,
            # and clusters of a distorted crystal are turned a little apart
            atol=0.1,
        )
        assert sorted((line.loop, line.infinite) for line in ours.lines) == sorted(
            (line.is_loop, line.is_infinite_line) for line in theirs
        )
    attributes = data.attributes
    for family, length in ours.lengths().items():
        assert length == pytest.approx(
            attributes[f"DislocationAnalysis.length.{SPELLING.get(family, family)}"],
            rel=rel,
            abs=0.5,
        )
    return ours, data


QUADRUPOLES = {
    "bcc screw": ("bcc", ("bcc screw", 2.865, FE, (6, 10, 5)), {"elements": "Fe"}),
    "bcc edge": ("bcc", ("bcc edge", 2.865, FE, (16, 10, 2)), {"elements": "Fe"}),
    "fcc screw": ("fcc", ("fcc screw", 3.615, CU, (6, 10, 4)), {"elements": "Cu"}),
    "fcc edge, dissociated": (
        "fcc",
        ("fcc edge", 3.615, CU, (14, 8, 3)),
        {"elements": "Cu", "dissociation": 10},
    ),
    "diamond screw": (
        "cubic_diamond",
        ("diamond screw", 5.431, SI, (6, 10, 3)),
        {"elements": "Si"},
    ),
    "diamond 60": (
        "cubic_diamond",
        ("diamond 60", 5.431, SI, (8, 6, 3)),
        {"elements": "Si"},
    ),
    "diamond 60, dissociated": (
        "cubic_diamond",
        ("diamond 60", 5.431, SI, (8, 6, 3)),
        {"elements": "Si", "dissociation": 12},
    ),
}


@pytest.mark.parametrize("name", list(QUADRUPOLES))
def test_quadrupoles(name):
    """Straight lines through the periodic box, partials among them."""
    crystal, args, options = QUADRUPOLES[name]
    ours, _ = compare(dislocation_quadrupole(*args, **options), crystal, rel=0.005)
    assert all(line.infinite for line in ours.lines)


def cubic_loops():
    cu = build_crystal(
        "fcc", 3.615, elements="Cu", orientation=PLANE_FCC, repeat=(12, 7, 6)
    )
    center = cu.box.matrix.diagonal() / 2 + [0.1, 0.1, 0.05]
    fe = build_crystal(
        "bcc",
        2.87,
        elements="Fe",
        orientation=[(1, 1, -2), (-1, 1, 0), (1, 1, 1)],
        repeat=(8, 11, 6),
    )
    center_fe = fe.box.matrix.diagonal() / 2 + [0.1, 0.05, 0.1]
    a = 3.615
    return {
        "perfect fcc loop": (
            "fcc",
            shear_loop(cu, [a / np.sqrt(2), 0, 0], center, 14.0),
        ),
        "shockley loop": ("fcc", shear_loop(cu, [0, -a / np.sqrt(6), 0], center, 14.0)),
        "bcc loop": (
            "bcc",
            shear_loop(
                fe, [0, 0, 2.87 * np.sqrt(3) / 2], center_fe, 11.0, normal=[0, 1, 0]
            ),
        ),
    }


def crossing_loops():
    """Two loops on crossing {111} planes of copper whose Burgers vectors
    add up to a perfect one: a glissile junction where they meet."""
    a = 3.615
    cu = build_crystal("fcc", a, elements="Cu", repeat=(12, 12, 12))
    center = cu.box.matrix.diagonal() / 2 + [0.11, 0.07, 0.03]
    first = shear_loop(cu, a / 2 * np.array([1, -1, 0]), center, 13.0, normal=[1, 1, 1])
    return shear_loop(
        first, a / 2 * np.array([0, 1, 1]), center, 13.0, normal=[1, 1, -1]
    )


@pytest.mark.parametrize("name", ["perfect fcc loop", "shockley loop", "bcc loop"])
def test_loops(name):
    crystal, system = cubic_loops()[name]
    compare(system, crystal, rel=0.03)


def test_crossing_loops():
    """Where two loops cross, a junction of their Burgers vectors' sum, or
    a node of four lines: the circuits' choices between the two depend on
    the order of the atoms, in OVITO as here; the lengths do not."""
    ours, _ = compare(crossing_loops(), "fcc", rel=0.03, lines=False)
    allowed = 3.615 / 2 * np.array([[1, -1, 0], [0, 1, 1], [1, 0, 1]])
    for line in ours.lines:
        assert line.family == "1/2<110>"
        assert np.isclose(
            np.abs(allowed @ line.spatial_burgers).max(), 3.615**2 / 2, atol=0.05
        )
    assert len(ours.junctions) >= 2


def hexagonal(structure, a, element, kind):
    """A straight basal dislocation of a hexagonal crystal in a box open
    across the line: a screw along a1 or an edge of Burgers vector a1."""
    if kind == "screw":
        orientation, repeat, burgers = (
            [(0, 1, -1, 0), (0, 0, 0, 1), (2, -1, -1, 0)],
            (9, 5, 4),
            [0, 0, a],
        )
    else:
        orientation, repeat, burgers = (
            [(2, -1, -1, 0), (0, 0, 0, 1), (0, 1, -1, 0)],
            (14, 5, 4),
            [a, 0, 0],
        )
    cell = build_crystal(
        structure, a, elements=element, orientation=orientation, repeat=repeat
    )
    cell = cell.with_box(mdapy.Box(cell.box.matrix, pbc=(False, False, True)))
    return insert_dislocation(
        cell, burgers, center=cell.box.matrix.diagonal() / 2 + [0.3, 0.2, 0]
    )


@pytest.mark.parametrize(
    "crystal, structure, a, element, kind",
    [
        ("hcp", "hcp", 3.2, "Mg", "screw"),
        ("hcp", "hcp", 3.2, "Mg", "edge"),
        ("hexagonal_diamond", "lonsdaleite", 2.51, "C", "screw"),
    ],
)
def test_hexagonal_crystals(crystal, structure, a, element, kind):
    ours, _ = compare(hexagonal(structure, a, element, kind), crystal, rel=0.005)
    assert [line.family for line in ours.lines] == ["1/3<11-20>"]


def test_a_loop_on_a_twin_boundary():
    """Clusters across a twin boundary meet through the boundary layer."""
    cell = build_crystal("fcc", 3.615, elements="Cu", orientation=PLANE_FCC)
    twins = replicate(twin_lamellae(cell, 6), (12, 7, 2))
    center = twins.box.matrix.diagonal() / 2 + [0.1, 0.1, 0.05]
    center[2] = twins.box.matrix[2, 2] / 4 + 0.05
    ours, _ = compare(
        shear_loop(twins, [3.615 / np.sqrt(2), 0, 0], center, 12.0), "fcc", 0.01
    )
    assert [line.loop for line in ours.lines] == [True]


def test_a_triclinic_box():
    """The quadrupole turned with its box."""
    crystal, args, options = QUADRUPOLES["fcc edge, dissociated"]
    system = dislocation_quadrupole(*args, **options)
    t = np.radians(25)
    turn = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
    matrix = system.box.matrix.copy()
    matrix[1] += 0.3 * matrix[0]  # sheared, the atoms with it
    fractional = system["positions"] @ system.box.inverse
    turned = system.with_box(mdapy.Box(matrix @ turn.T, pbc=True)).with_columns(
        positions=fractional @ matrix @ turn.T
    )
    compare(turned, crystal, rel=0.01)


def test_a_frank_loop():
    """A relaxed disc of vacancies: Shockley partials and stair-rods on the
    edges of a stacking-fault tetrahedron, meeting at junctions."""
    cu = build_crystal(
        "fcc", 3.615, elements="Cu", orientation=PLANE_FCC, repeat=(20, 12, 5)
    )
    p = cu["positions"] - cu.box.matrix.diagonal() / 2
    plane = p[np.argmin(np.abs(p[:, 2])), 2]
    disc = (np.abs(p[:, 2] - plane) < 0.3) & (np.linalg.norm(p[:, :2], axis=1) < 14)
    relaxed = minimize(
        cu.filter(~disc), EAM.zhou(["Cu"]), max_force=1e-2, max_steps=3000
    )
    ours, _ = compare(relaxed.system, "fcc", rel=0.03)
    assert len(ours.junctions) == 6


def test_the_defect_mesh_of_free_surfaces():
    """A cylinder with free surfaces around a dissociated dislocation, its
    atoms shaken: the same defect mesh as OVITO's."""
    system = dislocation_cylinder(
        "fcc edge", 3.615, CU, 22.0, elements="Cu", length=3, dissociation=8
    )
    noise = np.random.default_rng(5).normal(0, 0.03, (len(system), 3))
    system = system.with_columns(positions=system["positions"] + noise)
    ours, data = compare(system, "fcc", rel=0.03)
    mesh = data.surfaces["dxa-defect-mesh"]
    vertices = np.asarray(mesh.vertices["Position"])
    faces = np.asarray(mesh.get_face_vertices()).reshape(-1, 3)
    box = system.box
    sides = []
    for k in (1, 2):
        fraction = (vertices[faces[:, k]] - vertices[faces[:, 0]]) @ box.inverse
        fraction -= np.where(box.pbc, np.round(fraction), 0.0)
        sides.append(fraction @ box.matrix)
    area = 0.5 * np.linalg.norm(np.cross(*sides), axis=1).sum()
    assert len(ours.defect_mesh.faces) == len(faces)
    assert ours.defect_mesh.areas.sum() == pytest.approx(area, rel=1e-6)
