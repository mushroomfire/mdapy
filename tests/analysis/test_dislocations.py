# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import pytest

import mdapy
from mdapy.analysis import DislocationAnalysis
from mdapy.build import (
    build_crystal,
    dislocation_quadrupole,
    insert_dislocation,
)
from mdapy.calculators import EAM
from mdapy.errors import Cancelled, InvalidParameter
from mdapy.simulation import minimize

from dislocation_models import shear_loop

CU = (168.4, 121.4, 75.4)
FE = (243.3, 145.0, 116.1)
PLANE_FCC = [(1, -1, 0), (1, 1, -2), (1, 1, 1)]


def shaken(system, sigma, seed=0):
    noise = np.random.default_rng(seed).normal(0, sigma, (len(system), 3))
    return system.with_columns(positions=system["positions"] + noise)


def edges_of(mesh):
    """Each edge of a mesh as (vertex, vertex, image of the second from the
    first), once per triangle it borders, in its direction there."""
    faces, shifts = mesh.faces, mesh.shifts.astype(int)
    edges = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        edges.append(
            np.column_stack([faces[:, a], faces[:, b], shifts[:, b] - shifts[:, a]])
        )
    return np.concatenate(edges)


def assert_closed(mesh):
    """Every edge borders two triangles, run in opposite directions."""
    edges = edges_of(mesh)
    reverse = np.column_stack([edges[:, 1], edges[:, 0], -edges[:, 2:]])
    forward = {tuple(e) for e in edges}
    assert len(forward) == len(edges)
    assert forward == {tuple(e) for e in reverse}


def frank_rule(result):
    """The sum of the Burgers vectors of the lines into each junction, in Å."""
    into = np.zeros((len(result.junctions), 3))
    for line in result.lines:
        start, end = line.junctions
        if end >= 0:
            into[end] += line.spatial_burgers
        if start >= 0:
            into[start] -= line.spatial_burgers
    return into


def crossing_loops():
    """Two loops on crossing {111} planes of copper whose Burgers vectors
    add up to a perfect one."""
    a = 3.615
    cu = build_crystal("fcc", a, elements="Cu", repeat=(12, 12, 12))
    center = cu.box.matrix.diagonal() / 2 + [0.11, 0.07, 0.03]
    first = shear_loop(cu, a / 2 * np.array([1, -1, 0]), center, 13.0, normal=[1, 1, 1])
    return shear_loop(
        first, a / 2 * np.array([0, 1, 1]), center, 13.0, normal=[1, 1, -1]
    )


@pytest.mark.parametrize(
    "crystal, structure, a",
    [
        ("fcc", "fcc", 3.615),
        ("hcp", "hcp", 3.21),
        ("bcc", "bcc", 2.865),
        ("cubic_diamond", "diamond", 5.431),
        ("hexagonal_diamond", "lonsdaleite", 2.51),
    ],
)
def test_a_hot_crystal_has_no_dislocations(crystal, structure, a):
    """Every atom of a shaken crystal is of it, in one cluster whose frame
    is that of the box, with no lines and no defects."""
    system = shaken(build_crystal(structure, a, repeat=(6, 6, 6)), 0.02 * a)
    result = DislocationAnalysis(crystal, interface_mesh=True).compute(system)
    assert result.lines == () and result.junctions.shape == (0, 3)
    assert (np.asarray(result.names)[result.structure] == crystal).all()
    assert (result.crystal_cluster == 0).all()
    np.testing.assert_array_equal(result.clusters.size, [len(system)])
    orientation = result.clusters.orientation[0]
    scale = np.cbrt(np.linalg.det(orientation))
    np.testing.assert_allclose(orientation / scale, np.eye(3), atol=0.02)
    assert len(result.defect_mesh) == 0 and len(result.interface_mesh) == 0
    assert result.total_length == 0
    assert set(result.lengths().values()) == {0.0}


@pytest.mark.parametrize(
    "kind, a, elastic, repeat, element, crystal, family, count",
    [
        ("bcc screw", 2.865, FE, (6, 10, 5), "Fe", "bcc", "1/2<111>", 2),
        ("bcc edge", 2.865, FE, (16, 10, 2), "Fe", "bcc", "1/2<111>", 2),
        ("fcc screw", 3.615, CU, (6, 10, 4), "Cu", "fcc", "1/2<110>", 2),
        ("fcc edge", 3.615, CU, (14, 8, 3), "Cu", "fcc", "1/2<110>", 2),
    ],
)
def test_straight_lines_through_the_box(
    kind, a, elastic, repeat, element, crystal, family, count
):
    """Each dislocation of a quadrupole runs straight along z through the
    periodic box: a line as long as the box, of the Burgers vector put in,
    opposite for the two of a dipole."""
    system = dislocation_quadrupole(kind, a, elastic, repeat, elements=element)
    result = DislocationAnalysis(crystal).compute(system)
    assert len(result.lines) == count
    length = system.box.matrix[2, 2]
    for line in result.lines:
        assert line.family == family and line.loop and line.infinite
        assert line.junctions == (-1, -1)
        assert line.length == pytest.approx(length, rel=0.02)
        np.testing.assert_allclose(
            line.points[-1] - line.points[0], [0, 0, length], atol=1e-6
        )
        expected = a / 2 * np.sqrt(3 if crystal == "bcc" else 2)
        assert np.linalg.norm(line.spatial_burgers) == pytest.approx(expected, rel=0.01)
    np.testing.assert_allclose(
        sum(line.spatial_burgers for line in result.lines), 0, atol=1e-6
    )
    assert result.lengths()[family] == pytest.approx(result.total_length)


def test_partials():
    """A dissociated edge dislocation of copper: two Shockley partials that
    add up to the perfect Burgers vector, the stacking fault between them
    of hcp atoms; or, of perfect dislocations only, one line."""
    system = dislocation_quadrupole(
        "fcc edge", 3.615, CU, (14, 8, 3), elements="Cu", dissociation=10
    )
    result = DislocationAnalysis("fcc").compute(system)
    assert [line.family for line in result.lines] == ["1/6<112>"] * 4
    assert (np.asarray(result.names)[result.structure] == "hcp").sum() > 0
    # the partner of a partial lies closest to it across z
    box = system.box
    places = np.array([line.points[0] @ box.inverse for line in result.lines])
    for k, line in enumerate(result.lines):
        gap = places - places[k]
        gap -= np.round(gap)
        distance = np.linalg.norm((gap @ box.matrix)[:, :2], axis=1)
        distance[k] = np.inf
        partner = result.lines[int(np.argmin(distance))]
        total = line.burgers + partner.burgers
        assert np.allclose(np.sort(np.abs(total)), [0, 0.5, 0.5], atol=1e-6)
    perfect = DislocationAnalysis("fcc", perfect_only=True).compute(system)
    assert [line.family for line in perfect.lines] == ["1/2<110>"] * 2
    assert (np.asarray(perfect.names)[perfect.structure] == "hcp").sum() == 0


def test_frank_rule_at_junctions():
    """Where lines meet, the Burgers vectors going in add up to those going
    out: crossing loops and a stacking-fault tetrahedron."""
    crossing = DislocationAnalysis("fcc").compute(crossing_loops())
    cu = build_crystal(
        "fcc", 3.615, elements="Cu", orientation=PLANE_FCC, repeat=(14, 8, 4)
    )
    p = cu["positions"] - cu.box.matrix.diagonal() / 2
    plane = p[np.argmin(np.abs(p[:, 2])), 2]
    disc = (np.abs(p[:, 2] - plane) < 0.3) & (np.linalg.norm(p[:, :2], axis=1) < 10)
    relaxed = minimize(
        cu.filter(~disc), EAM.zhou(["Cu"]), max_force=1e-2, max_steps=3000
    )
    tetrahedron = DislocationAnalysis("fcc").compute(relaxed.system)
    for result in (crossing, tetrahedron):
        assert len(result.junctions) >= 2
        np.testing.assert_allclose(frank_rule(result), 0, atol=0.02)
        for line in result.lines:
            for end, point in zip(line.junctions, (line.points[0], line.points[-1])):
                if end >= 0:  # the line ends at its junction, in some image
                    gap = (point - result.junctions[end]) @ relaxed.system.box.inverse
                    np.testing.assert_allclose(gap, np.round(gap), atol=1e-6)
    assert {line.family for line in tetrahedron.lines} == {"1/6<112>", "1/6<110>"}


def test_the_same_for_any_number_of_threads():
    """The circuits' choices are fixed: the same lines on every run."""
    system = crossing_loops()
    found = []
    for threads in (1, 4, 1):
        with mdapy.num_threads(threads):
            result = DislocationAnalysis("fcc").compute(system)
        found.append(np.concatenate([line.points.ravel() for line in result.lines]))
    for points in found[1:]:
        np.testing.assert_array_equal(points, found[0])


def test_the_sense_of_a_loop():
    """A loop with the opposite Burgers vector runs the other way round."""
    a = 3.615
    cu = build_crystal(
        "fcc", a, elements="Cu", orientation=PLANE_FCC, repeat=(16, 9, 6)
    )
    center = cu.box.matrix.diagonal() / 2 + [0.1, 0.1, 0.05]
    b = np.array([a / np.sqrt(2), 0, 0])
    found = []
    for sign in (1, -1):
        result = DislocationAnalysis("fcc").compute(
            shear_loop(cu, sign * b, center, 14.0)
        )
        (line,) = result.lines
        assert line.loop and not line.infinite and line.family == "1/2<110>"
        assert line.length == pytest.approx(2 * np.pi * 14.0, rel=0.05)
        area = np.cross(line.points[:-1] - center, line.points[1:] - center).sum(axis=0)
        found.append(np.sign(area[2]) * line.spatial_burgers)
    np.testing.assert_allclose(found[0], -found[1], atol=0.01)
    np.testing.assert_allclose(np.abs(found[0]), np.abs(b), atol=0.01)


def test_the_sign_of_the_burgers_vector():
    """As OVITO's, the Burgers vector insert_dislocation puts in, along
    +z."""
    b = 3.615 / np.sqrt(2)
    for orientation, burgers in [
        ([(1, 1, -2), (1, 1, 1), (1, -1, 0)], [0, 0, b]),  # screw
        ([(1, 1, -2), (1, 1, 1), (1, -1, 0)], [0, 0, -b]),
        ([(1, -1, 0), (1, 1, 1), (-1, -1, 2)], [b, 0, 0]),  # edge
    ]:
        cell = build_crystal(
            "fcc", 3.615, elements="Cu", orientation=orientation, repeat=(8, 5, 4)
        )
        cell = cell.with_box(mdapy.Box(cell.box.matrix, pbc=(False, False, True)))
        center = cell.box.matrix.diagonal() / 2 + [0.3, 0.2, 0]
        system = insert_dislocation(cell, burgers, center=center)
        result = DislocationAnalysis("fcc", perfect_only=True).compute(system)
        (line,) = result.lines
        assert line.points[-1, 2] > line.points[0, 2]  # along +z
        np.testing.assert_allclose(line.spatial_burgers, burgers, atol=0.02)


def test_invariance():
    """Moving all atoms, through the periodic boundaries, moves the lines;
    other orders of the atoms and turns of the box give lines as long, of
    the same Burgers vectors."""
    system = dislocation_quadrupole(
        "fcc edge", 3.615, CU, (14, 8, 3), elements="Cu", dissociation=10
    )
    analysis = DislocationAnalysis("fcc")
    base = analysis.compute(system)
    shift = np.array([7.3, -4.1, 2.2])
    moved = analysis.compute(system.with_columns(positions=system["positions"] + shift))
    assert moved.total_length == pytest.approx(base.total_length, rel=0.02)
    np.testing.assert_allclose(
        sorted(tuple(line.burgers) for line in moved.lines),
        sorted(tuple(line.burgers) for line in base.lines),
        rtol=1e-6,
        atol=1e-12,
    )
    order = np.random.default_rng(3).permutation(len(system))
    shuffled = analysis.compute(system.gather(order))
    assert shuffled.total_length == pytest.approx(base.total_length, rel=0.02)
    np.testing.assert_array_equal(shuffled.structure, base.structure[order])
    t = np.radians(40)
    turn = np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
    turned = system.with_box(
        mdapy.Box(system.box.matrix @ turn.T, pbc=True)
    ).with_columns(positions=system["positions"] @ turn.T)
    result = analysis.compute(turned)
    assert result.total_length == pytest.approx(base.total_length, rel=0.02)
    expected = sorted(
        tuple(np.round(turn @ line.spatial_burgers, 2)) for line in base.lines
    )
    assert (
        sorted(tuple(np.round(line.spatial_burgers, 2)) for line in result.lines)
        == expected
    )


def test_free_surfaces_and_their_mesh():
    """A slab with free surfaces: the defect mesh is the two surfaces,
    closed through the periodic directions; a screw dislocation across it
    ends on them, the mesh closed over its ends."""
    slab = build_crystal("fcc", 3.615, elements="Cu", repeat=(8, 8, 8))
    slab = slab.with_box(mdapy.Box(slab.box.matrix, pbc=(True, True, False)))
    result = DislocationAnalysis("fcc").compute(shaken(slab, 0.03))
    assert result.lines == ()
    area = slab.box.matrix[0, 0] * slab.box.matrix[1, 1]
    assert result.defect_mesh.areas.sum() == pytest.approx(2 * area, rel=0.05)
    assert_closed(result.defect_mesh)
    assert (result.defect_atoms >= 0).all()
    plate = build_crystal(
        "fcc",
        3.615,
        elements="Cu",
        orientation=[(1, 1, -2), (1, 1, 1), (1, -1, 0)],
        repeat=(8, 5, 6),
    )
    plate = plate.with_box(mdapy.Box(plate.box.matrix, pbc=False))
    center = plate.box.matrix.diagonal() / 2 + [0.3, 0.2, 0]
    screw = insert_dislocation(plate, [0, 0, 3.615 / np.sqrt(2)], center=center)
    result = DislocationAnalysis("fcc", mesh_smoothing=0, interface_mesh=True).compute(
        screw
    )
    (line,) = result.lines
    assert line.family == "1/2<110>" and not line.loop and line.junctions == (-1, -1)
    assert line.length == pytest.approx(screw.box.matrix[2, 2], rel=0.15)
    assert_closed(result.defect_mesh)
    assert_closed(result.interface_mesh)
    assert (result.defect_atoms == -1).sum() == 2  # the two ends
    assert len(result.interface_mesh) > len(result.defect_mesh) - 2 * 20


def test_options():
    system = dislocation_quadrupole("fcc screw", 3.615, CU, (6, 10, 4), elements="Cu")
    raw = DislocationAnalysis("fcc", point_separation=0, smoothing=0).compute(system)
    smooth = DislocationAnalysis("fcc").compute(system)
    assert len(raw.lines) == len(smooth.lines)
    for r, s in zip(raw.lines, smooth.lines):
        assert len(r.points) > len(s.points) and r.length > s.length
    # the compact cores of screws, short circuits find as well
    short = DislocationAnalysis("fcc", circuit_length=3, stretch=0).compute(system)
    assert short.total_length == pytest.approx(smooth.total_length, rel=0.02)
    flat = DislocationAnalysis("fcc", mesh_smoothing=0).compute(system)
    assert len(flat.defect_mesh) == 0


def test_results():
    system = dislocation_quadrupole("bcc screw", 2.865, FE, (6, 10, 5), elements="Fe")
    result = DislocationAnalysis("bcc").compute(system)
    assert set(result.columns()) == {"structure", "crystal_cluster"}
    assert list(result.lengths()) == ["1/2<111>", "<100>", "<110>", "other"]
    assert result.total_length == pytest.approx(
        sum(line.length for line in result.lines)
    )
    assert result.names[3] == "bcc" and len(result.colors) == len(result.names)
    assert result.counts() == {
        name: int((result.structure == k).sum()) for k, name in enumerate(result.names)
    }
    assert "crystal_cluster" in system.with_columns(result).table.columns
    assert result.interface_mesh is None and result.interface_atoms is None
    with pytest.raises(ValueError):
        result.structure[0] = 0
    with pytest.raises(ValueError):
        result.lines[0].points[0, 0] = 0
    assert repr(DislocationAnalysis("bcc")) == (
        "DislocationAnalysis('bcc', circuit_length=14, stretch=9, perfect_only=False, "
        "point_separation=2.5, smoothing=1, mesh_smoothing=8, interface_mesh=False)"
    )


def test_families_of_the_hexagonal_lattices():
    """Burgers vectors of hcp in its frame, c along z and a1 along x, by
    their Miller-Bravais indices."""
    from mdapy.analysis._dislocations import _family

    a, c = np.sqrt(0.5), np.sqrt(4 / 3)
    a1 = np.array([a, 0, 0])
    a2 = np.array([-a / 2, a * np.sqrt(3) / 2, 0])
    cases = {
        "1/3<11-20>": a1,
        "<0001>": np.array([0, 0, c]),
        "<1-100>": a1 + 2 * a2,
        "1/3<1-100>": (a1 + 2 * a2) / 3,
        "1/3<11-23>": a2 + np.array([0, 0, -c]),
        "other": 2 * a1,
    }
    for family, burgers in cases.items():
        assert _family(burgers, "hcp") == family
    assert _family(cases["1/3<11-23>"], "hexagonal_diamond") == "other"
    assert _family(np.array([0.5, 0.5, 0.5]), "fcc") == "other"


def test_no_atoms_and_few_atoms():
    system = build_crystal("fcc", 3.615, elements="Cu", repeat=(3, 3, 3))
    empty = DislocationAnalysis("fcc").compute(
        system.filter(np.zeros(len(system), dtype=bool))
    )
    assert empty.lines == () and empty.structure.shape == (0,)
    assert len(empty.defect_mesh) == 0
    few = system.gather(np.arange(10)).with_box(mdapy.Box(30.0, pbc=False))
    found = DislocationAnalysis("fcc").compute(few)
    assert (found.structure == 0).all() and found.lines == ()


def test_progress_and_cancel():
    system = dislocation_quadrupole("bcc screw", 2.865, FE, (6, 10, 5), elements="Fe")
    calls = []
    DislocationAnalysis("bcc").compute(
        system, progress=lambda done, total: calls.append((done, total))
    )
    assert calls and calls[-1][0] == calls[-1][1]
    token = mdapy.CancelToken()
    token.cancel()
    with pytest.raises(Cancelled):
        DislocationAnalysis("bcc").compute(system, cancel=token)


def test_errors():
    with pytest.raises(InvalidParameter, match="crystal"):
        DislocationAnalysis("sc")
    for name, value in [
        ("circuit_length", 2),
        ("circuit_length", 4.5),
        ("stretch", -1),
        ("smoothing", True),
        ("mesh_smoothing", -1),
    ]:
        with pytest.raises(InvalidParameter, match=name):
            DislocationAnalysis("fcc", **{name: value})
    for value in (-1, np.inf):
        with pytest.raises(InvalidParameter, match="point_separation"):
            DislocationAnalysis("fcc", point_separation=value)
    for thickness, error in [(1, "own periodic image"), (2, "half the box")]:
        system = dislocation_quadrupole(
            "bcc screw", 2.865, FE, (6, 10, thickness), elements="Fe"
        )
        with pytest.raises(InvalidParameter, match=error):
            DislocationAnalysis("bcc").compute(system)
    with pytest.raises(InvalidParameter, match="replicate"):
        DislocationAnalysis("fcc").compute(build_crystal("fcc", 3.615, elements="Cu"))
