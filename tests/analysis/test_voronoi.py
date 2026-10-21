# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

import numpy as np
import pytest
from lattices import crystal

import mdapy
from mdapy.analysis import VoronoiAnalysis
from mdapy.errors import InvalidParameter

BOX = [[12.0, 0, 0], [2.0, 11.0, 0], [-1.5, 1.0, 13.0]]


def random_system(n=300, pbc=True, seed=0):
    box = mdapy.Box(BOX, origin=[1.0, -2.0, 0.5], pbc=pbc)
    rng = np.random.default_rng(seed)
    return mdapy.System(box.to_cartesian(rng.random((n, 3))), box, elements="Ar")


def faces_of(result, i):
    s = slice(result.offsets[i], result.offsets[i + 1])
    return result.neighbors[s], result.shifts[s], result.areas[s], result.vertices[s]


def test_fcc_is_a_rhombic_dodecahedron():
    a = 3.6
    result = VoronoiAnalysis().compute(crystal("fcc", a))
    np.testing.assert_allclose(result.volume, a**3 / 4, rtol=1e-12)
    np.testing.assert_allclose(result.areas, a**2 * np.sqrt(2) / 8, rtol=1e-12)
    assert (result.coordination == 12).all() and (result.vertices == 4).all()
    np.testing.assert_array_equal(result.index()[:, :2], [[0, 12]] * len(result.volume))
    np.testing.assert_allclose(result.radius, a / 2, rtol=1e-12)


def test_bcc_is_a_truncated_octahedron():
    a = 2.87
    result = VoronoiAnalysis().compute(crystal("bcc", a))
    np.testing.assert_allclose(result.volume, a**3 / 2, rtol=1e-12)
    edge = a * np.sqrt(2) / 4
    squares, hexagons = result.vertices == 4, result.vertices == 6
    np.testing.assert_allclose(result.areas[squares], edge**2, rtol=1e-12)
    np.testing.assert_allclose(
        result.areas[hexagons], 1.5 * np.sqrt(3) * edge**2, rtol=1e-12
    )
    assert squares.sum() == 6 * len(result.volume) and hexagons.sum() == 8 * len(
        result.volume
    )


@pytest.mark.parametrize(
    "kind, a, index", [("hcp", 2.5, [0, 12, 0, 0]), ("diamond", 5.43, [12, 0, 0, 4])]
)
def test_other_crystals(kind, a, index):
    system = crystal(kind, a)
    result = VoronoiAnalysis().compute(system)
    np.testing.assert_allclose(
        result.volume, system.box.volume / len(system), rtol=1e-12
    )
    np.testing.assert_array_equal(result.index(6), [index] * len(system))


@pytest.mark.parametrize("pbc", [True, (True, False, True), False])
def test_cells_fill_the_box(pbc):
    system = random_system(pbc=pbc)
    result = VoronoiAnalysis().compute(system)
    np.testing.assert_allclose(result.volume.sum(), system.box.volume, rtol=1e-12)
    walls = result.neighbors < 0
    assert walls.any() == (not np.all(pbc))
    # the walls of each closed direction add up to two faces of the box
    m = system.box.matrix
    closed = [k for k in range(3) if not system.box.pbc[k]]
    sides = sum(
        2 * np.linalg.norm(np.cross(m[(k + 1) % 3], m[(k + 2) % 3])) for k in closed
    )
    np.testing.assert_allclose(result.areas[walls].sum(), sides, rtol=1e-10)


def test_faces_are_shared():
    system = random_system(seed=1)
    result = VoronoiAnalysis().compute(system)
    faces = {}
    for i in range(len(system)):
        for j, shift, area, _ in zip(*faces_of(result, i)):
            faces[i, int(j), tuple(shift)] = area
    for (i, j, shift), area in faces.items():
        back = faces[j, i, tuple(-np.array(shift))]
        assert back == pytest.approx(area, rel=1e-9)


def test_volumes_by_sampling():
    """Points belong to the cell of the nearest atom (minimum image)."""
    system = random_system(n=40, seed=2)
    result = VoronoiAnalysis().compute(system)
    rng = np.random.default_rng(3)
    points = system.box.to_cartesian(rng.random((40000, 3)))
    images = np.array(np.meshgrid(*[[-1, 0, 1]] * 3, indexing="ij")).reshape(3, -1).T
    shifted = system.positions[None, :, :] + (images @ system.box.matrix)[:, None, :]
    shifted = shifted.reshape(-1, 3)
    owner = np.argmin(
        ((points[:, None, :] - shifted[None]) ** 2).sum(-1), axis=1
    ) % len(system)
    fraction = np.bincount(owner, minlength=len(system)) / len(points)
    expected = result.volume / system.box.volume
    sigma = np.sqrt(expected * (1 - expected) / len(points))
    assert (np.abs(fraction - expected) < 5 * sigma).all()


def test_invariance_threads_and_columns():
    system = random_system(seed=4)
    result = VoronoiAnalysis().compute(system)
    q, _ = np.linalg.qr(np.random.default_rng(5).normal(size=(3, 3)))
    turned = mdapy.System(
        system.positions @ q.T + 3.0,
        mdapy.Box(system.box.matrix @ q.T, origin=system.box.origin @ q.T + 3.0),
        elements="Ar",
    )
    np.testing.assert_allclose(
        VoronoiAnalysis().compute(turned).volume, result.volume, rtol=1e-9
    )
    with mdapy.num_threads(3):
        again = VoronoiAnalysis().compute(system)
    np.testing.assert_array_equal(again.volume, result.volume)
    np.testing.assert_array_equal(again.areas, result.areas)
    table = system.with_columns(result).table
    assert {"voronoi_volume", "voronoi_coordination", "cavity_radius"} <= set(
        table.columns
    )


def test_area_threshold():
    system = crystal("fcc", 3.6, jitter=0.03, seed=6)
    every = VoronoiAnalysis().compute(system)
    assert every.coordination.max() > 12  # slivers between second neighbors
    large = VoronoiAnalysis(area_threshold=0.01).compute(system)
    assert (large.coordination == 12).all()
    np.testing.assert_array_equal(large.volume, every.volume)
    assert (
        repr(VoronoiAnalysis(area_threshold=0.01))
        == "VoronoiAnalysis(area_threshold=0.01)"
    )
    for bad in (-0.1, 1.0):
        with pytest.raises(InvalidParameter, match="area_threshold"):
            VoronoiAnalysis(area_threshold=bad)


def test_one_atom_and_atoms_outside_the_box():
    single = mdapy.System([[0.5, 0.5, 0.5]], mdapy.Box([3.0, 4.0, 5.0]), elements="Ar")
    result = VoronoiAnalysis().compute(single)
    assert result.volume[0] == pytest.approx(60.0, rel=1e-12)
    assert result.coordination[0] == 6 and (result.neighbors == 0).all()
    outside = mdapy.System(
        [[1.0, 1, 1], [2.0, 2, 7.0]],
        mdapy.Box(5.0, pbc=(True, True, False)),
        elements="Ar",
    )
    result = VoronoiAnalysis().compute(outside)
    # the closed direction reaches out to the atom at z = 7
    assert result.volume.sum() == pytest.approx(25.0 * 7.0, rel=1e-12)
