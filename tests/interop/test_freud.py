# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

import numpy as np
import pytest

import mdapy
from mdapy.errors import InvalidParameter
from mdapy.interop import from_freud, to_freud

freud = pytest.importorskip("freud")


def general_system(seed=1, n=300):
    """Atoms in a triclinic box turned to no particular orientation, with an
    origin, some of them outside the box."""
    rng = np.random.default_rng(seed)
    matrix = np.array([[9.0, 0.5, 0.3], [1.2, 8.5, 0.4], [0.7, 0.9, 10.0]])
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rotation[:, 0] *= np.sign(np.linalg.det(rotation))
    matrix = matrix @ rotation
    origin = np.array([3.0, -2.0, 1.0])
    positions = rng.random((n, 3)) @ matrix + origin + rng.normal(0, 2, (n, 3))
    return mdapy.System(positions, mdapy.Box(matrix, origin=origin), elements="Ar")


def pairs(i, j, distances):
    return {(int(a), int(b)): float(d) for a, b, d in zip(i, j, distances)}


def test_freud_finds_the_neighbors_mdapy_finds():
    system = general_system()
    box, points = to_freud(system)
    assert points.dtype == np.float32 and points.shape == (300, 3)
    assert box.volume == pytest.approx(system.box.volume, rel=1e-6)
    query = freud.locality.AABBQuery(box, points)
    found = query.query(points, {"r_max": 3.0, "exclude_ii": True}).toNeighborList()
    theirs = pairs(found.query_point_indices, found.point_indices, found.distances)
    neighbors = mdapy.NeighborList.build(system, 3.0)
    ours = pairs(neighbors.i, neighbors.j, neighbors.distances)
    assert theirs.keys() == ours.keys()
    # freud computes in single precision
    assert max(abs(theirs[k] - ours[k]) for k in ours) < 1e-5


def test_points_lie_in_the_box_centered_on_the_origin():
    box, points = to_freud(general_system(seed=2))
    fractional = box.make_fractional(points)
    assert fractional.min() > -1e-6 and fractional.max() < 1 + 1e-6
    matrix = box.to_matrix()  # vectors as columns
    assert matrix[1, 0] == matrix[2, 0] == matrix[2, 1] == 0  # a along x, b in xy


def test_back_from_freud():
    system = general_system(seed=3)
    box, points = to_freud(system)
    back = from_freud(box, points)
    assert back.box.volume == pytest.approx(system.box.volume, rel=1e-6)
    np.testing.assert_allclose(back.box.origin, -0.5 * back.box.matrix.sum(axis=0))
    # the same atoms in a turned box: the same distances
    ours = mdapy.NeighborList.build(system, 3.0)
    again = mdapy.NeighborList.build(back, 3.0)
    first, second = (
        pairs(ours.i, ours.j, ours.distances),
        pairs(again.i, again.j, again.distances),
    )
    assert first.keys() == second.keys()
    assert max(abs(first[k] - second[k]) for k in first) < 1e-5


def test_a_box_of_freud():
    back = from_freud(freud.box.Box.cube(5.0), [[0.0, 0.0, 0.0], [1.0, 2.0, 2.4]])
    np.testing.assert_allclose(back.box.matrix, 5.0 * np.eye(3))
    np.testing.assert_allclose(back.box.origin, [-2.5, -2.5, -2.5])
    assert back.box.pbc.all()
    np.testing.assert_allclose(back["positions"][1], [1.0, 2.0, 2.4])


def test_boxes_freud_cannot_take():
    slab = mdapy.System([[0.5, 0.5, 0.5]], mdapy.Box(4.0, pbc=[True, True, False]))
    with pytest.raises(InvalidParameter, match="periodic in every direction"):
        to_freud(slab)
    with pytest.raises(InvalidParameter, match="three-dimensional"):
        from_freud(freud.box.Box.square(5.0), [[0.0, 0.0, 0.0]])
    left = mdapy.Box([[0, 4.0, 0], [4.0, 0, 0], [0, 0, 4.0]])
    with pytest.raises(InvalidParameter, match="right-handed"):
        to_freud(mdapy.System([[1.0, 1.0, 1.0]], left))
