# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""The k-d tree that NeighborList uses, searched for arbitrary points."""

import itertools
import math

import numpy as np
import pytest
from mdapy._core import neighbor as _native

import mdapy

BOXES = {
    "orthogonal": [[5.0, 0, 0], [0, 6.0, 0], [0, 0, 7.0]],
    "triclinic": [[5.0, 0, 0], [2.0, 4.5, 0], [-1.5, 1.0, 6.0]],
    "rotated": [[3.0, 4.0, 0], [-1.0, 3.0, 2.0], [0.5, -1.0, 5.0]],
    "left-handed": [[5.0, 0, 0], [0, 0, 6.0], [0, 6.0, 0]],
    "flat": [[9.0, 0, 0], [0, 9.0, 0], [0, 0, 2.0]],
}


def tree_of(system):
    box = system.box
    return _native.KdTree(
        np.ascontiguousarray(system.positions),
        np.ascontiguousarray(box.matrix),
        np.ascontiguousarray(box.origin),
        np.ascontiguousarray(box.pbc),
    )


def brute_points(system, points, k, reach=4):
    """The k nearest atom images of each point, ordered by distance, then
    atom, then image, with the arithmetic of the compiled code."""
    box = system.box
    h = box.matrix
    x = system.positions
    rows = []
    for q in points:
        # around the image of the box that holds the point
        home = np.floor(box.to_fractional(q[None])[0]).astype(int)
        ranges = [
            range(c - reach, c + reach + 1) if p else range(1)
            for c, p in zip(home, box.pbc)
        ]
        found = []
        for shift in itertools.product(*ranges):
            r = x - q
            r = r + shift[0] * h[0]
            r = r + shift[1] * h[1]
            r = r + shift[2] * h[2]
            r2 = r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2]
            found += [(r2[j], j, shift) for j in range(len(x))]
        found.sort()
        rows.append([(j, shift, math.sqrt(r2)) for r2, j, shift in found[:k]])
    return rows


def as_rows(j, shifts, distances):
    return [
        [(int(a), tuple(int(v) for v in s), float(d)) for a, s, d in zip(*row)]
        for row in zip(j, shifts, distances)
    ]


@pytest.mark.parametrize("box_name", BOXES)
@pytest.mark.parametrize("pbc", [True, (True, False, True), False])
def test_nearest_atoms_of_points(box_name, pbc):
    rng = np.random.default_rng(2)
    box = mdapy.Box(BOXES[box_name], origin=[1.0, -2.0, 0.5], pbc=pbc)
    system = mdapy.System(box.to_cartesian(rng.random((30, 3))), box)
    tree = tree_of(system)
    assert len(tree) == 30
    # points inside and outside the box, one on an atom, one far away
    points = box.to_cartesian(rng.random((20, 3)) * 3 - 1)
    points[0] = system.positions[4]
    points[1] = box.to_cartesian([0.5, 0.5, 40.0])
    for k in (1, 3):
        j, shifts, distances = tree.nearest(points, k)
        assert (
            j.shape == (20, k)
            and shifts.shape == (20, k, 3)
            and distances.shape == (20, k)
        )
        assert j.dtype == mdapy.index_dtype and shifts.dtype == np.int8
        assert as_rows(j, shifts, distances) == brute_points(system, points, k)
    assert tree.nearest(points, 1)[0][0, 0] == 4  # the atom itself, at distance 0
    # the tree is searched again without being built again
    j, _, _ = tree.nearest(system.positions, 1)
    assert (j[:, 0] == np.arange(30)).all()


def test_a_tree_of_one_atom_in_a_tiny_box():
    system = mdapy.System([[0.1, 0.2, 0.3]], mdapy.Box(1.5))
    points = np.array([[0.0, 0.0, 0.0], [5.0, -3.0, 2.2]])
    rows = as_rows(*tree_of(system).nearest(points, 7))
    assert rows == brute_points(system, points, 7)


def test_tree_errors():
    empty = mdapy.System(np.zeros((0, 3)), mdapy.Box(5.0))
    with pytest.raises(ValueError, match="no atoms"):
        tree_of(empty).nearest(np.zeros((1, 3)), 1)
    system = mdapy.System([[0.0, 0, 0], [1.0, 0, 0]], mdapy.Box(5.0, pbc=False))
    tree = tree_of(system)
    with pytest.raises(ValueError, match="at least 1"):
        tree.nearest(np.zeros((1, 3)), 0)
    with pytest.raises(ValueError, match="only 2 neighbors"):
        tree.nearest(np.zeros((1, 3)), 3)
    j, shifts, distances = tree.nearest(np.zeros((0, 3)), 2)
    assert j.shape == (0, 2) and shifts.shape == (0, 2, 3) and distances.shape == (0, 2)
    with pytest.raises(ValueError, match="finite"):
        tree.nearest(np.array([[np.nan, 0.0, 0.0]]), 1)


@pytest.mark.parametrize("layout", ["uniform", "layers", "duplicates", "plane"])
def test_a_tree_of_many_atoms_matches_brute_force(layout):
    # wide enough to be built by all threads together at its top levels
    rng = np.random.default_rng(5)
    n = 40000
    x = rng.random((n, 3)) * 30.0
    if layout == "layers":  # many equal coordinates along one axis
        x[:, 2] = rng.integers(0, 5, n) * 3.0
    elif layout == "duplicates":  # equal points, tied in distance
        x = np.repeat(x[: n // 4], 4, axis=0)
    elif layout == "plane":  # all coordinates equal along one axis
        x[:, 1] = 7.0
    system = mdapy.System(x, mdapy.Box(30.0, pbc=False))
    points = np.vstack([rng.random((60, 3)) * 30.0, x[rng.integers(0, n, 20)]])
    k = 9
    tree = tree_of(system)
    result = tree.nearest(points, k)
    for q, found, d in zip(points, result[0], result[2]):
        r = x - q
        r2 = r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2]
        near = np.lexsort((np.arange(n), r2))[:k]
        assert (found == near).all()
        assert np.array_equal(d, np.sqrt(r2[near]))
    # the same tree, whatever the number of threads
    with mdapy.num_threads(1):
        again = tree_of(system).nearest(points, k)
    assert all(np.array_equal(a, b) for a, b in zip(result, again))
