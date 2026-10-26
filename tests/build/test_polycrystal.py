# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import pytest

import mdapy
from mdapy.build import (
    build_crystal,
    build_polycrystal,
    random_rotations,
    twin_lamellae,
)
from mdapy.errors import InvalidParameter
from mdapy.identify import CommonNeighborAnalysis

A = 3.615
FCC111 = [(1, 1, -2), (-1, 1, 0), (1, 1, 1)]


def unit():
    return build_crystal("fcc", A, elements="Cu")


def test_one_grain_is_the_crystal():
    box = mdapy.Box(6 * A)
    one = build_polycrystal(unit(), box, [[0.0, 0, 0]], orientations=np.eye(3)[None])
    assert len(one) == 4 * 6**3
    assert CommonNeighborAnalysis().compute(one).counts()["fcc"] == len(one)
    assert (one.table["grain"].to_numpy() == 0).all()


def test_grains():
    box = mdapy.Box(60.0)
    poly = build_polycrystal(unit(), box, 6, seed=1)
    seeds_again = build_polycrystal(unit(), box, 6, seed=1)
    np.testing.assert_array_equal(poly["positions"], seeds_again["positions"])
    grain = poly.table["grain"].to_numpy()
    assert set(grain.tolist()) == set(range(6))
    # no two atoms closer than the minimum distance
    assert len(mdapy.NeighborList.build(poly, 0.7 * A / np.sqrt(2) - 1e-9)) == 0
    # the inside of every grain is fcc
    structures = CommonNeighborAnalysis().compute(poly).structure
    nl = mdapy.NeighborList.build(poly, 6.0)
    mixed = np.zeros(len(poly), bool)
    np.logical_or.at(mixed, nl.i, grain[nl.i] != grain[nl.j])
    assert (structures[~mixed] == 1).mean() > 0.99
    assert np.isclose(len(poly) / (box.volume / A**3 * 4), 1.0, atol=0.08)


def test_atoms_in_the_cells_of_their_seeds():
    box = mdapy.Box(50.0)
    seeds = np.random.default_rng(2).random((5, 3)) * 50
    poly = build_polycrystal(unit(), box, seeds, seed=2, min_distance=0.0)
    grain = poly.table["grain"].to_numpy()
    images = (
        np.array(np.meshgrid(*[[-1, 0, 1]] * 3, indexing="ij")).reshape(3, -1).T * 50.0
    )
    d = np.linalg.norm(
        poly["positions"][:, None, None] - seeds[None, :, None] - images[None, None],
        axis=3,
    ).min(axis=2)
    np.testing.assert_allclose(d[np.arange(len(poly)), grain], d.min(axis=1), atol=1e-6)


def test_given_orientations():
    rotations = random_rotations(3, seed=4)
    np.testing.assert_allclose(
        np.einsum("gij,gkj->gik", rotations, rotations),
        np.broadcast_to(np.eye(3), (3, 3, 3)),
        atol=1e-12,
    )
    assert (np.linalg.det(rotations) > 0).all()
    seeds = np.array([[10.0, 10, 10], [30, 30, 30], [10, 30, 20]])
    poly = build_polycrystal(unit(), mdapy.Box(40.0), seeds, orientations=rotations)
    nl = mdapy.NeighborList.build(poly, num_neighbors=12)
    grain = poly.table["grain"].to_numpy()
    for g in range(3):
        # the atom at the seed has its twelve neighbors along the rotated <110>
        i = np.argmin(np.linalg.norm(poly["positions"] - seeds[g], axis=1))
        assert grain[i] == g
        bonds = (
            nl.vectors[nl.offsets[i] : nl.offsets[i + 1]] @ rotations[g]
        )  # back to the cube axes
        expected = (
            A
            / 2
            * np.array(
                [
                    v
                    for v in np.array(np.meshgrid(*[[-1, 0, 1]] * 3)).reshape(3, -1).T
                    if (v != 0).sum() == 2
                ]
            )
        )
        distance = np.linalg.norm(bonds[:, None] - expected[None], axis=2).min(axis=1)
        assert distance.max() < 1e-9


def test_columnar_polycrystal():
    cell = build_crystal(
        "fcc", A, elements="Cu", orientation=[(1, -1, 0), (1, 1, -2), (1, 1, 1)]
    )
    a = 40.0
    seeds = [[0, 0, 0], [0.5 * a * np.sqrt(3), 0.5 * a, 0]]
    box = mdapy.Box([a * np.sqrt(3), a, 1.0])
    poly = build_polycrystal(cell, box, seeds, seed=5)
    assert poly.box.matrix[2, 2] == pytest.approx(cell.box.matrix[2, 2])
    structures = CommonNeighborAnalysis().compute(poly).structure
    assert (structures == 1).mean() > 0.7
    rotations = np.stack([np.eye(3), np.eye(3)])
    rotations[1, :2, :2] = [[0, -1], [1, 0]]
    turned = build_polycrystal(cell, box, seeds, orientations=rotations)
    assert set(turned.table["grain"].to_list()) == {0, 1}
    tilted = random_rotations(2, seed=6)
    with pytest.raises(InvalidParameter, match="turn about its axis"):
        build_polycrystal(cell, box, seeds, orientations=tilted)
    with pytest.raises(InvalidParameter, match="along two axes"):
        build_polycrystal(cell, mdapy.Box([a, 1.0, 1.0]), 2)
    with pytest.raises(InvalidParameter, match="orthogonal boxes"):
        build_polycrystal(cell, mdapy.Box([[a, 0, 0], [5.0, a, 0], [0, 0, 1.0]]), 2)


def test_twin_lamellae():
    cell = build_crystal("fcc", A, elements="Cu", orientation=FCC111)
    twins = twin_lamellae(cell, 6)
    spacing = A / np.sqrt(3)
    assert twins.box.matrix[2, 2] == pytest.approx(12 * spacing)
    assert len(twins) == 12 * len(cell) // 3
    counts = CommonNeighborAnalysis().compute(twins).counts()
    assert (
        counts["hcp"] == 2 * len(cell) // 3
        and counts["fcc"] == len(twins) - counts["hcp"]
    )
    # a polycrystal of twinned grains
    poly = build_polycrystal(twin_lamellae(cell, 8), mdapy.Box(60.0), 4, seed=7)
    assert CommonNeighborAnalysis().compute(poly).counts()["hcp"] > 0.03 * len(poly)
    for bad, message in [
        (
            build_crystal(
                "fcc", A, elements="Cu", orientation=[(1, 1, 0), (-1, 1, 0), (0, 0, 1)]
            ),
            None,
        ),
        (
            cell.with_box(mdapy.Box(cell.box.matrix, pbc=(True, True, False))),
            "cell must be periodic",
        ),
    ]:
        if message is None:
            assert len(twin_lamellae(bad, 2)) > 0  # (001) planes are equally spaced too
            continue
        with pytest.raises(InvalidParameter, match=message):
            twin_lamellae(bad, 4)
    with pytest.raises(InvalidParameter, match="layers must be at least 2"):
        twin_lamellae(cell, 1)
    hcp = build_crystal("hcp", 3.2, elements="Mg")  # a tilted box
    with pytest.raises(InvalidParameter, match="xy plane|third vector"):
        twin_lamellae(
            hcp.with_box(mdapy.Box([[3.2, 0, 0], [0, 5.5, 0.5], [0, 0, 5.2]])), 2
        )
    uneven = mdapy.System([[0.0, 0, 0], [0, 0, 1.0]], mdapy.Box(4.0), elements="Cu")
    with pytest.raises(InvalidParameter, match="equally spaced planes"):
        twin_lamellae(uneven, 2)


def test_errors():
    box = mdapy.Box(30.0)
    for arguments, message in [
        ({"grains": 0}, "grains must be at least 1"),
        ({"grains": [[0.0, 0]]}, "grains must be a number"),
        (
            {"grains": 2, "orientations": np.eye(3)[None]},
            "orientations must have shape",
        ),
        ({"grains": 1, "orientations": 2 * np.eye(3)[None]}, "rotation matrices"),
        ({"grains": 1, "min_distance": -1.0}, "must not be negative"),
    ]:
        with pytest.raises(InvalidParameter, match=message):
            build_polycrystal(unit(), box, **arguments)
    with pytest.raises(InvalidParameter, match="periodic in every direction"):
        build_polycrystal(unit(), mdapy.Box(30.0, pbc=(True, True, False)), 2)
    with pytest.raises(InvalidParameter, match="unit must be a periodic box"):
        build_polycrystal(unit().with_box(mdapy.Box(A, pbc=False)), box, 2)
