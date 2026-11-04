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
    gradient_seeds,
    random_rotations,
    rotations_about,
    twin_lamellae,
)
from mdapy.errors import InvalidParameter
from mdapy.identify import CommonNeighborAnalysis
from mdapy.neighbor import NeighborList

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


def test_rotations_about_each_axis():
    quarter = rotations_about("z", [90.0, 0.0])
    assert quarter.shape == (2, 3, 3)
    np.testing.assert_allclose(quarter[0] @ [1.0, 0.0, 0.0], [0, 1, 0], atol=1e-14)
    np.testing.assert_allclose(quarter[1], np.eye(3))
    # counterclockwise seen from the end of each axis: x to y, y to z, z to x
    for axis, name, start, end in ((2, "z", 0, 1), (0, "x", 1, 2), (1, "y", 2, 0)):
        r = rotations_about(name, 90.0)
        np.testing.assert_allclose(r, rotations_about(axis, [90.0]))
        np.testing.assert_allclose(r[0] @ np.eye(3)[start], np.eye(3)[end], atol=1e-14)
    angles = np.array([12.0, 77.0, 200.0])
    for r in rotations_about("y", angles):
        np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-14)
        assert np.linalg.det(r) == pytest.approx(1.0)
    with pytest.raises(InvalidParameter, match="one number or a list"):
        rotations_about("z", np.zeros((2, 2)))
    with pytest.raises(InvalidParameter, match="axis must be"):
        rotations_about("w", 10.0)


def _expected_seeds(sizes, volume, edges, power, factor):
    """Grains of the sizes between edges of fractions: the integral of the density."""
    sizes = np.asarray(sizes, dtype=float)
    fractions = np.linspace(0.0, 1.0, 20001)
    density = factor * volume / np.interp(fractions, *sizes.T) ** power
    total = np.concatenate(
        [[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(fractions))]
    )
    return np.diff(np.interp(edges, fractions, total))


@pytest.mark.parametrize(
    "sizes",
    [
        [(0, 20), (1, 20)],
        [(0, 15), (1, 60)],
        [(0, 60), (1, 15)],
        [(0, 15), (0.5, 60), (1, 15)],
        [(0, 15), (0.25, 60), (1, 25)],
    ],
    ids=["even", "one side", "reversed", "both sides", "uneven"],
)
def test_gradient_seeds_follow_the_density_of_the_sizes(sizes):
    box = mdapy.Box([100.0, 100.0, 600.0], origin=[5.0, -3.0, 2.0])
    seeds = gradient_seeds(box, sizes, seed=1)
    fractions = box.to_fractional(seeds)
    assert (fractions >= 0).all() and (fractions <= 1).all()
    edges = np.linspace(0.0, 1.0, 7)
    expected = _expected_seeds(sizes, box.volume, edges, 3, 6.0 / np.pi)
    assert len(seeds) == round(expected.sum())
    # each seed has its own equal share of the total, so no slab is off by more than one
    found, _ = np.histogram(fractions[:, 2], bins=edges)
    assert np.abs(found - expected).max() <= 1.5


def test_gradient_seeds_along_another_axis_in_columns():
    box = mdapy.Box([400.0, 7.0, 200.0])
    sizes = [(0, 20), (1, 50)]
    seeds = gradient_seeds(box, sizes, axis="x", columns=1, seed=2)
    assert np.all(seeds[:, 1] == box.origin[1])
    edges = np.linspace(0.0, 1.0, 5)
    expected = _expected_seeds(sizes, 400.0 * 200.0, edges, 2, 4.0 / np.pi)
    assert len(seeds) == round(expected.sum())
    found, _ = np.histogram(seeds[:, 0] / 400.0, bins=edges)
    assert np.abs(found - expected).max() <= 1.5
    # thickness does not count
    thick = mdapy.Box([400.0, 70.0, 200.0])
    assert len(gradient_seeds(thick, sizes, axis=0, columns="y", seed=2)) == len(seeds)


def test_gradient_seeds_keep_apart_and_repeat():
    box = mdapy.Box([120.0, 120.0, 240.0])
    sizes = [(0, 30), (1, 30)]
    seeds = gradient_seeds(box, sizes, seed=3)
    np.testing.assert_array_equal(seeds, gradient_seeds(box.matrix, sizes, seed=3))
    generator = np.random.default_rng(3)
    np.testing.assert_array_equal(seeds, gradient_seeds(box, sizes, seed=generator))
    assert not np.array_equal(seeds, gradient_seeds(box, sizes, seed=4))
    # closer to even than seeds at random, whose nearest may be arbitrarily close
    nearest = NeighborList.build(mdapy.System(seeds, box), num_neighbors=1).distances
    assert nearest.min() > 0.4 * 30.0
    # one grain when the box holds less
    assert len(gradient_seeds(mdapy.Box(10.0), [(0, 100), (1, 100)], seed=1)) == 1


def test_a_gradient_polycrystal_has_the_grain_sizes_asked():
    a = 3.615
    unit = build_crystal("fcc", a, elements="Cu")
    box = mdapy.Box([100.0, 100.0, 400.0])
    sizes = [(0, 25), (1, 65)]
    seeds = gradient_seeds(box, sizes, seed=2)
    grains = build_polycrystal(unit, box, seeds, seed=2)
    atoms = np.bincount(grains.table["grain"].to_numpy(), minlength=len(seeds))
    diameter = (6.0 * atoms * a**3 / 4.0 / np.pi) ** (1.0 / 3.0)
    where = seeds[:, 2] / 400.0
    for low, high in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75)):
        chosen = (where >= low) & (where < high)
        target = 25.0 + 40.0 * 0.5 * (low + high)
        assert diameter[chosen].mean() == pytest.approx(target, rel=0.1)


@pytest.mark.parametrize(
    "sizes, message",
    [
        ([20.0, 30.0], "pairs of a position"),
        ([(0, 20)], "pairs of a position"),
        ([(0.1, 20), (1, 30)], "increase from 0 to 1"),
        ([(0, 20), (0.9, 30)], "increase from 0 to 1"),
        ([(0, 20), (0.5, 30), (0.5, 20), (1, 30)], "increase from 0 to 1"),
        ([(0, 20), (1, 0)], "must be positive"),
    ],
)
def test_gradient_seeds_errors(sizes, message):
    with pytest.raises(InvalidParameter, match=message):
        gradient_seeds(mdapy.Box(100.0), sizes)


def test_gradient_seeds_not_along_the_columns():
    with pytest.raises(InvalidParameter, match="along the columns"):
        gradient_seeds(mdapy.Box(100.0), [(0, 20), (1, 30)], axis="z", columns=2)
