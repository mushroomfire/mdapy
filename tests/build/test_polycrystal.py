# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import itertools

import numpy as np
import pytest
from mdapy._core import build as _native

import mdapy
from mdapy.build import (
    build_crystal,
    build_polycrystal,
    gradient_seeds,
    random_rotations,
    rotations_about,
    twin_lamellae,
)
from mdapy.build._polycrystal import _whole_molecules
from mdapy.core.threads import get_num_threads, set_num_threads
from mdapy.core.topology import Topology
from mdapy.errors import InvalidParameter
from mdapy.identify import CommonNeighborAnalysis
from mdapy.neighbor import NeighborList, find_overlapping
from mdapy.transform import unwrap

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
    with pytest.raises(InvalidParameter, match="layers must be a whole number"):
        twin_lamellae(cell, [3, 3])
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


def _water(a=6.35):
    """Ordered cubic ice, 8 molecules of O-H 0.96 Å with their bonds and angles: the
    oxygen on a diamond lattice, each giving hydrogens along two of its four bonds."""
    oxygen = build_crystal("diamond", a, elements="O")
    positions = oxygen["positions"]
    fractions = oxygen.box.to_fractional(positions)
    odd = np.rint(4 * fractions).astype(int)[:, 0] % 2 == 1
    bonds = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3)
    given = np.where(odd[:, None, None], -bonds[[2, 3]], bonds[[0, 1]])
    atoms = np.stack(
        [positions, positions + 0.96 * given[:, 0], positions + 0.96 * given[:, 1]],
        axis=1,
    ).reshape(-1, 3)
    first = 3 * np.arange(len(positions))[:, None]
    return (
        mdapy.System(atoms, oxygen.box, elements=["O", "H", "H"] * len(positions))
        .with_columns(
            molecule=np.repeat(np.arange(1, len(positions) + 1), 3),
            charge=[-0.8, 0.4, 0.4] * len(positions),
        )
        .with_topology(
            Topology(
                bonds=np.vstack([first + [0, 1], first + [0, 2]]),
                angles=first + [1, 0, 2],
            )
        )
    )


def _molecules(system):
    """(M, 3, 3) the positions of the whole molecules of water, oxygen first."""
    order = np.argsort(system["molecule"], kind="stable")
    return unwrap(system)["positions"][order].reshape(-1, 3, 3)


def test_molecules_stay_whole():
    ice = build_polycrystal(_water(), mdapy.Box(30.0), 3, seed=2)
    numbers = np.bincount(ice["molecule"])[1:]
    assert (numbers == 3).all() and len(numbers) * 3 == len(ice)
    elements = np.asarray(ice["element"])[np.argsort(ice["molecule"], kind="stable")]
    assert (elements.reshape(-1, 3) == ["O", "H", "H"]).all()
    # the shape of each molecule, whatever its grain and across the box
    atoms = _molecules(ice)
    one, two = atoms[:, 1] - atoms[:, 0], atoms[:, 2] - atoms[:, 0]
    np.testing.assert_allclose(np.linalg.norm(one, axis=1), 0.96, atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(two, axis=1), 0.96, atol=1e-9)
    np.testing.assert_allclose((one * two).sum(axis=1), -(0.96**2) / 3, atol=1e-9)
    # a molecule is in one grain, and in the grain of its center
    grain = np.asarray(ice["grain"])[np.argsort(ice["molecule"], kind="stable")]
    assert (grain.reshape(-1, 3) == grain.reshape(-1, 3)[:, :1]).all()
    assert set(np.unique(ice["grain"])) == {0, 1, 2}
    # other columns follow the atoms
    np.testing.assert_array_equal(
        np.asarray(ice["charge"])[np.asarray(ice["element"]) == "O"], -0.8
    )


def test_molecules_too_close_go_whole():
    unit = _water()
    # H...O at 1.79 Å in the crystal, the nearest between molecules
    between = 6.35 * np.sqrt(3) / 4 - 0.96
    for distance in (None, 1.5):
        ice = build_polycrystal(unit, mdapy.Box(30.0), 3, seed=2, min_distance=distance)
        close = NeighborList.build(ice, 0.7 * between if distance is None else distance)
        molecule = np.asarray(ice["molecule"])
        assert (molecule[close.i] == molecule[close.j]).all()
    # with no removal, more molecules, and all of them still whole
    every = build_polycrystal(unit, mdapy.Box(30.0), 3, seed=2, min_distance=0.0)
    assert every["molecule"].max() > ice["molecule"].max()
    assert (np.bincount(every["molecule"])[1:] == 3).all()


def test_the_topology_follows_the_molecules():
    ice = build_polycrystal(_water(), mdapy.Box(30.0), 3, seed=2)
    molecules = int(ice["molecule"].max())
    bonds = ice.topology.bonds["atoms"].to_numpy()
    angles = ice.topology.angles["atoms"].to_numpy()
    assert len(bonds) == 2 * molecules and len(angles) == molecules
    molecule = np.asarray(ice["molecule"])
    assert (molecule[bonds[:, 0]] == molecule[bonds[:, 1]]).all()
    assert (np.asarray(ice["element"])[bonds[:, 0]] == "O").all()
    assert (np.asarray(ice["element"])[bonds[:, 1]] == "H").all()
    assert (np.asarray(ice["element"])[angles[:, 1]] == "O").all()
    lengths = ice.box.minimum_image(
        ice["positions"][bonds[:, 1]] - ice["positions"][bonds[:, 0]]
    )
    np.testing.assert_allclose(np.linalg.norm(lengths, axis=1), 0.96, atol=1e-9)


def test_a_molecule_across_the_box_of_the_unit():
    """Moved across the faces of its box, the unit makes the same molecules."""
    unit = _water()
    shifted = unit.with_columns(positions=unit["positions"] + [0.5, 0.2, 0.1])
    wrapped = shifted.with_columns(positions=shifted.box.wrap(shifted["positions"])[0])
    assert np.any(wrapped["positions"] != shifted["positions"])
    ice = build_polycrystal(wrapped, mdapy.Box(30.0), 3, seed=2)
    atoms = _molecules(ice)
    np.testing.assert_allclose(
        np.linalg.norm(atoms[:, 1] - atoms[:, 0], axis=1), 0.96, atol=1e-9
    )
    assert len(np.unique(ice["molecule"])) * 3 == len(ice)


def test_one_grain_of_molecules_is_the_crystal():
    unit = _water()
    box = mdapy.Box(5 * 6.35)
    one = build_polycrystal(unit, box, [[0.0, 0.0, 0.0]], orientations=np.eye(3)[None])
    assert len(one) == 3 * 8 * 125


def test_columns_of_molecules():
    ice = _water()
    columns = build_polycrystal(ice, mdapy.Box([30.0, 30.0, 1.0]), 3, seed=1)
    assert columns.box.lengths[2] == pytest.approx(6.35)
    assert (np.bincount(columns["molecule"])[1:] == 3).all()


def test_atoms_keep_their_columns_without_molecules():
    crystal = unit().with_columns(
        charge=[0.5, -0.5, 0.5, -0.5], velocities=np.ones((4, 3))
    )
    grains = build_polycrystal(crystal, mdapy.Box(20.0), 2, seed=1)
    assert "charge" in grains.table.columns and "velocities" not in grains.table.columns
    assert (
        "molecule" not in grains.table.columns and "image" not in grains.table.columns
    )
    assert set(np.unique(grains["charge"])) == {-0.5, 0.5}


def test_errors_of_molecules():
    water = _water()
    with pytest.raises(InvalidParameter, match="needs the molecule column"):
        build_polycrystal(water.drop("molecule"), mdapy.Box(30.0), 2, seed=1)
    across = water.with_topology(Topology(bonds=[[0, 3]]))
    with pytest.raises(InvalidParameter, match="bonds must lie within one molecule"):
        build_polycrystal(across, mdapy.Box(30.0), 2, seed=1)


def _by_brute_force(unit, box, seeds, rotations, axis=None):
    """The atoms of the grains without the compiled code: whole blocks of copies of the
    unit, each center compared with every seed nearby, in the order the grains, the
    copies and the sites come. A unit with molecules goes by their centers."""
    cell = unit.box.matrix
    sites = unit.box.to_fractional(unit["positions"])
    if "molecule" in unit.table.columns:
        topology = unit.topology if unit.topology is not None else Topology()
        molecule, sites, centers, nearest = _whole_molecules(unit, sites, topology)
    else:
        sites = sites - np.floor(sites)
        nearest = NeighborList.build(unit, num_neighbors=1).distances.min()
        molecule, centers = np.arange(len(sites)), sites
    origin = sites[0] @ cell
    wrapped = box.wrap(seeds)[0]
    around = np.array(list(itertools.product([-1, 0, 1], repeat=3))) @ box.matrix
    others = (wrapped[None] + around[:, None]).reshape(-1, 3)
    diagonal = np.linalg.norm(np.abs(box.matrix).sum(axis=0))
    span = np.ceil(diagonal * np.linalg.norm(np.linalg.inv(cell), axis=0)).astype(int)
    if axis is not None:
        span[axis] = 0
    copies = np.stack(
        np.meshgrid(*[np.arange(-s, s + 1) for s in span], indexing="ij"), -1
    ).reshape(-1, 1, 3)
    positions, grains, kinds = [], [], []
    for g, rotation in enumerate(rotations):
        points = ((copies + sites) @ cell - origin) @ rotation.T + seeds[g]
        query = (((copies + centers) @ cell - origin) @ rotation.T + seeds[g]).reshape(
            -1, 3
        )
        if axis is not None:
            query[:, axis] = seeds[g, axis]
        own = np.linalg.norm(query - seeds[g], axis=1)
        query = box.wrap(query)[0]
        closest = np.full(len(query), np.inf)
        for start in range(0, len(others), 64):
            apart = query[:, None] - others[start : start + 64][None]
            closest = np.minimum(closest, np.linalg.norm(apart, axis=2).min(axis=1))
        inside = own <= closest + 1e-9 * nearest
        taken = inside.reshape(len(copies), -1)[:, molecule]
        positions.append(points[taken])
        grains.append(np.full(int(taken.sum()), g))
        kinds.append(np.flatnonzero(taken.ravel()) % len(sites))
    return (
        box.wrap(np.concatenate(positions))[0],
        np.concatenate(grains),
        np.concatenate(kinds),
    )


def _lattice_of_seeds():
    return np.array([[0.0, 0, 0], [5 * A, 0, 0], [0, 5 * A, 0], [5 * A, 5 * A, 0]])


@pytest.mark.parametrize(
    ("make_unit", "box", "seeds", "rotations", "axis"),
    [
        (unit, mdapy.Box(22.0), [[3.0, 4, 5], [15, 2, 9], [8, 17, 13]], None, None),
        (
            unit,
            mdapy.Box([[24, 0, 0], [7, 23, 0], [-4, 5, 25]]),
            [[3.0, 4, 5], [15, 2, 9], [8, 17, 13]],
            None,
            None,
        ),
        # atoms exactly on the faces of the cells, kept by both grains
        (unit, mdapy.Box(10 * A), _lattice_of_seeds(), np.eye(3)[None], None),
        (unit, mdapy.Box(6 * A), [[0.0, 0, 0]], np.eye(3)[None], None),
        (unit, mdapy.Box(21.0, origin=[3.0, -2, 7]), [[1.0, 2, 3]], None, None),
        # seeds outside the box
        (unit, mdapy.Box(21.0), [[-5.0, 3, 50], [60, 20, -3]], None, None),
        (
            lambda: build_crystal("hcp", 3.2, elements="Mg"),
            mdapy.Box(21.0),
            [[3.0, 4, 5], [15, 2, 9]],
            None,
            None,
        ),
        (
            lambda: mdapy.System(
                np.zeros((1, 3)),
                mdapy.Box([[0, A / 2, A / 2], [A / 2, 0, A / 2], [A / 2, A / 2, 0]]),
                elements="Cu",
            ),
            mdapy.Box(18.0),
            [[3.0, 4, 5], [15, 2, 9]],
            None,
            None,
        ),
        (unit, mdapy.Box([A, 30.0, 30.0]), [[0.0, 3, 5], [0, 20, 11]], None, 0),
        (unit, mdapy.Box([30.0, A, 30.0]), [[3.0, 0, 5], [20, 0, 11]], None, 1),
        (unit, mdapy.Box([30.0, 30.0, A]), [[3.0, 5, 0], [20, 11, 0]], None, 2),
        # seeds not in one plane: each column is a slice of a cell of its own
        (unit, mdapy.Box([30.0, 30.0, A]), [[3.0, 5, 1], [20, 11, 3]], None, 2),
        (_water, mdapy.Box(24.0), [[3.0, 4, 5], [15, 2, 9]], None, None),
        (_water, mdapy.Box([24.0, 24.0, 6.35]), [[3.0, 4, 0], [15, 12, 0]], None, 2),
    ],
)
def test_the_atoms_of_each_grain_are_those_of_a_search_by_hand(
    make_unit, box, seeds, rotations, axis
):
    seeds = np.array(seeds)
    if rotations is None:
        if axis is None:
            rotations = random_rotations(len(seeds), seed=3)
        else:
            rotations = rotations_about(axis, 20.0 + 35.0 * np.arange(len(seeds)))
    else:
        rotations = np.tile(rotations, (len(seeds), 1, 1))
    cell = make_unit()
    positions, grain, site = _by_brute_force(cell, box, seeds, rotations, axis)
    poly = build_polycrystal(cell, box, seeds, orientations=rotations, min_distance=0.0)
    np.testing.assert_array_equal(poly["grain"], grain)
    np.testing.assert_allclose(poly["positions"], positions, rtol=0, atol=1e-9)
    assert poly["element"].tolist() == cell["element"][site].tolist()


@pytest.mark.parametrize(
    ("box", "seeds", "rotations"),
    [
        (mdapy.Box(40.0), 7, None),
        (mdapy.Box([[42, 0, 0], [9, 40, 0], [-5, 7, 44]]), 7, None),
        (mdapy.Box(10 * A), _lattice_of_seeds(), np.tile(np.eye(3), (4, 1, 1))),
        (mdapy.Box(6 * A), np.zeros((1, 3)), np.eye(3)[None]),
    ],
)
@pytest.mark.parametrize("share", [0.004, 0.3, 0.7, 0.99, 1.2])
def test_the_atoms_removed_are_those_of_a_search_of_all(box, seeds, rotations, share):
    # the atoms too close are searched among those near a face below the nearest
    # distance, among all from it on
    nearest = A / np.sqrt(2)
    options = dict(orientations=rotations, seed=5)
    every = build_polycrystal(unit(), box, seeds, min_distance=0.0, **options)
    expected = every.filter(~find_overlapping(every, share * nearest))
    poly = build_polycrystal(
        unit(), box, seeds, min_distance=share * nearest, **options
    )
    np.testing.assert_array_equal(poly["positions"], expected["positions"])
    np.testing.assert_array_equal(poly["grain"], expected["grain"])


def test_a_polycrystal_is_the_same_for_any_number_of_threads():
    box = mdapy.Box(50.0)
    water_box = mdapy.Box(30.0)
    threads = get_num_threads()
    try:
        set_num_threads(1)
        one = build_polycrystal(unit(), box, 6, seed=2)
        one_water = build_polycrystal(_water(), water_box, 3, seed=2)
        set_num_threads(4)
        many = build_polycrystal(unit(), box, 6, seed=2)
        many_water = build_polycrystal(_water(), water_box, 3, seed=2)
    finally:
        set_num_threads(threads)
    assert one.table.equals(many.table)
    assert one_water.table.equals(many_water.table)
    for kind, table in one_water.topology.tables().items():
        assert table.equals(many_water.topology.tables()[kind])


def test_fill_grains_checks_what_fits():
    eye = np.eye(3)
    good = dict(
        seeds=np.zeros((1, 3)),
        rotations=eye[None],
        radii=np.ones(1),
        spans=np.ones((1, 3), dtype=np.int64),
        offsets=np.array([0, 1], dtype=np.int64),
        faces=eye[:1],
        cell=eye,
        origin=np.zeros(3),
        sites=np.zeros((1, 3)),
        centers_of=np.zeros(1, dtype=np.int64),
        centers=np.zeros((1, 3)),
        axis=-1,
        tolerance=1e-9,
        copies=False,
        margin=-1.0,
    )
    assert _native.fill_grains(**good)[3:] == (None, None)
    for wrong in (
        dict(offsets=np.array([0, 2], dtype=np.int64)),
        dict(radii=np.ones(2)),
        dict(axis=3),
        dict(centers_of=np.array([1], dtype=np.int64)),
    ):
        with pytest.raises(ValueError):
            _native.fill_grains(**{**good, **wrong})
