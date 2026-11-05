# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Atoms too close to another, and as few of them to delete as possible."""

import numpy as np
import pytest

import mdapy
from mdapy.build import build_crystal
from mdapy.errors import InvalidParameter, MissingProperty
from mdapy.neighbor import NeighborList, find_overlapping


def along_x(*x):
    return mdapy.System(np.c_[x, np.ones(len(x)), np.ones(len(x))], mdapy.Box(20.0))


def test_pairs_chains_and_groups():
    """Deleting the selection leaves no pair that close; both without keep_one."""
    np.testing.assert_array_equal(
        find_overlapping(along_x(0, 0.5, 5), 1.0), [False, True, False]
    )
    # a chain: deleting the middle atom is enough
    chain = along_x(0, 0.8, 1.6, 5)
    np.testing.assert_array_equal(
        find_overlapping(chain, 1.0), [False, True, False, False]
    )
    np.testing.assert_array_equal(
        find_overlapping(chain, 1.0, keep_one=False), [True, True, True, False]
    )
    # a longer one, every other atom
    np.testing.assert_array_equal(
        find_overlapping(along_x(0, 0.8, 1.6, 2.4, 3.2), 1.0),
        [False, True, False, True, False],
    )
    # atoms on top of each other overlap; a pair exactly the distance apart
    # does not
    np.testing.assert_array_equal(
        find_overlapping(along_x(3, 3, 3, 4), 1.0), [False, True, True, False]
    )
    # through the periodic boundary
    np.testing.assert_array_equal(
        find_overlapping(along_x(0.2, 19.9, 10), 0.5, keep_one=False),
        [True, True, False],
    )


def test_none_too_close_after_deleting():
    rng = np.random.default_rng(4)
    gas = mdapy.System(rng.uniform(0, 12, (600, 3)), mdapy.Box(12.0))
    selected = find_overlapping(gas, 1.0)
    assert 0 < selected.sum() < len(gas)
    left = gas.filter(~selected)
    assert len(NeighborList.build(left, 1.0).j) == 0
    # every atom kept or not overlaps another when it is selected
    assert not (selected & ~find_overlapping(gas, 1.0, keep_one=False)).any()


def test_atoms_considered():
    """Atoms outside the selection neither overlap nor are selected."""
    system = along_x(0, 0.5, 5, 5.5, 10, 10.5)
    np.testing.assert_array_equal(
        find_overlapping(system, 1.0, atoms=[0, 1, 2, 4]),
        [False, True, False, False, False, False],
    )
    np.testing.assert_array_equal(
        find_overlapping(
            system, 1.0, keep_one=False, atoms=[True, True, True, True, True, False]
        ),
        [True, True, True, True, False, False],
    )


@pytest.mark.parametrize(
    "distance, atoms, message",
    [(-1.0, None, "cutoff"), (1.0, [7], "mask or indices"), (1.0, [True], "mask")],
)
def test_errors(distance, atoms, message):
    with pytest.raises(InvalidParameter, match=message):
        find_overlapping(along_x(0, 0.5), distance, atoms=atoms)


@pytest.fixture(scope="module")
def with_copies():
    """Copper with copies of some atoms: 20 doubled, 10 tripled, the copies
    within 0.35 Å of each other, all in a random order; group gives the
    atom each copies, -1 for the others."""
    crystal = build_crystal("fcc", 3.615, elements="Cu", repeat=(5, 5, 5))
    rng = np.random.default_rng(8)
    originals = rng.choice(len(crystal), 30, replace=False)
    copied = np.concatenate([originals, originals[20:]])
    positions = np.concatenate(
        [
            crystal["positions"],
            crystal["positions"][copied] + rng.uniform(-0.1, 0.1, (len(copied), 3)),
        ]
    )
    group = np.full(len(positions), -1)
    group[originals] = originals
    group[len(crystal) :] = copied
    order = rng.permutation(len(positions))
    return mdapy.System(positions[order], crystal.box), group[order]


def test_copies_leave_one_of_each_group(with_copies):
    system, group = with_copies
    every = find_overlapping(system, 0.5, keep_one=False)
    np.testing.assert_array_equal(every, group >= 0)
    selected = find_overlapping(system, 0.5)
    assert selected.sum() == 20 + 2 * 10
    left = np.bincount(group[(group >= 0) & ~selected])
    assert set(left[left > 0]) == {1}


def test_the_atom_in_the_most_pairs_goes_first():
    """Whatever the order of the atoms: of a star, its center alone."""
    star = np.array([[10.0, 10, 10]]) + [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0]]
    for center in range(4):
        order = np.roll(np.arange(4), center)
        system = mdapy.System(star[order], mdapy.Box(30.0))
        expected = order == 0
        np.testing.assert_array_equal(find_overlapping(system, 1.2), expected)


def test_as_few_as_the_exact_answer_nearly():
    """Against all subsets of a few atoms: nothing close is left, nothing is deleted
    without need, and the number deleted is near the very fewest."""
    ours = fewest = 0
    for seed in range(60):
        rng = np.random.default_rng(seed)
        system = mdapy.System(rng.uniform(0, 5, (12, 3)), mdapy.Box(5.0))
        neighbors = NeighborList.build(system, 1.6)
        first = np.repeat(np.arange(12), np.diff(neighbors.offsets))
        pairs = np.c_[first, neighbors.j]
        selected = find_overlapping(system, 1.6)
        assert not (selected[pairs[:, 0]] == selected[pairs[:, 1]])[
            ~selected[pairs[:, 0]]
        ].any()
        for k in np.flatnonzero(selected):
            # deleted because of an atom kept
            assert (~selected)[pairs[pairs[:, 0] == k, 1]].any()
        ours += selected.sum()
        fewest += min(
            bin(m).count("1")
            for m in range(1 << 12)
            if all((m >> i & 1) or (m >> j & 1) for i, j in pairs)
        )
    assert fewest <= ours <= 1.1 * fewest


def test_too_close_to_its_own_image():
    # nothing else can part them
    system = mdapy.System(
        np.array([[1.0, 1, 1], [1.0, 1, 3]]), mdapy.Box([2.0, 20, 20])
    )
    np.testing.assert_array_equal(find_overlapping(system, 2.5), [True, True])
    assert len(system.filter(~find_overlapping(system, 2.5))) == 0


def test_atoms_far_apart():
    # crashed: the number of cells did not fit an integer
    positions = np.array([[0.0, 0, 0], [1e9, 1e9, 1e9], [0.5, 0, 0]])
    selected = find_overlapping(mdapy.System(positions, elements="Cu"), 1.0)
    np.testing.assert_array_equal(selected, [False, False, True])


def water(*centers):
    """Water molecules along x, with their molecule column."""
    atoms = np.vstack([[[c, 5, 5], [c + 0.6, 5, 5], [c - 0.6, 5, 5]] for c in centers])
    numbers = np.repeat(np.arange(1, len(centers) + 1), 3)
    system = mdapy.System(
        atoms, mdapy.Box(40.0), elements=["O", "H", "H"] * len(centers)
    )
    return system.with_columns(molecule=numbers)


def test_molecules_go_whole():
    # the atoms of one molecule do not overlap each other
    assert not find_overlapping(water(5), 1.0, molecules=True).any()
    assert find_overlapping(water(5), 1.0).any()
    # far apart, none; close, the later whole
    assert not find_overlapping(water(5, 20), 1.0, molecules=True).any()
    np.testing.assert_array_equal(
        find_overlapping(water(5, 6.5), 1.0, molecules=True), [False] * 3 + [True] * 3
    )
    # a chain of molecules: the middle one is enough
    chain = find_overlapping(water(5, 6.5, 8), 1.0, molecules=True)
    np.testing.assert_array_equal(chain, [False] * 3 + [True] * 3 + [False] * 3)
    left = water(5, 6.5, 8).filter(~chain)
    close = NeighborList.build(left, 1.0)
    assert (left["molecule"][close.i] == left["molecule"][close.j]).all()
    # without keep_one, all that overlap another molecule
    np.testing.assert_array_equal(
        find_overlapping(water(5, 6.5, 20), 1.0, molecules=True, keep_one=False),
        [True] * 6 + [False] * 3,
    )
    # only the atoms considered count
    considered = np.zeros(9, dtype=bool)
    considered[:3] = True
    assert not find_overlapping(
        water(5, 6.5, 8), 1.0, molecules=True, atoms=considered
    ).any()


def test_atoms_of_no_molecule_count_alone():
    system = along_x(0, 0.5, 1.0, 5).with_columns(molecule=[0, 0, 1, 1])
    # the two of molecule 0 overlap, and the second overlaps molecule 1
    selected = find_overlapping(system, 0.7, molecules=True)
    assert selected.tolist() == [False, True, False, False]


def test_molecules_need_their_column():
    with pytest.raises(MissingProperty, match="molecule column"):
        find_overlapping(along_x(0, 0.5), 1.0, molecules=True)
    empty = mdapy.System(np.zeros((0, 3)), mdapy.Box(5.0)).with_columns(molecule=[])
    assert len(find_overlapping(empty, 1.0, molecules=True)) == 0
