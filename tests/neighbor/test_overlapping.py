# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Atoms too close to another: as OVITO's find overlapping particles."""

import numpy as np
import pytest

import mdapy
from mdapy.build import build_crystal
from mdapy.errors import InvalidParameter
from mdapy.neighbor import NeighborList, find_overlapping


def along_x(*x):
    return mdapy.System(np.c_[x, np.ones(len(x)), np.ones(len(x))], mdapy.Box(20.0))


def test_pairs_chains_and_groups():
    """The atom of the higher index of each pair; both without keep_one."""
    np.testing.assert_array_equal(
        find_overlapping(along_x(0, 0.5, 5), 1.0), [False, True, False]
    )
    # a chain: deleting the two later atoms leaves no pair closer than 1
    chain = along_x(0, 0.8, 1.6, 5)
    np.testing.assert_array_equal(
        find_overlapping(chain, 1.0), [False, True, True, False]
    )
    np.testing.assert_array_equal(
        find_overlapping(chain, 1.0, keep_one=False), [True, True, True, False]
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


def test_as_ovito(with_copies):
    """Without keep_one the same atoms; with it, all but one of each group
    of copies, OVITO picking the one it keeps at random."""
    pytest.importorskip("ovito")
    from ovito.modifiers import (
        ExpressionSelectionModifier,
        FindOverlappingParticlesModifier,
    )

    from ovito_reference import ovito_compute

    system, group = with_copies
    data = ovito_compute(
        system, FindOverlappingParticlesModifier(overlap_distance=0.5, keep_one=False)
    )
    every = find_overlapping(system, 0.5, keep_one=False)
    np.testing.assert_array_equal(every, np.asarray(data.particles.selection) != 0)
    np.testing.assert_array_equal(every, group >= 0)
    data = ovito_compute(system, FindOverlappingParticlesModifier(overlap_distance=0.5))
    theirs = np.asarray(data.particles.selection) != 0
    ours = find_overlapping(system, 0.5)
    np.testing.assert_array_equal(
        np.bincount(group[ours], minlength=len(group)),
        np.bincount(group[theirs], minlength=len(group)),
    )
    assert ours.sum() == 20 + 2 * 10
    data = ovito_compute(
        system,
        ExpressionSelectionModifier(expression="Position.Z < 9"),
        FindOverlappingParticlesModifier(
            overlap_distance=0.5, keep_one=False, only_selected=True
        ),
    )
    lower = find_overlapping(
        system, 0.5, keep_one=False, atoms=system.where("positions.z < 9")
    )
    np.testing.assert_array_equal(lower, np.asarray(data.particles.selection) != 0)
    assert 0 < lower.sum() < every.sum()
