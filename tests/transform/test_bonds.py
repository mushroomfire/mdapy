# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import pytest

import mdapy
from mdapy import transform
from mdapy.errors import InvalidParameter


def test_bonds_by_hand():
    """Every pair closer than the cutoff, across the boundaries, once."""
    rng = np.random.default_rng(8)
    system = mdapy.System(rng.random((60, 3)) * 9, mdapy.Box(9.0), elements="Ar")
    bonded = transform.create_bonds(system, 2.5)
    pairs = np.asarray(bonded.topology.bonds["atoms"].to_list())
    assert (pairs[:, 0] < pairs[:, 1]).all()
    step = system["positions"][:, None] - system["positions"][None]
    step -= 9.0 * np.round(step / 9.0)
    near = np.linalg.norm(step, axis=2) < 2.5
    expected = {(i, j) for i, j in zip(*np.nonzero(near)) if i < j}
    assert set(map(tuple, pairs)) == expected


def test_bonds_replace_bonds_and_keep_angles():
    system = mdapy.System(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0]], mdapy.Box(20.0, pbc=False), elements="C"
    ).with_topology(mdapy.Topology(bonds=[[0, 2]], angles=[[1, 0, 2]]))
    bonded = transform.create_bonds(system, 1.2)
    assert sorted(map(tuple, bonded.topology.bonds["atoms"].to_list())) == [
        (0, 1),
        (0, 2),
    ]
    assert bonded.topology.angles["atoms"].to_list() == [[1, 0, 2]]
    alone = transform.create_bonds(system.with_topology(None), 1.2)
    assert len(alone.topology.angles) == 0


def test_covalent_bonds_of_water():
    water = mdapy.System(
        [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0], [3, 3, 3]],
        mdapy.Box(20.0, pbc=False),
        elements=["O", "H", "H", "O"],
    )
    bonded = transform.create_bonds(water, "covalent")
    assert sorted(map(tuple, bonded.topology.bonds["atoms"].to_list())) == [
        (0, 1),
        (0, 2),
    ]


def test_bond_errors():
    system = mdapy.System(np.zeros((1, 3)), mdapy.Box(5.0), elements="Cu")
    with pytest.raises(InvalidParameter, match="covalent"):
        transform.create_bonds(system, "vdw")
    with pytest.raises(InvalidParameter, match="positive"):
        transform.create_bonds(system, 0.0)
