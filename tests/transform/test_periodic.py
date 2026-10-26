# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import pytest

import mdapy
from mdapy.errors import InvalidParameter, MissingProperty
from mdapy.transform import replicate, unwrap, wrap

MATRIX = np.array([[6.0, 0, 0], [1.5, 5.5, 0], [-0.8, 1.1, 6.2]])


def molecules(seed=0, pbc=True):
    """Six three-atom molecules, some across the boundary of the box."""
    rng = np.random.default_rng(seed)
    box = mdapy.Box(MATRIX, origin=[1.0, -2.0, 0.5], pbc=pbc)
    centers = rng.random((6, 3))
    bond = np.array([[0, 0, 0], [0.9, 0.3, 0], [-0.4, 0.8, 0.2]])
    unwrapped = box.to_cartesian(centers)[:, None, :] + bond
    positions, image = box.wrap(unwrapped.reshape(-1, 3))
    system = mdapy.System(
        positions,
        box,
        elements=["O", "H", "H"] * 6,
        info={"energy": -3.0, "virial": np.eye(3), "timestep": 100},
        topology=mdapy.Topology(
            bonds=[[3 * m, 3 * m + k] for m in range(6) for k in (1, 2)],
            angles=[[3 * m + 1, 3 * m, 3 * m + 2] for m in range(6)],
        ),
    )
    return system.with_columns(
        id=rng.permutation(18) + 3,
        molecule=np.repeat([1, 4, 0, 2, 6, 3], 3),
        image=image,
        velocities=rng.standard_normal((18, 3)),
    )


def unwrapped(system):
    return system["positions"] + system.table["image"].to_numpy() @ system.box.matrix


def test_copies_follow_the_unwrapped_molecules():
    system = molecules()
    copies = replicate(system, (2, 3, 1))
    n = len(system)
    assert len(copies) == 6 * n
    np.testing.assert_allclose(copies.box.matrix, MATRIX * [[2], [3], [1]])
    np.testing.assert_array_equal(copies.box.origin, system.box.origin)
    # copy k is (i, j, 0) with k = i + 2 j
    cells = np.array([[i, j, 0] for j in range(3) for i in range(2)])
    shift = np.repeat(cells @ MATRIX, n, axis=0)
    np.testing.assert_allclose(
        unwrapped(copies), np.tile(unwrapped(system), (6, 1)) + shift, atol=1e-12
    )
    # inside the new box, and the image flags are those of the new box
    fractions = copies.box.to_fractional(copies["positions"])
    assert fractions.min() >= 0 and fractions.max() < 1
    # a bond stays between the atoms of one copy, as long as before
    bonds = copies.topology.bonds["atoms"].to_numpy()
    old = system.topology.bonds["atoms"].to_numpy()
    np.testing.assert_array_equal(
        bonds, np.concatenate([old + k * n for k in range(6)])
    )
    lengths = np.linalg.norm(np.diff(unwrapped(copies)[bonds], axis=1), axis=2)
    np.testing.assert_allclose(
        lengths,
        np.tile(
            np.linalg.norm(np.diff(unwrapped(system)[old], axis=1), axis=2), (6, 1)
        ),
    )
    assert len(copies.topology.angles) == 6 * len(system.topology.angles)


def test_ids_molecules_columns_and_info():
    system = molecules()
    copies = replicate(system, 2)
    n = len(system)
    k = np.repeat(np.arange(8), n)
    ids = system.table["id"].to_numpy()
    np.testing.assert_array_equal(
        copies.table["id"].to_numpy(), np.tile(ids, 8) + k * ids.max()
    )
    assert len(np.unique(copies.table["id"].to_numpy())) == 8 * n
    molecule = np.tile(system.table["molecule"].to_numpy(), 8)
    expected = np.where(molecule > 0, molecule + 6 * k, 0)
    np.testing.assert_array_equal(copies.table["molecule"].to_numpy(), expected)
    assert copies.table["element"].to_list() == system.table["element"].to_list() * 8
    np.testing.assert_array_equal(
        copies.table["velocities"].to_numpy(),
        np.tile(system.table["velocities"].to_numpy(), (8, 1)),
    )
    # what describes the whole of the original system is left out
    assert dict(copies.info) == {"timestep": 100}


def test_without_image_flags_the_copies_are_translated():
    system = mdapy.System(
        [[0.5, 0.5, 0.5], [6.5, 1.0, 2.0]], mdapy.Box(MATRIX), elements="Cu"
    )
    copies = replicate(system, (1, 2, 2))
    cells = np.array([[0, j, k] for k in range(2) for j in range(2)])
    np.testing.assert_allclose(
        copies["positions"],
        np.tile(system["positions"], (4, 1)) + np.repeat(cells @ MATRIX, 2, axis=0),
    )
    assert "image" not in copies.table.columns and copies.topology is None


def test_neighbors_of_a_crystal_do_not_change():
    """Every copy of an atom sees what the atom saw."""
    rng = np.random.default_rng(3)
    box = mdapy.Box(MATRIX)
    system = mdapy.System(box.to_cartesian(rng.random((30, 3))), box, elements="Ar")
    copies = replicate(system, (3, 1, 2))

    def shells(s):
        r = s["positions"][:, None, :] - s["positions"][None, :, :]
        f = r @ s.box.inverse
        f -= np.round(f)
        d = np.linalg.norm(f @ s.box.matrix, axis=2)
        return np.sort(d, axis=1)[:, 1:6]

    np.testing.assert_allclose(
        shells(copies), np.tile(shells(system), (6, 1)), atol=1e-12
    )


def test_single_copies_empty_systems_and_errors():
    system = molecules()
    same = replicate(system, 1)
    np.testing.assert_allclose(same["positions"], system["positions"], atol=1e-12)
    np.testing.assert_array_equal(
        same.table["image"].to_numpy(), system.table["image"].to_numpy()
    )
    assert same.topology == system.topology
    empty = mdapy.System(np.zeros((0, 3)), mdapy.Box(5.0), elements=[]).with_columns(
        id=np.zeros(0, np.int64)
    )
    assert len(replicate(empty, 3)) == 0
    slab = molecules(pbc=(True, True, False))
    assert len(replicate(slab, (2, 2, 1))) == 4 * len(slab)
    with pytest.raises(InvalidParameter, match="only along periodic directions"):
        replicate(slab, (1, 1, 2))
    for bad in (0, (1, 2), (1.5, 1, 1), (1, -1, 1)):
        with pytest.raises(InvalidParameter, match="repeat"):
            replicate(system, bad)


def test_wrap_and_unwrap_keep_the_unwrapped_positions():
    system = unwrap(molecules())
    np.testing.assert_array_equal(system.table["image"].to_numpy(), 0)
    outside = system.box.to_fractional(system["positions"])
    assert outside.min() < 0 or outside.max() >= 1
    wrapped = wrap(system)
    fractions = wrapped.box.to_fractional(wrapped["positions"])
    assert fractions.min() >= 0 and fractions.max() < 1
    np.testing.assert_allclose(unwrapped(wrapped), system["positions"], atol=1e-12)
    np.testing.assert_allclose(
        unwrap(wrapped)["positions"], system["positions"], atol=1e-12
    )
    assert wrapped.topology == system.topology
    # without image flags, and along non-periodic directions
    bare = mdapy.System(system["positions"], mdapy.Box(MATRIX, pbc=(True, False, True)))
    moved = wrap(bare)
    assert "image" not in moved.table.columns
    fractions = moved.box.to_fractional(moved["positions"])
    np.testing.assert_allclose(
        fractions[:, 1], bare.box.to_fractional(bare["positions"])[:, 1]
    )
    assert fractions[:, [0, 2]].min() >= 0 and fractions[:, [0, 2]].max() < 1
    with pytest.raises(MissingProperty, match="image"):
        unwrap(bare)
