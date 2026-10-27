# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import pytest

import mdapy
from mdapy.analysis import AtomicDisplacement
from mdapy.errors import InvalidParameter


def pair(seed=0, n=50):
    rng = np.random.default_rng(seed)
    box = mdapy.Box([[10.0, 0, 0], [2.0, 9.0, 0], [1.0, 1.0, 8.0]], origin=[1, 2, 3])
    reference = mdapy.System(box.to_cartesian(rng.random((n, 3))), box, elements="Cu")
    reference = reference.with_columns(id=np.arange(1, n + 1))
    moves = rng.normal(scale=0.3, size=(n, 3))
    current = reference.with_columns(positions=reference["positions"] + moves)
    return reference, current, moves


def test_moves_and_order():
    reference, current, moves = pair()
    result = AtomicDisplacement(reference).compute(current)
    np.testing.assert_allclose(result.displacement, moves, atol=1e-12)
    np.testing.assert_allclose(result.magnitude, np.linalg.norm(moves, axis=1))
    # matched by id, in the order of the system
    order = np.random.default_rng(1).permutation(len(current))
    shuffled = AtomicDisplacement(reference).compute(current.gather(order))
    np.testing.assert_allclose(shuffled.displacement, moves[order], atol=1e-12)
    # without ids, by order
    plain = AtomicDisplacement(reference.drop("id")).compute(current.drop("id"))
    np.testing.assert_allclose(plain.displacement, moves, atol=1e-12)
    table = current.with_columns(result).table
    assert {"displacement", "displacement_magnitude"} <= set(table.columns)


def test_periodic_images():
    reference, current, moves = pair()
    jumped = current.with_columns(
        positions=current["positions"] + reference.box.matrix[0] * 2
    )
    np.testing.assert_allclose(
        AtomicDisplacement(reference).compute(jumped).displacement, moves, atol=1e-12
    )
    whole = AtomicDisplacement(reference, minimum_image=False).compute(jumped)
    np.testing.assert_allclose(
        whole.displacement, moves + 2 * reference.box.matrix[0], atol=1e-12
    )


def test_affine_mapping():
    """In a box stretched with the atoms, no displacement in the reference
    box."""
    reference, _, _ = pair()
    strain = np.eye(3) + [[0.05, 0.02, 0], [0, -0.03, 0.01], [0, 0, 0.04]]
    box = mdapy.Box(reference.box.matrix @ strain, origin=reference.box.origin)
    stretched = mdapy.System(
        box.to_cartesian(reference.box.to_fractional(reference["positions"])),
        box,
        elements="Cu",
    ).with_columns(id=reference["id"])
    np.testing.assert_allclose(
        AtomicDisplacement(reference, affine=True).compute(stretched).displacement,
        0,
        atol=1e-12,
    )
    assert (
        np.abs(AtomicDisplacement(reference).compute(stretched).displacement).max()
        > 0.1
    )
    assert (
        repr(AtomicDisplacement(reference))
        == "AtomicDisplacement(minimum_image=True, affine=False)"
    )


def test_errors():
    reference, current, _ = pair()
    stranger = current.with_columns(id=current["id"] + 1000)
    with pytest.raises(InvalidParameter, match="not in the reference"):
        AtomicDisplacement(reference).compute(stranger)
    with pytest.raises(InvalidParameter, match="no ids"):
        AtomicDisplacement(reference.drop("id")).compute(
            current.drop("id").gather([0, 1])
        )
