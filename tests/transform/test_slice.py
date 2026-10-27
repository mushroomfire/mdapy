# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import pytest

import mdapy
from mdapy.errors import InvalidParameter
from mdapy.transform import slice


def scattered(n=300, seed=0):
    rng = np.random.default_rng(seed)
    box = mdapy.Box([[10.0, 0, 0], [2.0, 9.0, 0], [1.0, 1.0, 8.0]])
    system = mdapy.System(box.to_cartesian(rng.random((n, 3))), box, elements="Cu")
    return system.with_columns(id=np.arange(n))


def test_one_side_and_the_other():
    system = scattered()
    normal = np.array([1.0, 2.0, 0.5])
    side = system["positions"] @ normal / np.linalg.norm(normal) - 4.0
    below = slice(system, normal, 4.0)
    above = slice(system, 3 * normal, 4.0, invert=True)  # any length
    np.testing.assert_array_equal(below["id"], np.flatnonzero(side <= 0))
    np.testing.assert_array_equal(above["id"], np.flatnonzero(side > 0))
    assert len(below) + len(above) == len(system)
    assert below.box == system.box
    assert set(below.table.columns) == set(system.table.columns)


def test_slab():
    system = scattered()
    z = system["positions"][:, 2]
    slab = slice(system, "z", 3.0, width=2.0)
    np.testing.assert_array_equal(slab["id"], np.flatnonzero(np.abs(z - 3.0) <= 1.0))
    rest = slice(system, (0, 0, 1), 3.0, width=2.0, invert=True)
    np.testing.assert_array_equal(rest["id"], np.flatnonzero(np.abs(z - 3.0) > 1.0))
    np.testing.assert_array_equal(
        slice(system, "x")["id"], np.flatnonzero(system["positions"][:, 0] <= 0)
    )


@pytest.mark.parametrize(
    "normal, width, message",
    [
        ((0, 0, 0), 0.0, "not all zero"),
        ((1, 0), 0.0, "three numbers"),
        ("w", 0.0, "x, y, z"),
        ((0, 0, 1), -1.0, "width"),
    ],
)
def test_errors(normal, width, message):
    with pytest.raises(InvalidParameter, match=message):
        slice(scattered(10), normal, width=width)
