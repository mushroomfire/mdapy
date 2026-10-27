# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import pytest

import mdapy
from mdapy.errors import InvalidParameter
from mdapy.transform import slice, slice_mask


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


def test_miller_indices():
    """The plane hkl . f = distance in fractional coordinates f of the box,
    its points distance d_hkl from the origin of the box."""
    system = scattered()
    box = system.box
    f = (system["positions"] - box.origin) @ box.inverse
    hkl = np.array([1, -2, 1])
    d = 1 / np.linalg.norm(box.inverse @ hkl)
    side = (f @ hkl - 0.3) * d
    np.testing.assert_array_equal(
        slice(system, hkl, 0.3, miller=True)["id"], np.flatnonzero(side <= 0)
    )
    slab = slice(system, hkl, 0.3, width=2.0, miller=True)
    np.testing.assert_array_equal(slab["id"], np.flatnonzero(np.abs(side) <= 1.0))
    # (002) of the box: planes of constant c, half a period apart
    np.testing.assert_array_equal(
        slice_mask(system, (0, 0, 2), 1.0, miller=True), f[:, 2] <= 0.5
    )


def test_a_plane_through_a_point_and_some_atoms():
    system = scattered()
    center = system.box.origin + system.box.matrix.sum(axis=0) / 2
    normal = np.array([1.0, 2.0, 0.5])
    np.testing.assert_array_equal(
        slice_mask(system, normal, through=center),
        slice_mask(system, normal, center @ normal / np.linalg.norm(normal)),
    )
    np.testing.assert_array_equal(
        slice_mask(system, (1, 1, 1), through=center, miller=True),
        slice_mask(system, (1, 1, 1), 1.5, miller=True),
    )
    # the plane acts on the atoms given, the others stay
    chosen = np.arange(0, len(system), 2)
    mask = np.zeros(len(system), dtype=bool)
    mask[chosen] = True
    kept = slice_mask(system, "z", 3.0, atoms=chosen)
    np.testing.assert_array_equal(kept, slice_mask(system, "z", 3.0, atoms=mask))
    np.testing.assert_array_equal(kept, ~mask | (system["positions"][:, 2] <= 3.0))
    np.testing.assert_array_equal(
        slice(system, "z", 3.0, atoms=mask)["id"], np.flatnonzero(kept)
    )
    assert slice_mask(system, "z", 3.0, atoms=[]).all()


@pytest.mark.parametrize(
    "options, message",
    [
        ({"normal": "z", "miller": True}, "Miller"),
        ({"normal": "z", "distance": 1.0, "through": (0, 0, 0)}, "not both"),
        ({"normal": "z", "through": (0, 0)}, "point"),
        ({"normal": "z", "atoms": np.ones(3, dtype=bool)}, "entries"),
        ({"normal": "z", "atoms": [0, 1000]}, "indices"),
        ({"normal": "z", "atoms": [0.5]}, "indices"),
    ],
)
def test_more_errors(options, message):
    with pytest.raises(InvalidParameter, match=message):
        slice_mask(scattered(10), **options)


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
