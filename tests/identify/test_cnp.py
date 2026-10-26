# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import pytest

import mdapy
from mdapy.errors import InvalidParameter
from mdapy.identify import CommonNeighborParameter

from lattices import crystal


def reference(system, cutoff):
    """The parameter by its definition, common neighbors found by atom and
    periodic image in the neighbor lists of both atoms."""
    nl = mdapy.NeighborList.build(system, cutoff)
    x, h = system["positions"], system.box.matrix
    pairs = [
        {(int(j), tuple(s)) for j, s in zip(nl.j[a:b], nl.shifts[a:b])}
        for a, b in zip(nl.offsets[:-1], nl.offsets[1:])
    ]
    values = np.full(len(system), np.nan)
    for i in range(len(system)):
        if not pairs[i]:
            continue
        total = 0.0
        for j, sj in pairs[i]:
            r = np.zeros(3)
            for k, sk in pairs[j]:
                image = tuple(np.add(sj, sk))  # k as seen from i
                if (k, image) in pairs[i]:
                    xk = x[k] + np.array(image) @ h
                    r += (x[i] - xk) + (x[j] + np.array(sj) @ h - xk)
            total += r @ r
        values[i] = total / len(pairs[i])
    return values


@pytest.mark.parametrize(
    "kind, a, cutoff", [("fcc", 3.6, 3.0), ("bcc", 2.87, 3.3), ("diamond", 5.43, 2.6)]
)
def test_zero_in_symmetric_crystals(kind, a, cutoff):
    cnp = CommonNeighborParameter(cutoff).compute(crystal(kind, a)).cnp
    np.testing.assert_allclose(cnp, 0.0, atol=1e-20)


def test_ideal_hcp():
    # each atom has the same value, two thirds of a squared
    cnp = CommonNeighborParameter(2.8).compute(crystal("hcp", 2.5)).cnp
    np.testing.assert_allclose(cnp, 2.0 / 3.0 * 2.5**2, rtol=1e-13)


def test_matches_the_definition():
    system = crystal("fcc", 3.6, repeat=(3, 3, 3), jitter=0.2, seed=3)
    result = CommonNeighborParameter(3.1).compute(system)
    np.testing.assert_allclose(result.cnp, reference(system, 3.1), rtol=1e-12)
    assert list(result.columns()) == ["cnp"]
    with mdapy.num_threads(3):
        again = CommonNeighborParameter(3.1).compute(system).cnp
    np.testing.assert_array_equal(again, result.cnp)
    # a longer neighbor list gives the same, summed in another order than
    # the grid search
    nl = mdapy.NeighborList.build(system, 4.0)
    np.testing.assert_allclose(
        CommonNeighborParameter(3.1).compute(system, neighbors=nl).cnp,
        result.cnp,
        rtol=1e-13,
    )


def test_small_box_with_images():
    # a cutoff longer than half the box: neighbors are several images
    system = crystal("bcc", 2.87, repeat=(1, 1, 2), jitter=0.05, seed=1)
    np.testing.assert_allclose(
        CommonNeighborParameter(3.0).compute(system).cnp,
        reference(system, 3.0),
        rtol=1e-12,
    )


def test_rotation_and_translation_change_nothing():
    system = crystal("fcc", 3.6, jitter=0.15, seed=5)
    angle = 0.4
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    moved = mdapy.System(
        system["positions"] @ rotation.T + 1.7,
        mdapy.Box(system.box.matrix @ rotation.T, origin=system.box.origin + 1.7),
        elements="Cu",
    )
    np.testing.assert_allclose(
        CommonNeighborParameter(3.1).compute(moved).cnp,
        CommonNeighborParameter(3.1).compute(system).cnp,
        rtol=1e-10,
        atol=1e-12,
    )


def test_isolated_atom_and_errors():
    pair = mdapy.System(
        [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [7.0, 7.0, 7.0]],
        mdapy.Box(10.0),
        elements="Ar",
    )
    cnp = CommonNeighborParameter(1.5).compute(pair).cnp
    # a lone pair has no common neighbors; the third atom has no neighbors
    np.testing.assert_array_equal(cnp[:2], [0.0, 0.0])
    assert np.isnan(cnp[2])
    assert repr(CommonNeighborParameter(3.0)) == "CommonNeighborParameter(cutoff=3.0)"
    with pytest.raises(InvalidParameter, match="cutoff must be positive"):
        CommonNeighborParameter(0.0)
