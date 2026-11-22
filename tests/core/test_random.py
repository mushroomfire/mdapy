# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The counter-based random numbers of the noise of molecular dynamics,
Philox4x64-10 (core/philox.h): the known answer of Random123, numpy's
Philox, and Gaussian numbers that depend only on their counter."""

import numpy as np
import pytest

from mdapy import _core

native = _core.core


def test_the_known_answer_of_random123():
    # kat_vectors of Random123: philox4x64-10 of a zero counter and key
    assert native.philox(0, 0, 0, 0, 0, 0) == (
        0x16554D9ECA36314C, 0xDB20FE9D672D0FDC, 0xD7E772CEE186176B, 0x7E68B68AEC7BA23B,
    )  # fmt: skip


@pytest.mark.parametrize(
    "counter, key",
    [
        ([1, 0, 0, 0], [0, 0]),
        ([6, 3, 0, 0], [7, 9]),
        ([2**64 - 1, 5, 2**63, 1], [2**64 - 1, 3]),
        ([123456789, 987654321, 42, 2**40], [2**62 + 11, 2**33]),
    ],
)
def test_the_blocks_of_numpy(counter, key):
    """numpy's Philox adds one to its counter before a block."""
    before = [counter[0] - 1, *counter[1:]]
    theirs = np.random.Philox(
        counter=np.array(before, dtype=np.uint64), key=np.array(key, dtype=np.uint64)
    ).random_raw(4)
    assert native.philox(*counter, *key) == tuple(int(v) for v in theirs)


def test_gaussian_numbers_of_each_atom_and_draw():
    n = 200_000
    g = native.normals(np.arange(n, dtype=np.int64), 12345, 0, 7)
    assert g.shape == (n, 3) and g.dtype == np.float64
    assert abs(g.mean()) < 0.01 and g.var() == pytest.approx(1.0, abs=0.01)
    assert np.mean(g**4) == pytest.approx(3.0, abs=0.05)  # kurtosis of a Gaussian
    assert np.abs(np.corrcoef(g.T) - np.eye(3)).max() < 0.01
    # the numbers of an atom depend on its index alone, not on the others
    some = np.array([7, 199_999, 3, 7], dtype=np.int64)
    np.testing.assert_array_equal(native.normals(some, 12345, 0, 7), g[some])
    # other draws, streams and seeds give other numbers
    for other in ((12345, 0, 8), (12345, 1, 7), (12346, 0, 7)):
        h = native.normals(np.arange(n, dtype=np.int64), *other)
        assert abs(np.corrcoef(g.ravel(), h.ravel())[0, 1]) < 0.01
    assert native.normals(np.zeros(0, dtype=np.int64), 1, 0, 0).shape == (0, 3)
