# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The steps of the integrator in the core: the values NumPy gives, to the
last bit, on any number of threads."""

import numpy as np
import pytest

from mdapy import _core
from mdapy.core.threads import set_num_threads
from mdapy.simulation import _integrator

core = _core.core


def atoms(n=10_000, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(n, 3)), rng.normal(size=(n, 3)), rng.normal(size=(n, 3)),
            rng.uniform(1.0, 200.0, n))  # fmt: skip


@pytest.mark.parametrize("threads", [1, 4])
def test_kick_and_drift_are_those_of_numpy(threads):
    set_num_threads(threads)
    x, v, f, m = atoms()
    vk, xk = v.copy(), x.copy()
    core.kick(vk, f, m, 0.37)
    np.testing.assert_array_equal(vk, v + 0.37 * f / m[:, None])
    vd, xd = v.copy(), x.copy()
    core.kick_drift(vd, f, m, 0.37, xd, 0.002)
    expected_v = v + 0.37 * f / m[:, None]
    np.testing.assert_array_equal(vd, expected_v)
    np.testing.assert_array_equal(xd, x + 0.002 * expected_v)
    with pytest.raises(ValueError, match="differ in length"):
        core.kick(vk, f[:-1], m, 1.0)
    with pytest.raises(ValueError, match="differ in length"):
        core.kick_drift(vk, f, m, 1.0, xk[:-1], 1.0)


@pytest.mark.parametrize("threads", [1, 4])
def test_the_largest_step_is_that_of_numpy(threads):
    set_num_threads(threads)
    x, built, _, _ = atoms()
    step = x - built
    assert core.largest_step(x, built) == float((step * step).sum(axis=1).max())
    assert core.largest_step(x[:0], built[:0]) == 0.0
    x[7_777, 1] = np.nan  # a position gone wrong is found, as NumPy's max finds it
    assert np.isnan(core.largest_step(x, built))
    with pytest.raises(ValueError, match="differ in length"):
        core.largest_step(x, built[:-1])


def test_the_steps_of_arrays_out_of_order_go_through_numpy():
    _, v, f, m = atoms(100)
    assert _integrator._fused(v, f, m)
    assert not _integrator._fused(v[::2], f[::2], m[::2])
    assert not _integrator._fused(v, f.astype(np.float32), m)
    frozen = v.copy()
    frozen.flags.writeable = False
    assert not _integrator._fused(frozen, f, m)
