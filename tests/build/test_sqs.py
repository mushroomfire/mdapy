# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

import numpy as np
import pytest
from mdapy._core import build as _native

import mdapy
from mdapy.build import build_crystal, random_alloy, sqs_alloy
from mdapy.build._sqs import _shell_index, _shells
from mdapy.errors import Cancelled, InvalidParameter


def warren_cowley(system, shells):
    """alpha_ab of each shell from the pairs, by brute force: 1 - P(b
    around a) / x_b."""
    nl = _shells(system, shells)
    shell = _shell_index(nl, 1e-3 * nl.distances.min())
    kinds = np.array(system.table["element"].to_list())
    names = sorted(set(kinds))
    x = np.array([(kinds == e).mean() for e in names])
    alpha = np.zeros((shells, len(names), len(names)))
    for s in range(shells):
        pairs = shell == s
        i, j = nl.i[pairs], nl.j[pairs]
        for a, ea in enumerate(names):
            around = kinds[j[kinds[i] == ea]]
            for b, eb in enumerate(names):
                alpha[s, a, b] = 1 - (around == eb).mean() / x[b]
    return alpha


def test_shells_of_fcc():
    sites = build_crystal("fcc", 3.6, repeat=(4, 4, 4))
    nl = _shells(sites, 3)
    shell = _shell_index(nl, 1e-3)
    np.testing.assert_array_equal(
        np.bincount(shell), np.array([12, 6, 24]) * len(sites)
    )
    np.testing.assert_allclose(
        [nl.distances[shell == s].max() for s in range(3)],
        3.6 * np.sqrt([0.5, 1.0, 1.5]),
    )


def test_binary_fcc_is_random_in_three_shells():
    sites = build_crystal("fcc", 3.6, repeat=(4, 4, 4))
    alloy = sqs_alloy(sites, {"Cu": 1, "Ni": 1}, shells=3, steps=50_000, seed=1)
    kinds = alloy.table["element"].to_list()
    assert kinds.count("Cu") == kinds.count("Ni") == 128
    np.testing.assert_array_equal(alloy.positions, sites.positions)
    np.testing.assert_allclose(warren_cowley(alloy, 3), 0.0, atol=1e-12)
    random = random_alloy(sites, {"Cu": 1, "Ni": 1}, seed=1)
    assert np.abs(warren_cowley(random, 3)).max() > 0.01
    again = sqs_alloy(sites, {"Cu": 1, "Ni": 1}, shells=3, steps=50_000, seed=1)
    assert again.table["element"].to_list() == kinds


@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_kernel_keeps_its_counts(temperature):
    """The objective after many swaps, kept up to date one atom at a time,
    is that of the final configuration counted afresh; in a cell so small
    that atoms are their own neighbors."""
    for repeat in ((1, 1, 1), (3, 3, 3)):
        sites = build_crystal("bcc", 2.87, repeat=repeat)
        nl = _shells(sites, 2)
        shell = _shell_index(nl, 1e-3)
        rng = np.random.default_rng(5)
        start = rng.permutation(np.arange(len(sites)) % 3).astype(np.int32)
        weights = np.array([1.0, 0.5])
        before = _native.sqs_objective(nl.offsets, nl.j, shell, 2, start, 3, weights)
        best, objective, first, accepted = _native.sqs(
            nl.offsets, nl.j, shell, 2, start, 3, weights, 5000, temperature, 7
        )
        assert first == pytest.approx(before, rel=1e-12)
        assert objective <= before and accepted >= 0
        after = _native.sqs_objective(nl.offsets, nl.j, shell, 2, best, 3, weights)
        assert after == pytest.approx(objective, rel=1e-9, abs=1e-15)
        np.testing.assert_array_equal(
            np.bincount(best, minlength=3), np.bincount(start, minlength=3)
        )


def test_ternary_better_than_random():
    sites = build_crystal("fcc", 3.6, repeat=(4, 4, 4))
    fractions = {"Co": 1, "Cr": 1, "Ni": 1}
    random = np.abs(warren_cowley(random_alloy(sites, fractions, seed=2), 2)).max()
    sqs = np.abs(warren_cowley(sqs_alloy(sites, fractions, shells=2, seed=2), 2)).max()
    assert sqs < random / 3


def test_progress_cancel_and_errors():
    sites = build_crystal("fcc", 3.6, repeat=(3, 3, 3))
    reports = []
    sqs_alloy(
        sites,
        {"Cu": 1, "Au": 3},
        steps=1000,
        seed=1,
        progress=lambda d, t: reports.append((d, t)),
    )
    assert reports and reports[-1][0] == reports[-1][1]
    token = mdapy.CancelToken()
    token.cancel()
    with pytest.raises(Cancelled):
        sqs_alloy(sites, {"Cu": 1, "Au": 1}, cancel=token)
    one = sqs_alloy(sites, {"Cu": 1}, steps=100, seed=1)
    assert set(one.table["element"].to_list()) == {"Cu"}
    for arguments, message in [
        ({"shells": 0}, "shells must be from 1 to 10"),
        ({"weights": [1.0]}, "weights must be 3"),
        ({"steps": -1}, "steps must be"),
        ({"temperature": -1.0}, "temperature must not be negative"),
    ]:
        with pytest.raises(InvalidParameter, match=message):
            sqs_alloy(sites, {"Cu": 1, "Ni": 1}, **arguments)
    nl = _shells(sites, 1)
    shell = _shell_index(nl, 1e-3)
    types = np.zeros(len(sites), np.int32)
    with pytest.raises(ValueError, match="do not fit"):
        _native.sqs_objective(nl.offsets, nl.j, shell[:-1], 1, types, 1, np.ones(1))
    with pytest.raises(ValueError, match="types must be"):
        _native.sqs_objective(nl.offsets, nl.j, shell, 1, types + 2, 1, np.ones(1))
    with pytest.raises(ValueError, match="shells must be"):
        _native.sqs_objective(nl.offsets, nl.j, shell + 1, 1, types, 1, np.ones(1))
