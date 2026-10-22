# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

import numpy as np
import pytest
from lattices import crystal

import mdapy
from mdapy.analysis import EntropyFingerprint
from mdapy.errors import InvalidParameter


def reference(system, cutoff, sigma, local=False, average=None):
    """The fingerprint written out with NumPy on the same grid."""
    nl = mdapy.NeighborList.build(system, max(cutoff, average or 0.0))
    n = len(system)
    r_k = np.arange(1, int(cutoff / sigma) + 1) * sigma
    values = np.zeros(n)
    for i in range(n):
        d = nl.distances[nl.offsets[i] : nl.offsets[i + 1]]
        d = d[d < cutoff]
        rho = len(d) / (4 / 3 * np.pi * cutoff**3) if local else n / system.box.volume
        gauss = np.exp(-((d[None, :] - r_k[:, None]) ** 2) / (2 * sigma**2)).sum(1)
        g = gauss / (4 * np.pi * rho * r_k**2 * np.sqrt(2 * np.pi) * sigma)
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(g < 1e-10, 1.0, g * np.log(g) - g + 1) * r_k**2
        term[-1] *= 0.5  # the trapezoid rule; the point r = 0 adds nothing
        values[i] = -2 * np.pi * rho * term.sum() * sigma
    if average is not None:
        near = nl.within(average) if nl.cutoff > average else nl
        values = (values + np.bincount(near.i, weights=values[near.j], minlength=n)) / (
            1 + near.counts
        )
    return values


@pytest.mark.parametrize(
    "options",
    [{}, {"local_density": True}, {"average_cutoff": 3.0}, {"average_cutoff": 6.0}],
)
def test_matches_numpy(options):
    system = crystal("fcc", 3.6, (3, 3, 3), jitter=0.2, seed=1)
    result = EntropyFingerprint(5.0, 0.25, **options).compute(system)
    expected = reference(
        system,
        5.0,
        0.25,
        options.get("local_density", False),
        options.get("average_cutoff"),
    )
    np.testing.assert_allclose(result.entropy, expected, rtol=1e-12)
    assert result.average == ("average_cutoff" in options)


def test_order_lowers_the_entropy():
    perfect = EntropyFingerprint(5.0).compute(crystal("fcc", 3.6, (4, 4, 4))).entropy
    np.testing.assert_allclose(perfect, perfect[0], rtol=1e-13)  # every atom alike
    hot = (
        EntropyFingerprint(5.0)
        .compute(crystal("fcc", 3.6, (4, 4, 4), jitter=0.3))
        .entropy
    )
    rng = np.random.default_rng(2)
    gas = mdapy.System(rng.random((2000, 3)) * 30, mdapy.Box(30.0), elements="Ar")
    random = EntropyFingerprint(5.0).compute(gas).entropy
    assert perfect[0] < hot.mean() < random.mean() < 0


def test_options_threads_and_columns():
    system = crystal("bcc", 2.87, (3, 3, 3), jitter=0.1, seed=3)
    fingerprint = EntropyFingerprint(4.5, 0.2, average_cutoff=2.6)
    result = fingerprint.compute(system)
    with mdapy.num_threads(3):
        np.testing.assert_array_equal(
            fingerprint.compute(system).entropy, result.entropy
        )
    nl = mdapy.NeighborList.build(system, 6.0)
    np.testing.assert_allclose(
        fingerprint.compute(system, neighbors=nl).entropy, result.entropy, rtol=1e-14
    )
    # with the local density too, the grid and a given list agree
    local = EntropyFingerprint(4.5, 0.2, local_density=True)
    np.testing.assert_allclose(
        local.compute(system, neighbors=nl).entropy,
        local.compute(system).entropy,
        rtol=1e-14,
    )
    assert list(system.with_columns(result).table.columns)[-1] == "entropy_average"
    assert list(EntropyFingerprint(4.5).compute(system).columns()) == ["entropy"]
    assert "average_cutoff=2.6" in repr(fingerprint)
    with pytest.raises(InvalidParameter, match="below the required"):
        fingerprint.compute(system, neighbors=mdapy.NeighborList.build(system, 3.0))


@pytest.mark.parametrize(
    "arguments, message",
    [
        ((0.0,), "cutoff must be positive"),
        ((5.0, 0.0), "sigma must be positive and below cutoff"),
        ((5.0, 6.0), "sigma must be positive and below cutoff"),
    ],
)
def test_invalid_options(arguments, message):
    with pytest.raises(InvalidParameter, match=message):
        EntropyFingerprint(*arguments)
    with pytest.raises(InvalidParameter, match="average_cutoff must be positive"):
        EntropyFingerprint(5.0, average_cutoff=-1.0)
