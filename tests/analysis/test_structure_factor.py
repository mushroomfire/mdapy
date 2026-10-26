# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

import numpy as np
import pytest

import mdapy
from mdapy.analysis import StructureFactor
from mdapy.errors import InvalidParameter

from lattices import crystal


def reference(system, q_max, bins, names):
    """Sums over the wave vectors of the half space, written out with NumPy."""
    m = system.box.matrix
    b = 2 * np.pi * np.linalg.inv(m).T
    reach = np.ceil(q_max * np.linalg.norm(m, axis=1) / (2 * np.pi)).astype(int)
    hkl = (
        np.array(np.meshgrid(*[np.arange(-r, r + 1) for r in reach], indexing="ij"))
        .reshape(3, -1)
        .T
    )
    half = (
        (hkl[:, 0] > 0)
        | ((hkl[:, 0] == 0) & (hkl[:, 1] > 0))
        | ((hkl[:, 0] == 0) & (hkl[:, 1] == 0) & (hkl[:, 2] > 0))
    )
    q = hkl[half] @ b
    length = np.linalg.norm(q, axis=1)
    q, length = q[length <= q_max], length[length <= q_max]
    which = np.minimum((length / q_max * bins).astype(int), bins - 1)
    elements = system.table["element"].to_numpy()
    rho = np.stack(
        [np.exp(1j * system["positions"][elements == e] @ q.T).sum(0) for e in names]
    )
    sums = np.zeros((len(names), len(names), bins))
    for a in range(len(names)):
        for c in range(len(names)):
            np.add.at(sums[a, c], which, (rho[a] * np.conj(rho[c])).real)
    return sums, np.bincount(which, minlength=bins)


def test_bragg_peaks_of_fcc():
    a = 3.6
    system = crystal("fcc", a, (4, 4, 4))
    result = StructureFactor(6.0, bins=300).compute(system)
    peaks = {
        (1, 1, 1): 8,
        (2, 0, 0): 6,
        (2, 2, 0): 12,
        (3, 1, 1): 24,
    }  # vectors per family
    expected = np.zeros(300)
    for hkl, count in peaks.items():
        k = int(2 * np.pi * np.linalg.norm(hkl) / a / 6.0 * 300)
        expected[k] += count // 2 * len(system)  # one of each pair q, -q
    filled = result.vectors > 0
    np.testing.assert_allclose(
        result.s[filled] * result.vectors[filled], expected[filled], atol=1e-6
    )
    assert np.isnan(result.s[~filled]).all()


def test_matches_numpy():
    system = crystal("fcc", 3.6, (2, 2, 3), jitter=0.2, seed=1)
    rng = np.random.default_rng(2)
    system = system.with_columns(element=rng.choice(["Cu", "Ni", "Al"], len(system)))
    result = StructureFactor(4.0, bins=25).compute(system)
    names = ("Al", "Ni", "Cu")  # by atomic number
    sums, vectors = reference(system, 4.0, 25, names)
    np.testing.assert_array_equal(result.vectors, vectors)
    filled = vectors > 0
    np.testing.assert_allclose(
        result.s[filled],
        sums.sum(axis=(0, 1))[filled] / len(system) / vectors[filled],
        rtol=1e-10,
    )
    counts = {e: np.sum(system.table["element"].to_numpy() == e) for e in names}
    c = {e: counts[e] / len(system) for e in names}
    for x, first in enumerate(names):
        for y, second in enumerate(names):
            expected = sums[x, y][filled] / len(system) / vectors[filled]
            np.testing.assert_allclose(
                result.pair(first, second)[filled], expected, rtol=1e-9, atol=1e-12
            )
            if x <= y:
                np.testing.assert_allclose(
                    result.faber_ziman[first, second][filled],
                    (expected - (c[first] if x == y else 0.0)) / (c[first] * c[second])
                    + 1.0,
                    rtol=1e-9,
                    atol=1e-12,
                )
    check_sum_rule(result)
    with pytest.raises(InvalidParameter, match="no partial factor"):
        result.pair("Cu", "Fe")


def check_sum_rule(result):
    """The partial factors S_ab add up to the total one, S_ab twice for a != b."""
    total = sum(
        (1 if a == b else 2) * value for (a, b), value in result.partial.items()
    )
    filled = np.isfinite(result.s)
    np.testing.assert_allclose(total[filled], result.s[filled], rtol=1e-10, atol=1e-12)


def test_ideal_gas_threads_and_errors():
    rng = np.random.default_rng(3)
    gas = mdapy.System(rng.random((3000, 3)) * 30, mdapy.Box(30.0), elements="Ar")
    factor = StructureFactor(8.0, bins=40)
    result = factor.compute(gas)
    assert np.nanmean(result.s[5:]) == pytest.approx(1.0, abs=0.02)
    with mdapy.num_threads(3):
        np.testing.assert_array_equal(factor.compute(gas).s, result.s)
    assert (
        StructureFactor(2.0).compute(mdapy.System(gas["positions"], gas.box)).partial
        == {}
    )
    assert repr(factor) == "StructureFactor(q_max=8.0, bins=40, method='direct')"
    with pytest.raises(InvalidParameter, match="periodic in every direction"):
        factor.compute(
            mdapy.System(gas["positions"], mdapy.Box(30.0, pbc=(True, True, False)))
        )
    with pytest.raises(InvalidParameter, match="q_max must be positive"):
        StructureFactor(0.0)
    with pytest.raises(InvalidParameter, match="bins must be at least 1"):
        StructureFactor(5.0, bins=0)
    with pytest.raises(InvalidParameter, match="method must be one of"):
        StructureFactor(5.0, method="fft")
    with pytest.raises(InvalidParameter, match="Debye method only"):
        StructureFactor(5.0, r_max=10.0)
    with pytest.raises(InvalidParameter, match="Debye method only"):
        StructureFactor(5.0, method="debye", r_max=0.0)
    with pytest.raises(InvalidParameter, match="takes no neighbor list"):
        factor.compute(gas, neighbors=mdapy.NeighborList.build(gas, 3.0))


def test_blocks_of_atoms():
    """Atoms are summed in blocks that keep the phase tables small."""
    from mdapy._core import analysis as native

    system = crystal("fcc", 3.6, (3, 3, 3), jitter=0.2, seed=5)
    args = (
        system["positions"],
        system.box.matrix,
        np.zeros(len(system), np.int32),
        1,
        5.0,
        20,
    )
    whole, vectors = native.structure_factor(*args, 0)
    for block in (1, 7, 50):
        parts, counted = native.structure_factor(*args, block)
        np.testing.assert_array_equal(counted, vectors)
        np.testing.assert_allclose(parts, whole, rtol=1e-12)


def debye_reference(system, q, r_max, names):
    """The Debye transform of g(r) from all pair distances up to r_max,
    found by brute force over the periodic images, with NumPy."""
    m, x = system.box.matrix, system["positions"]
    reach = np.ceil(r_max / system.box.thickness).astype(int)
    images = (
        np.array(np.meshgrid(*[np.arange(-r, r + 1) for r in reach], indexing="ij"))
        .reshape(3, -1)
        .T
        @ m
    )
    elements = system.table["element"].to_numpy()
    bins = int(np.ceil(r_max / 0.01))
    edges = np.linspace(0.0, r_max, bins + 1)
    r = 0.5 * (edges[1:] + edges[:-1])
    shells = 4 / 3 * np.pi * np.diff(edges**3)
    n, volume = len(system), system.box.volume
    counts = {}
    for i in range(n):
        # x_j - x_i + image, in this order as every pair vector of mdapy
        d = np.linalg.norm((x[None, :, :] - x[i]) + images[:, None, :], axis=2)
        mask = (d > 0) & (d < r_max)
        for b in names:
            near = d[mask & (elements == b)[None, :]]
            # bin k holds k dr <= r < (k + 1) dr, as the functions are defined
            hist = np.bincount((near / r_max * bins).astype(int), minlength=bins)
            counts[elements[i], b] = counts.get((elements[i], b), 0) + hist
    kernel = (
        4 * np.pi * n / volume * (edges[1] - edges[0])
        * np.sin(np.outer(q, r)) * r * np.sinc(r / r_max) / q[:, None]
    )  # fmt: skip
    total = sum(counts.values()) * volume / (n * n * shells)
    sizes = {e: np.sum(elements == e) for e in names}
    fz = {
        (a, b): 1
        + kernel @ (counts[a, b] * volume / (sizes[a] * sizes[b] * shells) - 1)
        for a in names
        for b in names
    }
    return 1 + kernel @ (total - 1), fz


def test_debye_matches_numpy():
    system = crystal("fcc", 3.6, (2, 2, 2), jitter=0.3, seed=7)
    rng = np.random.default_rng(8)
    system = system.with_columns(element=rng.choice(["Cu", "Ni"], len(system)))
    result = StructureFactor(6.0, bins=30, method="debye", r_max=9.0).compute(system)
    s, fz = debye_reference(system, result.q, 9.0, ("Ni", "Cu"))
    assert result.vectors is None and result.method == "debye"
    np.testing.assert_allclose(result.s, s, rtol=1e-9, atol=1e-12)
    for pair, value in result.faber_ziman.items():
        np.testing.assert_allclose(value, fz[pair], rtol=1e-9, atol=1e-12)
    check_sum_rule(result)
    # a neighbor list gives the same, and r_max defaults to half the box
    nl = mdapy.NeighborList.build(system, 9.0)
    given = StructureFactor(6.0, bins=30, method="debye", r_max=9.0).compute(
        system, neighbors=nl
    )
    np.testing.assert_allclose(given.s, result.s, rtol=1e-12)
    half = StructureFactor(6.0, bins=30, method="debye").compute(system)
    np.testing.assert_allclose(
        half.s, debye_reference(system, half.q, 3.6, ("Ni", "Cu"))[0], rtol=1e-9
    )
    assert repr(StructureFactor(6.0, method="debye", r_max=9.0)) == (
        "StructureFactor(q_max=6.0, bins=200, method='debye', r_max=9.0)"
    )


def test_debye_of_an_ideal_gas():
    rng = np.random.default_rng(4)
    gas = mdapy.System(rng.random((3000, 3)) * 30, mdapy.Box(30.0), elements="Ar")
    result = StructureFactor(8.0, bins=40, method="debye", r_max=10.0).compute(gas)
    assert np.abs(result.s[10:] - 1.0).max() < 0.05
