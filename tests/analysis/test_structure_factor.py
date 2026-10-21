# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

import numpy as np
import pytest
from lattices import crystal

import mdapy
from mdapy.analysis import StructureFactor
from mdapy.errors import InvalidParameter


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
        [np.exp(1j * system.positions[elements == e] @ q.T).sum(0) for e in names]
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
    for x, first in enumerate(names):
        for y, second in enumerate(names):
            np.testing.assert_allclose(
                result.pair(first, second)[filled],
                sums[x, y][filled]
                / np.sqrt(counts[first] * counts[second])
                / vectors[filled],
                rtol=1e-9,
                atol=1e-12,
            )
    # the partial factors recombine to the total one
    c = {e: counts[e] / len(system) for e in names}
    total = sum(np.sqrt(c[x] * c[y]) * result.pair(x, y) for x in names for y in names)
    np.testing.assert_allclose(total[filled], result.s[filled], rtol=1e-10)
    with pytest.raises(InvalidParameter, match="no partial factor"):
        result.pair("Cu", "Fe")


def test_ideal_gas_threads_and_errors():
    rng = np.random.default_rng(3)
    gas = mdapy.System(rng.random((3000, 3)) * 30, mdapy.Box(30.0), elements="Ar")
    factor = StructureFactor(8.0, bins=40)
    result = factor.compute(gas)
    assert np.nanmean(result.s[5:]) == pytest.approx(1.0, abs=0.02)
    with mdapy.num_threads(3):
        np.testing.assert_array_equal(factor.compute(gas).s, result.s)
    assert (
        StructureFactor(2.0).compute(mdapy.System(gas.positions, gas.box)).partial == {}
    )
    assert repr(factor) == "StructureFactor(q_max=8.0, bins=40)"
    with pytest.raises(InvalidParameter, match="periodic in every direction"):
        factor.compute(
            mdapy.System(gas.positions, mdapy.Box(30.0, pbc=(True, True, False)))
        )
    with pytest.raises(InvalidParameter, match="q_max must be positive"):
        StructureFactor(0.0)
    with pytest.raises(InvalidParameter, match="bins must be at least 1"):
        StructureFactor(5.0, bins=0)


def test_blocks_of_atoms():
    """Atoms are summed in blocks that keep the phase tables small."""
    from mdapy._core import analysis as native

    system = crystal("fcc", 3.6, (3, 3, 3), jitter=0.2, seed=5)
    args = (
        system.positions,
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
