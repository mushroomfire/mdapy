# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The augmentation charges of an ultrasoft pseudopotential as the GPU takes
them, on the processor: the projector functions and their pairs, and the terms
of each pair, whose Gaunt coefficients give back the products of two
harmonics and whose sum at G = 0 is q_ab / Omega of the file; and on the GPU
against NumPy: Q_ab(G) from the radial tables, the real harmonics and the
Gaunt coefficients; the augmented density of becsum and the strengths of a
potential at atoms off the origin; Omega Q_ab(0) is q_ab of the file."""

import math

import numpy as np
import pytest

from mdapy.calculators._dft import angular, augmentation, radial
from mdapy.calculators._dft.upf import read_upf

from dft import espresso, gpu, public, upf_files

BOHR = 0.529177210903


def synthetic(tmp_path, kind):
    return read_upf(upf_files.write(tmp_path / "X.upf", kind))


def check_terms(pp, tolerance=1e-9, volume=250.0, qmax=6.0):
    """The terms of pp against the harmonics, the radial tables and q_ab of the
    file, to tolerance."""
    functions = augmentation.functions(pp)
    assert functions == [(p, b.angular, m) for p, b in enumerate(pp.projectors)
                         for m in range(-b.angular, b.angular + 1)]  # fmt: skip
    pairs = augmentation.pairs(pp)
    n = len(functions)
    assert pairs == [(a, b) for a in range(n) for b in range(n) if a <= b]
    data = augmentation.terms(pp, qmax, volume)
    keys, tables = radial.augmentation_tables(pp, qmax, volume)
    np.testing.assert_array_equal(data["tables"], tables)
    assert data["tables"].flags.c_contiguous
    assert data["functions"] == n and data["lmax"] == 2 * pp.lmax
    assert data["points"] == tables.shape[1] == int(qmax / radial.DQ) + 4
    start = data["pair_start"]
    assert len(start) == len(pairs) + 1 and start[0] == 0
    assert start[-1] == len(data["term_lm"]) and (np.diff(start) >= 0).all()
    rng = np.random.default_rng(6)
    v = rng.normal(size=(30, 3))
    y = angular.real_harmonics(2 * pp.lmax, v)
    for k, (a, b) in enumerate(pairs):
        (pa, la, ma), (pb, lb, mb) = functions[a], functions[b]
        terms = range(start[k], start[k + 1])
        for t in terms:
            ell = data["term_l"][t]
            assert math.isqrt(data["term_lm"][t]) == ell
            assert keys[data["term_table"][t]] == (min(pa, pb), max(pa, pb), ell)
        # with a charge for every L, the terms give back Y_a Y_b
        present = {ell for ell in range(abs(la - lb), la + lb + 1, 2)
                   if (min(pa, pb), max(pa, pb), ell) in pp.augmentation}  # fmt: skip
        if len(present) == (la + lb - abs(la - lb)) // 2 + 1:
            total = sum(data["term_coef"][t] * y[data["term_lm"][t]] for t in terms)
            product = y[angular.index(la, ma)] * y[angular.index(lb, mb)]
            np.testing.assert_allclose(total, product, rtol=0, atol=1e-12)
        # Omega Q_ab(G = 0): the terms of L = 0 alone, q_ab for one l and m
        zero = sum(data["term_coef"][t] * data["tables"][data["term_table"][t], 0]
                   for t in terms if data["term_l"][t] == 0) / math.sqrt(4 * math.pi)  # fmt: skip
        expected = pp.q[pa, pb] if (la, ma) == (lb, mb) else 0.0
        assert volume * zero == pytest.approx(expected, abs=tolerance)
    return data


def test_the_terms_of_an_ultrasoft_file(tmp_path):
    pp = synthetic(tmp_path, "US")
    data = check_terms(pp)
    # the projectors s, s, p, d: 1 + 1 + 3 + 5 functions, all their pairs
    assert data["functions"] == 10 and len(data["pair_start"]) == 56


def test_the_terms_of_a_paw_file(tmp_path):
    """The charge of a vanishing multipole is left out of its pair, which
    keeps the terms of the other L."""
    pp = synthetic(tmp_path, "PAW")
    data = check_terms(pp)
    i, j, ell = upf_files.VANISHING
    assert (i, j, ell) not in pp.augmentation
    functions = augmentation.functions(pp)
    pairs = augmentation.pairs(pp)
    for k, (a, b) in enumerate(pairs):
        if sorted((functions[a][0], functions[b][0])) == [i, j]:
            ls = data["term_l"][data["pair_start"][k] : data["pair_start"][k + 1]]
            assert ell not in ls and set(ls) <= {1}


@pytest.mark.parametrize(
    ("name", "tolerance"),
    [
        ("Al.pbe-n-rrkjus_psl.1.0.0.UPF", 1e-12),
        ("Fe.pbe-spn-kjpaw_psl.1.0.0.UPF", 1e-6),
    ],
)
def test_the_terms_of_pslibrary(name, tolerance):
    """q_ab of the files, to the agreement of their two parts: the pseudized
    charges of PAW keep their multipoles to 5e-7 (test_upf)."""
    check_terms(read_upf(public.path(name)), tolerance)


def reference_q(pp, sphere_g, tables, data):
    """Q_ab(G) of NumPy, (n, pairs)."""
    q = np.linalg.norm(sphere_g, axis=1)
    y = angular.real_harmonics(data["lmax"], sphere_g)
    values = radial.interpolate(tables, q)  # (tables, n)
    out = np.zeros((len(q), len(data["pair_start"]) - 1), dtype=complex)
    for pair in range(out.shape[1]):
        for k in range(data["pair_start"][pair], data["pair_start"][pair + 1]):
            out[:, pair] += ((-1j) ** data["term_l"][k] * data["term_coef"][k]
                             * values[data["term_table"][k]] * y[data["term_lm"][k]])  # fmt: skip
    return out


def test_iron():
    _dft = gpu.device()
    path = espresso.pseudo("pslibrary/Fe.pbe-spn-rrkjus_psl.1.0.0.UPF")
    pp = read_upf(path)
    a = 2.87 / BOHR
    cell = np.diag([a, a, a])
    cutoff = 60.0  # hartree, of the density
    grid = _dft.fft_grid(cell, cutoff)
    volume = a**3
    data = augmentation.terms(pp, math.sqrt(2 * cutoff) + 1, volume)
    positions = np.array([[0.1, 0.2, -0.05], [0.53, 0.47, 0.51]])
    aug = _dft.Augmentation(
        cell, cutoff, list(grid), positions, np.zeros(2, np.int32), [data]
    )
    miller, _, _, _ = _dft.sphere(cell, [0, 0, 0], cutoff, grid)
    g = miller @ (2 * math.pi * np.linalg.inv(cell).T)
    ours = aug.components(0)
    reference = reference_q(pp, g, data["tables"], data)
    np.testing.assert_allclose(ours, reference, atol=1e-12)
    # Omega Q_ab(0) is q of the file for functions of one l and m
    f = augmentation.functions(pp)
    pairs = augmentation.pairs(pp)
    q0 = aug.overlaps(0)
    for k, (i, j) in enumerate(pairs):
        (pa, la, ma), (pb, lb, mb) = f[i], f[j]
        expected = pp.q[pa, pb] if (la, ma) == (lb, mb) else 0.0
        assert q0[k] == pytest.approx(expected, abs=2e-6)
    # the augmented density and the strengths at the atoms
    rng = np.random.default_rng(1)
    becsum = rng.normal(size=2 * len(pairs))
    phases = np.exp(-2j * math.pi * (miller @ positions.T))  # (n, atoms)
    rho = aug.add_density(becsum, len(miller))
    expected = (reference * phases[:, :1]) @ becsum[: len(pairs)] + (
        reference * phases[:, 1:]
    ) @ becsum[len(pairs) :]
    np.testing.assert_allclose(rho, expected, atol=1e-12)
    assert volume * rho[0].real == pytest.approx(
        becsum[: len(pairs)] @ q0 + becsum[len(pairs) :] @ q0, rel=1e-12
    )
    v = rng.normal(size=len(miller)) + 1j * rng.normal(size=len(miller))
    d = aug.strengths(v)
    expected = np.concatenate([volume * np.real((v.conj()[:, None] * reference * phases[:, i : i + 1]).sum(0))
                               for i in range(2)])  # fmt: skip
    np.testing.assert_allclose(d, expected, rtol=1e-10, atol=1e-12)
