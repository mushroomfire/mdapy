# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The augmentation charges of an ultrasoft pseudopotential on the GPU against
NumPy: Q_ab(G) from the radial tables, the real harmonics and the Gaunt
coefficients; the augmented density of becsum and the strengths of a
potential at atoms off the origin; Omega Q_ab(0) is q_ab of the file; the real
harmonics against their closed forms and the Gaunt coefficients against their
orthonormality."""

import math

import numpy as np
import pytest

from mdapy.calculators._dft import angular, augmentation, radial
from mdapy.calculators._dft.upf import read_upf

from dft import espresso, gpu

_dft = gpu.module()
if _dft.device_count() == 0:
    pytest.skip("no GPU", allow_module_level=True)

BOHR = 0.529177210903


def test_harmonics_and_gaunt():
    rng = np.random.default_rng(0)
    y = angular.real_harmonics(6, rng.normal(size=(2000, 3)))
    # Gauss-Legendre times a uniform angle integrate their products exactly
    g = angular.gaunt(3)
    small = 16
    np.testing.assert_allclose(
        g[:, :, 0] * math.sqrt(4 * math.pi), np.eye(small), atol=1e-13
    )
    assert y.shape == (49, 2000)
    # symmetric in the first two, and zero unless l1 + l2 + L is even
    np.testing.assert_allclose(g, g.transpose(1, 0, 2), atol=1e-14)
    for l1 in range(4):
        for l2 in range(4):
            for ell in range(7):
                if (l1 + l2 + ell) % 2 or ell > l1 + l2 or ell < abs(l1 - l2):
                    block = g[
                        l1 * l1 : (l1 + 1) ** 2,
                        l2 * l2 : (l2 + 1) ** 2,
                        ell * ell : (ell + 1) ** 2,
                    ]
                    assert np.abs(block).max() < 1e-13


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
