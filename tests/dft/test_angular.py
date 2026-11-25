# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The real spherical harmonics and their Gaunt coefficients of the DFT, on the
processor: the harmonics against their closed forms in x, y and z and against
the addition theorem, orthonormal; the Gaunt coefficients against the Wigner
3j symbols of m = 0 in closed form and as the coefficients of the products of
two harmonics, which they give back at any direction."""

import math

import numpy as np
import pytest

from mdapy.calculators._dft import angular


def unit(rng, n):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1)[:, None]


def test_index():
    assert [angular.index(ell, m) for ell in range(3) for m in range(-ell, ell + 1)] == list(range(9))  # fmt: skip


def test_closed_forms():
    """Without the phase of Condon and Shortley: Y_11 is +sqrt(3 / 4 pi) x."""
    rng = np.random.default_rng(1)
    x, y, z = unit(rng, 50).T
    c1, c2 = math.sqrt(3 / (4 * math.pi)), math.sqrt(15 / (4 * math.pi))
    exact = [
        np.full_like(x, 1 / math.sqrt(4 * math.pi)),
        c1 * y, c1 * z, c1 * x,
        c2 * x * y, c2 * y * z, math.sqrt(5 / (16 * math.pi)) * (3 * z * z - 1),
        c2 * x * z, math.sqrt(15 / (16 * math.pi)) * (x * x - y * y),
        math.sqrt(35 / (32 * math.pi)) * y * (3 * x * x - y * y),
        math.sqrt(7 / (16 * math.pi)) * (5 * z**3 - 3 * z),
        math.sqrt(35 / (32 * math.pi)) * x * (x * x - 3 * y * y),
    ]  # fmt: skip
    values = angular.real_harmonics(3, np.stack([x, y, z], 1))
    for k, lm in enumerate([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15]):
        np.testing.assert_allclose(values[lm], exact[k], rtol=0, atol=1e-14)


def test_the_addition_theorem():
    """sum_m Y_lm(u) Y_lm(v) = (2l + 1) / (4 pi) P_l(u . v) up to l = 8."""
    rng = np.random.default_rng(2)
    u, v = unit(rng, 40), unit(rng, 40)
    yu, yv = angular.real_harmonics(8, u), angular.real_harmonics(8, v)
    cosine = (u * v).sum(1)
    for ell in range(9):
        part = slice(ell * ell, (ell + 1) ** 2)
        legendre = np.polynomial.legendre.Legendre.basis(ell)(cosine)
        np.testing.assert_allclose((yu[part] * yv[part]).sum(0),
                                   (2 * ell + 1) / (4 * math.pi) * legendre, rtol=0, atol=1e-13)  # fmt: skip


def test_lengths_and_the_zero_vector():
    """Directions of any length give the harmonics of their unit vectors; the
    zero vector has Y_00 alone."""
    rng = np.random.default_rng(3)
    v = unit(rng, 10)
    scaled = v * rng.uniform(0.1, 30.0, (10, 1))
    np.testing.assert_allclose(angular.real_harmonics(4, scaled),
                               angular.real_harmonics(4, v), rtol=0, atol=1e-14)  # fmt: skip
    zero = angular.real_harmonics(4, np.zeros((2, 3)))
    np.testing.assert_allclose(zero[0], 1 / math.sqrt(4 * math.pi))
    assert not zero[1:].any()


def test_orthonormal():
    """On 20000 directions of Gauss-Legendre in cos(theta) and a uniform phi,
    exact for these degrees."""
    nodes, weights = np.polynomial.legendre.leggauss(40)
    phi = 2 * math.pi * np.arange(80) / 80
    ct, pp = np.meshgrid(nodes, phi, indexing="ij")
    st = np.sqrt(1 - ct * ct)
    v = np.stack([st * np.cos(pp), st * np.sin(pp), ct], -1).reshape(-1, 3)
    w = (weights[:, None] * np.full(80, 2 * math.pi / 80)).ravel()
    y = angular.real_harmonics(6, v)
    np.testing.assert_allclose((y * w) @ y.T, np.eye(49), rtol=0, atol=1e-13)


def three_j_zero(l1, l2, l3):
    """The Wigner 3j symbol (l1 l2 l3; 0 0 0) in closed form."""
    j = l1 + l2 + l3
    if j % 2 or l3 > l1 + l2 or l3 < abs(l1 - l2):
        return 0.0
    g = j // 2
    f = math.factorial
    return (-1) ** g * math.sqrt(f(j - 2 * l1) * f(j - 2 * l2) * f(j - 2 * l3) / f(j + 1)) * (
        f(g) / (f(g - l1) * f(g - l2) * f(g - l3))
    )  # fmt: skip


def test_gaunt_of_m_zero():
    """int Y_l10 Y_l20 Y_L0 = sqrt((2 l1 + 1)(2 l2 + 1)(2L + 1) / 4 pi)
    (l1 l2 L; 0 0 0)^2, such as 1 / sqrt(5 pi) of l1 = l2 = 1, L = 2."""
    g = angular.gaunt(3)
    assert g[angular.index(1, 0), angular.index(1, 0), angular.index(2, 0)] == pytest.approx(
        1 / math.sqrt(5 * math.pi), rel=1e-13
    )  # fmt: skip
    for l1 in range(4):
        for l2 in range(4):
            for ell in range(7):
                exact = math.sqrt((2 * l1 + 1) * (2 * l2 + 1) * (2 * ell + 1) / (4 * math.pi)) * three_j_zero(l1, l2, ell) ** 2  # fmt: skip
                value = g[
                    angular.index(l1, 0), angular.index(l2, 0), angular.index(ell, 0)
                ]
                assert value == pytest.approx(exact, abs=1e-14)


def test_the_products_of_two_harmonics():
    """Y_a Y_b = sum_c G(a, b, c) Y_c at any direction, the products of
    degree up to 2 lmax lying in the harmonics of l up to 2 lmax."""
    rng = np.random.default_rng(4)
    v = unit(rng, 200)
    g = angular.gaunt(3)
    small = angular.real_harmonics(3, v)
    big = angular.real_harmonics(6, v)
    products = small[:, None, :] * small[None, :, :]
    np.testing.assert_allclose(np.einsum("abc,cn->abn", g, big), products, rtol=0, atol=1e-13)  # fmt: skip


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
