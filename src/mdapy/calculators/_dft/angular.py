# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Real spherical harmonics and their Gaunt coefficients, in the convention of
the GPU's: Y_lm of l from 0 and m from -l to l, the sines for m < 0 and the
cosines for m > 0,

  Y_l0 = N_l0 P_l(cos t),  Y_lm = sqrt(2) N_lm P_l^|m|(cos t) cos(m p) or sin(|m| p),
  N_lm = sqrt((2l + 1) / (4 pi) (l - |m|)! / (l + |m|)!),

P_l^m without the phase of Condon and Shortley, so that Y_11 = sqrt(3 / 4 pi) x.
The Gaunt coefficient of three is the integral of their product over the
sphere, exact here by Gauss-Legendre in cos t and a uniform grid in p."""

import math
from functools import lru_cache

import numpy as np


def index(l: int, m: int) -> int:  # noqa: E741
    """The place of (l, m) among all of l' <= l: l^2 + l + m."""
    return l * l + l + m


def real_harmonics(lmax: int, directions: np.ndarray) -> np.ndarray:
    """((lmax + 1)^2, n) the real Y_lm of the directions (n, 3), unit or not;
    the zero vector has Y_00 alone."""
    d = np.asarray(directions, dtype=float)
    r = np.linalg.norm(d, axis=1)
    safe = np.where(r > 1e-300, r, 1.0)
    x, y, z = (d / safe[:, None]).T
    rho = np.hypot(x, y)
    phi = np.arctan2(y, x)
    out = np.zeros(((lmax + 1) ** 2, len(d)))
    # associated Legendre functions P_l^m(z), without the phase, by recursion
    p = {(0, 0): np.ones_like(z)}
    for m in range(0, lmax + 1):
        if m > 0:
            p[(m, m)] = p[(m - 1, m - 1)] * (2 * m - 1) * rho
        if m + 1 <= lmax:
            p[(m + 1, m)] = z * (2 * m + 1) * p[(m, m)]
        for l in range(m + 2, lmax + 1):  # noqa: E741
            p[(l, m)] = (
                (2 * l - 1) * z * p[(l - 1, m)] - (l + m - 1) * p[(l - 2, m)]
            ) / (l - m)
    for l in range(lmax + 1):  # noqa: E741
        for m in range(-l, l + 1):
            a = abs(m)
            norm = math.sqrt(
                (2 * l + 1)
                / (4 * math.pi)
                * math.factorial(l - a)
                / math.factorial(l + a)
            )
            if m == 0:
                value = norm * p[(l, 0)]
            elif m > 0:
                value = math.sqrt(2) * norm * p[(l, a)] * np.cos(a * phi)
            else:
                value = math.sqrt(2) * norm * p[(l, a)] * np.sin(a * phi)
            out[index(l, m)] = np.where(
                r > 1e-300, value, 1.0 / math.sqrt(4 * math.pi) if l == 0 else 0.0
            )
    return out


@lru_cache(maxsize=None)
def gaunt(lmax: int) -> np.ndarray:
    """((lmax + 1)^2, (lmax + 1)^2, (2 lmax + 1)^2) the integrals of
    Y_l1m1 Y_l2m2 Y_LM over the sphere, for l1, l2 <= lmax."""
    big = 2 * lmax
    nodes, weights = np.polynomial.legendre.leggauss(big + lmax + 2)
    nphi = 2 * (big + lmax) + 2
    phi = 2 * math.pi * np.arange(nphi) / nphi
    ct, pp = np.meshgrid(nodes, phi, indexing="ij")
    st = np.sqrt(1 - ct * ct)
    directions = np.stack([st * np.cos(pp), st * np.sin(pp), ct], -1).reshape(-1, 3)
    w = (weights[:, None] * np.full(nphi, 2 * math.pi / nphi)[None, :]).ravel()
    y = real_harmonics(big, directions)
    small = (lmax + 1) ** 2
    return np.einsum("an,bn,cn,n->abc", y[:small], y[:small], y, w)
