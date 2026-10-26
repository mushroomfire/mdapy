# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""A NumPy reference of NEP4 descriptors and energies.

The angular descriptors are computed by another route than mdapy's: the
three-body terms as double sums of Legendre polynomials over pairs of
neighbors, the four- and five-body terms by coupling complex spherical
harmonics with Wigner 3j symbols.
"""

from math import factorial, pi, sqrt

import numpy as np
from numpy.polynomial import chebyshev, legendre

import mdapy
from mdapy.calculators._nep import COVALENT_RADII
from mdapy.elements import SYMBOLS

COULOMB = 14.399645  # e^2 / (4 pi epsilon_0) in eV Å, as in GPUMD
# 1 / 0.46848 Å rounded, as GPUMD has it
INVERSE_SCREENING = 2.134563
UNIVERSAL = (
    (0.18175, 3.1998),
    (0.50986, 0.94229),
    (0.28022, 0.4029),
    (0.02817, 0.20162),
)


def wigner_3j(j1, j2, j3, m1, m2, m3):
    """Racah's formula."""
    if m1 + m2 + m3 != 0 or not abs(j1 - j2) <= j3 <= j1 + j2:
        return 0.0
    if abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
        return 0.0
    f = factorial
    triangle = (
        f(j1 + j2 - j3) * f(j1 - j2 + j3) * f(-j1 + j2 + j3) / f(j1 + j2 + j3 + 1)
    )
    root = sqrt(
        triangle
        * f(j1 + m1) * f(j1 - m1) * f(j2 + m2) * f(j2 - m2) * f(j3 + m3) * f(j3 - m3)
    )  # fmt: skip
    total = 0.0
    for k in range(0, j1 + j2 + j3 + 1):
        terms = (
            k,
            j3 - j2 + k + m1,
            j3 - j1 + k - m2,
            j1 + j2 - j3 - k,
            j1 - k - m1,
            j2 - k + m2,
        )
        if min(terms) < 0:
            continue
        product = 1
        for t in terms:
            product *= f(t)
        total += (-1) ** k / product
    return (-1) ** (j1 - j2 - m3) * root * total


def harmonics(degree, u):
    """Complex spherical harmonics Y_lm, l = degree (1 or 2), m = -l..l, of unit
    vectors (k, 3), with the Condon-Shortley phase; columns ordered by m."""
    x, y, z = u.T
    w = x + 1j * y
    if degree == 1:
        c = sqrt(3 / (4 * pi))
        plus = -c / sqrt(2) * w
        return np.stack([-np.conj(plus), c * z + 0j, plus], axis=1)
    c = sqrt(15 / (8 * pi))
    one, two = -c * w * z, c / 2 * w * w
    zero = sqrt(5 / (16 * pi)) * (3 * z * z - 1) + 0j
    return np.stack([np.conj(two), -np.conj(one), zero, one, two], axis=1)


def cutoff_function(r, rc):
    return np.where(r < rc, 0.5 * np.cos(pi * r / rc) + 0.5, 0.0)


def radial_functions(r, rc, c):
    """g_n(r) for coefficients c (n, k): (n, pairs)."""
    x = 2 * (r / rc - 1) ** 2 - 1
    basis = (
        0.5 * (chebyshev.chebvander(x, c.shape[1] - 1).T + 1) * cutoff_function(r, rc)
    )
    return c @ basis


def zbl_energy(model, zi, zj, ti, tj, r):
    zbl = model["zbl"]
    if zbl["kind"] == "flexible":
        a, b = sorted((ti, tj))
        types = len(model["elements"])
        row = model["zbl"]["table"][a * types - a * (a - 1) // 2 + b - a]
        inner, outer, coefficients = row[0], row[1], row[2:].reshape(4, 2)
    else:
        inner, outer, coefficients = zbl["inner"], zbl["outer"], UNIVERSAL
        if zbl["kind"] == "typewise":
            outer = min(
                outer, (COVALENT_RADII[zi] + COVALENT_RADII[zj]) * zbl["factor"]
            )
            inner = 0.0
    x = r * (zi**0.23 + zj**0.23) * INVERSE_SCREENING
    phi = sum(a * np.exp(-b * x) for a, b in coefficients)
    switch = np.where(
        r < inner,
        1.0,
        np.where(
            r < outer, 0.5 * np.cos(pi * (r - inner) / (outer - inner)) + 0.5, 0.0
        ),
    )
    return COULOMB * zi * zj / r * phi * switch


def descriptors_and_energies(model, system):
    """Scaled descriptors (N, dim) and atomic energies (N,)."""
    types = np.array([model["elements"].index(e) for e in system.table["element"]])
    numbers = np.array([SYMBOLS.index(e) for e in model["elements"]])
    nl = mdapy.NeighborList.build(system, float(model["rc_radial"].max()))
    i_of, vectors, distances = nl.i, nl.vectors, nl.distances
    nr, na, l_max = model["n_radial"], model["n_angular"], model["l_max"]
    terms = l_max + model["four_body"] + model["five_body"]
    q = np.zeros((len(system), nr + 1 + (na + 1) * terms))
    energies = np.zeros(len(system))
    for i in range(len(system)):
        ti = types[i]
        mine = i_of == i
        tj, r, d = types[nl.j[mine]], vectors[mine], distances[mine]
        for t in range(len(model["elements"])):
            rc = 0.5 * (model["rc_radial"][ti] + model["rc_radial"][t])
            keep = (tj == t) & (d < rc)
            q[i, : nr + 1] += radial_functions(
                d[keep], rc, model["c_radial"][ti, t]
            ).sum(1)
        rc = 0.5 * (model["rc_angular"][ti] + model["rc_angular"][tj])
        keep = d < rc
        tj, r, d, rc = tj[keep], r[keep], d[keep], rc[keep]
        g = np.zeros((na + 1, len(d)))
        for t in range(len(model["elements"])):
            same = tj == t
            g[:, same] = radial_functions(d[same], rc[same], model["c_angular"][ti, t])
        u = r / d[:, None]
        cosines = np.clip(u @ u.T, -1.0, 1.0)
        start = nr + 1
        for degree in range(1, l_max + 1):
            p = legendre.legval(cosines, [0] * degree + [1])
            q[i, start : start + na + 1] = (
                (2 * degree + 1) / (4 * pi) * np.einsum("nj,jk,nk->n", g, p, g)
            )
            start += na + 1
        if model["four_body"]:
            a = g @ harmonics(2, u)  # (n, m + 2)
            total = np.zeros(na + 1, dtype=complex)
            for m1 in range(-2, 3):
                for m2 in range(-2, 3):
                    m3 = -m1 - m2
                    if abs(m3) <= 2:
                        total += (
                            wigner_3j(2, 2, 2, m1, m2, m3)
                            * a[:, m1 + 2] * a[:, m2 + 2] * a[:, m3 + 2]
                        )  # fmt: skip
            q[i, start : start + na + 1] = total.real
            start += na + 1
        if model["five_body"]:
            a = g @ harmonics(1, u)
            total = np.zeros(na + 1, dtype=complex)
            for coupled in range(3):
                for m1 in range(-1, 2):
                    for m2 in range(-1, 2):
                        for m3 in range(-1, 2):
                            m = m1 + m2
                            m4 = -m - m3
                            if abs(m) > coupled or abs(m4) > 1:
                                continue
                            total += (
                                (-1) ** (coupled + m)
                                * wigner_3j(1, 1, coupled, m1, m2, -m)
                                * wigner_3j(coupled, 1, 1, m, m3, m4)
                                * a[:, m1 + 1] * a[:, m2 + 1] * a[:, m3 + 1] * a[:, m4 + 1]
                            )  # fmt: skip
            q[i, start : start + na + 1] = total.real
        q[i] *= model["q_scaler"]
        hidden = np.tanh(model["w0"][ti] @ q[i] - model["b0"][ti])
        energies[i] = model["w1"][ti] @ hidden - model["b1"]
        if model["zbl"] is not None:
            for zj_type, rj in zip(tj, d):
                energies[i] += 0.5 * zbl_energy(
                    model, numbers[ti], numbers[zj_type], ti, zj_type, rj
                )
    return q, energies
