# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The Fourier transforms of the radial parts of a pseudopotential, in hartree
atomic units, with the conventions of Quantum ESPRESSO, so that totals compare
to a microhartree:
  - integrals by Simpson's rule on the radial mesh with its weights rab, the
    local parts up to the first point beyond 10 bohr, made odd;
  - the local potential with the long-range -Z/r taken out by -Z erf(r)/r and
    added back in reciprocal space, and its G = 0 limit of the potential plus
    Z/r;
  - the projectors tabulated every 0.01 / bohr and interpolated by Lagrange
    polynomials of four points.
Every transform is per unit volume of the cell: 4 pi / Omega for the local
parts, 4 pi / sqrt(Omega) for the projectors."""

import math

import numpy as np

DQ = 0.01  # 1/bohr, the step of the tables of the projectors
_RCUT = 10.0  # bohr, the reach of the integrals of the local parts


def simpson(f: np.ndarray, rab: np.ndarray, n: int | None = None) -> float:
    """The integral of f by Simpson's rule on a mesh of weights rab, over its
    first n points (all by default), the last left out when n is even."""
    n = len(f) if n is None else n
    g = f[:n] * rab[:n] / 3.0
    if n < 3:
        return 0.0
    last = n - 1 if n % 2 == 1 else n - 2
    return float(
        g[0:last:2].sum() + 4.0 * g[1:last:2].sum() + g[2 : last + 1 : 2].sum()
    )


def reach(r: np.ndarray) -> int:
    """The points of the mesh that the local parts are integrated over: up to
    the first beyond 10 bohr, an odd number."""
    beyond = np.nonzero(r > _RCUT)[0]
    n = int(beyond[0]) + 1 if len(beyond) else len(r)
    return 2 * ((n + 1) // 2) - 1


def bessel(l: int, x: np.ndarray) -> np.ndarray:  # noqa: E741
    """The spherical Bessel function j_l(x), l from 0 to 3: below x = 1 by its
    series to twelve terms, whose cancellation is harmless there, and by its
    closed form beyond, whose cancellation is harmless there."""
    if not 0 <= l <= 3:
        raise ValueError("spherical Bessel functions up to l = 3")
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = np.abs(x) < 1.0
    xs, xl = x[small], x[~small]
    # x^l / (2l+1)!! sum_k (-x^2/2)^k / (k! (2l+3)(2l+5)...(2l+2k+1))
    term = xs**l / [1.0, 3.0, 15.0, 105.0][l]
    total = term.copy()
    for k in range(1, 13):
        term = term * (-0.5 * xs * xs) / (k * (2 * l + 2 * k + 1))
        total += term
    out[small] = total
    s, c = np.sin(xl), np.cos(xl)
    if l == 0:
        out[~small] = s / xl
    elif l == 1:
        out[~small] = s / xl**2 - c / xl
    elif l == 2:
        out[~small] = (3 / xl**3 - 1 / xl) * s - 3 * c / xl**2
    else:
        out[~small] = (15 / xl**4 - 6 / xl**2) * s - (15 / xl**3 - 1 / xl) * c
    return out


def local_potential(pp, q: np.ndarray, volume: float) -> np.ndarray:
    """V_loc(q) of the pseudopotential pp at the moduli q (1/bohr), per unit
    volume of a cell of volume Omega: (4 pi / Omega) [int (r V(r) + Z erf(r))
    sin(qr)/q dr - Z exp(-q^2/4) / q^2], and at q = 0 (4 pi / Omega) int r (r V(r)
    + Z) dr."""
    n = reach(pp.r)
    r, rab, z = pp.r[:n], pp.rab[:n], pp.valence
    out = np.empty(len(q))
    zero = q < 1e-8
    if zero.any():
        out[zero] = simpson(r * (r * pp.local[:n] + z), rab)
    base = r * pp.local[:n] + z * np.vectorize(math.erf)(r)
    for i in np.nonzero(~zero)[0]:
        g = q[i]
        out[i] = simpson(base * np.sin(g * r) / g, rab) - z * math.exp(
            -0.25 * g * g
        ) / (g * g)
    return 4 * math.pi / volume * out


def radial_density(
    f: np.ndarray, pp, q: np.ndarray, volume: float, weight_r2: bool
) -> np.ndarray:
    """int f(r) j0(qr) dr / Omega, times r^2 4 pi when weight_r2: the Fourier
    transform of a density per unit volume, rho_core (f = rho) or the free atom
    (f = 4 pi r^2 rho)."""
    n = reach(pp.r)
    r, rab = pp.r[:n], pp.rab[:n]
    g = f[:n] * (4 * math.pi * r * r if weight_r2 else 1.0)
    return (
        np.array([simpson(g * bessel(0, np.array(qi * r)), rab) for qi in q]) / volume
    )


def projector_table(pp, qmax: float, volume: float) -> np.ndarray:
    """(P, nq) the transforms of the projectors, (4 pi / sqrt(Omega)) int r^2
    beta(r) j_l(qr) dr, every DQ from 0 to qmax and three steps more for the
    interpolation."""
    nq = int(qmax / DQ) + 4
    q = DQ * np.arange(nq)
    table = np.empty((len(pp.projectors), nq))
    for i, p in enumerate(pp.projectors):
        nonzero = np.nonzero(p.r_beta)[0]
        n = int(nonzero[-1]) + 1 if len(nonzero) else 1
        r, rab, rb = pp.r[:n], pp.rab[:n], p.r_beta[:n]
        table[i] = [simpson(rb * r * bessel(p.angular, qi * r), rab) for qi in q]
    return 4 * math.pi / math.sqrt(volume) * table


def interpolate(table: np.ndarray, q: np.ndarray) -> np.ndarray:
    """The values of a table every DQ at the moduli q, by Lagrange polynomials
    of the four points from the one at or below q on: (P, len(q))."""
    x = q / DQ
    i0 = np.floor(x).astype(int)
    p = x - i0
    u, v, w = 1 - p, 2 - p, 3 - p
    t = table
    return (t[:, i0] * (u * v * w / 6) + t[:, i0 + 1] * (p * v * w / 2)
            - t[:, i0 + 2] * (p * u * w / 2) + t[:, i0 + 3] * (p * u * v / 6))  # fmt: skip
