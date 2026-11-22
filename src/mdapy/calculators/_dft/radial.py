# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The Fourier transforms of the radial parts of a pseudopotential, in hartree
atomic units, with the conventions of Quantum ESPRESSO, so that totals compare
to a microhartree:
  - integrals by Simpson's rule on the radial mesh with its weights rab, an
    even number of points closed by the three-point rule on the last
    interval, the local parts up to the first point beyond 10 bohr, made odd,
    and the projectors of a species all up to the largest of their cutoffs;
  - the local potential with the long-range -Z/r taken out by -Z erf(r)/r and
    added back in reciprocal space, and its G = 0 limit of the potential plus
    Z/r;
  - the local potential without its long range, the densities and the
    projectors tabulated every 0.01 / bohr and interpolated by Lagrange
    polynomials of four points, as QE 7.6; their slopes in q are those of the
    interpolation, the exact derivatives of the values used.
Every transform is per unit volume of the cell: 4 pi / Omega for the local
parts, 4 pi / sqrt(Omega) for the projectors."""

import math

import numpy as np

DQ = 0.01  # 1/bohr, the step of the tables of the projectors
_RCUT = 10.0  # bohr, the reach of the integrals of the local parts


def simpson(f: np.ndarray, rab: np.ndarray, n: int | None = None) -> float:
    """The integral of f by Simpson's rule on a mesh of weights rab, over its
    first n points (all by default); with n even, Simpson's rule on the first
    n - 1 and the last interval by (-g[n-3] + 8 g[n-2] + 5 g[n-1]) / 12, exact
    for parabolas, g = f rab."""
    n = len(f) if n is None else n
    if n < 3:
        return 0.0
    g = f[:n] * rab[:n]
    last = n - 1 if n % 2 == 1 else n - 2
    total = (
        g[0:last:2].sum() + 4.0 * g[1:last:2].sum() + g[2 : last + 1 : 2].sum()
    ) / 3
    if n % 2 == 0:
        total += (-g[n - 3] + 8.0 * g[n - 2] + 5.0 * g[n - 1]) / 12.0
    return float(total)


def reach(r: np.ndarray) -> int:
    """The points of the mesh that the local parts are integrated over: up to
    the first beyond 10 bohr, an odd number."""
    beyond = np.nonzero(r > _RCUT)[0]
    n = int(beyond[0]) + 1 if len(beyond) else len(r)
    return 2 * ((n + 1) // 2) - 1


def bessel(l: int, x: np.ndarray) -> np.ndarray:  # noqa: E741
    """The spherical Bessel function j_l(x), l from 0 to 6: below x = max(1, l)
    by its series, whose cancellation is harmless there; beyond, for l up to 3
    by its closed form and above by the upward recurrence from j_0 and j_1,
    stable for x > l."""
    if not 0 <= l <= 6:
        raise ValueError("spherical Bessel functions up to l = 6")
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = np.abs(x) < max(1.0, float(l))
    xs, xl = x[small], x[~small]
    # x^l / (2l+1)!! sum_k (-x^2/2)^k / (k! (2l+3)(2l+5)...(2l+2k+1))
    term = xs**l / math.prod(range(1, 2 * l + 2, 2))
    total = term.copy()
    for k in range(1, 13 + 4 * l):
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
    elif l == 3:
        out[~small] = (15 / xl**4 - 6 / xl**2) * s - (15 / xl**3 - 1 / xl) * c
    else:
        before, now = s / xl, s / xl**2 - c / xl
        for n in range(1, l):
            before, now = now, (2 * n + 1) / xl * now - before
        out[~small] = now
    return out


def _table_points(q: np.ndarray) -> int:
    """The points of a table every DQ that interpolates to the largest q."""
    largest = float(np.max(q)) if len(q) else 0.0
    return int(largest / DQ) + 5


def local_table(pp, nq: int, volume: float) -> np.ndarray:
    """(nq,) the local potential without its long range, every DQ from 0:
    (4 pi / Omega) int (r V(r) + Z erf(r)) sin(qr)/q dr, at q = 0 the limit
    (4 pi / Omega) int r (r V(r) + Z erf(r)) dr."""
    n = reach(pp.r)
    r, rab, z = pp.r[:n], pp.rab[:n], pp.valence
    base = r * pp.local[:n] + z * np.vectorize(math.erf)(r)
    table = np.empty(nq)
    table[0] = simpson(base * r, rab)
    for i in range(1, nq):
        g = i * DQ
        table[i] = simpson(base * np.sin(g * r) / g, rab)
    return 4 * math.pi / volume * table


def alpha(pp, volume: float) -> float:
    """The G = 0 term of the local potential, (4 pi / Omega) int r (r V(r) + Z)
    dr."""
    n = reach(pp.r)
    r, rab = pp.r[:n], pp.rab[:n]
    return 4 * math.pi / volume * simpson(r * (r * pp.local[:n] + pp.valence), rab)


def local_potential(pp, q: np.ndarray, volume: float) -> np.ndarray:
    """V_loc(q) of the pseudopotential pp at the moduli q (1/bohr), per unit
    volume of a cell of volume Omega: the table of local_table interpolated,
    less (4 pi / Omega) Z exp(-q^2/4) / q^2, and alpha at q = 0."""
    q = np.asarray(q, dtype=float)
    table = local_table(pp, _table_points(q), volume)
    out = interpolate(table[None], q)[0]
    zero = q < 1e-8
    safe = np.where(zero, 1.0, q)
    out -= 4 * math.pi / volume * pp.valence * np.exp(-0.25 * safe * safe) / safe**2
    out[zero] = alpha(pp, volume)
    return out


def local_potential_slope(pp, q: np.ndarray, volume: float) -> np.ndarray:
    """dV_loc/dq of local_potential at the moduli q (1/bohr) at a fixed volume:
    the slope of the interpolation plus (4 pi / Omega) Z exp(-q^2/4) (1/(2q) +
    2/q^3); zero at q = 0."""
    q = np.asarray(q, dtype=float)
    table = local_table(pp, _table_points(q), volume)
    out = interpolate_slope(table[None], q)[0]
    zero = q < 1e-8
    safe = np.where(zero, 1.0, q)
    out += (4 * math.pi / volume * pp.valence * np.exp(-0.25 * safe * safe)
            * (0.5 / safe + 2.0 / safe**3))  # fmt: skip
    out[zero] = 0.0
    return out


def density_table(
    f: np.ndarray, pp, nq: int, volume: float, weight_r2: bool
) -> np.ndarray:
    """(nq,) int f(r) j0(qr) dr / Omega every DQ from 0, times r^2 4 pi when
    weight_r2."""
    n = reach(pp.r)
    r, rab = pp.r[:n], pp.rab[:n]
    g = f[:n] * (4 * math.pi * r * r if weight_r2 else 1.0)
    return (
        np.array([simpson(g * bessel(0, DQ * i * r), rab) for i in range(nq)]) / volume
    )


def radial_density(
    f: np.ndarray, pp, q: np.ndarray, volume: float, weight_r2: bool
) -> np.ndarray:
    """int f(r) j0(qr) dr / Omega, times r^2 4 pi when weight_r2, interpolated
    from density_table: the Fourier transform of a density per unit volume,
    rho_core (f = rho) or the free atom (f = 4 pi r^2 rho)."""
    q = np.asarray(q, dtype=float)
    table = density_table(f, pp, _table_points(q), volume, weight_r2)
    return interpolate(table[None], q)[0]


def radial_density_slope(
    f: np.ndarray, pp, q: np.ndarray, volume: float, weight_r2: bool
) -> np.ndarray:
    """d/dq of radial_density at a fixed volume, the slope of its
    interpolation; zero at q = 0."""
    q = np.asarray(q, dtype=float)
    table = density_table(f, pp, _table_points(q), volume, weight_r2)
    out = interpolate_slope(table[None], q)[0]
    out[q < 1e-8] = 0.0
    return out


def augmentation_tables(pp, qmax: float, volume: float) -> tuple[list, np.ndarray]:
    """The keys (p, q, L) of the augmentation charges of an ultrasoft
    pseudopotential and their transforms (tables, nq), (4 pi / Omega) int r^2
    Q^L_pq(r) j_L(qr) dr up to the largest cutoff of the projectors, every DQ
    from 0 to qmax and three steps more."""
    nq = int(qmax / DQ) + 4
    q = DQ * np.arange(nq)
    n = max((p.cutoff_index for p in pp.projectors), default=1)
    r, rab = pp.r[:n], pp.rab[:n]
    keys = sorted(pp.augmentation)
    table = np.empty((len(keys), nq))
    for i, (_, _, ell) in enumerate(keys):
        f = pp.augmentation[keys[i]][:n]
        table[i] = [simpson(f * bessel(ell, qi * r), rab) for qi in q]
    return keys, 4 * math.pi / volume * table


def projector_table(pp, qmax: float, volume: float) -> np.ndarray:
    """(P, nq) the transforms of the projectors, (4 pi / sqrt(Omega)) int r^2
    beta(r) j_l(qr) dr up to the largest cutoff of the species, every DQ from 0
    to qmax and three steps more for the interpolation."""
    nq = int(qmax / DQ) + 4
    q = DQ * np.arange(nq)
    table = np.empty((len(pp.projectors), nq))
    n = max((p.cutoff_index for p in pp.projectors), default=1)
    r, rab = pp.r[:n], pp.rab[:n]
    for i, p in enumerate(pp.projectors):
        rb = p.r_beta[:n]
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


def interpolate_slope(table: np.ndarray, q: np.ndarray) -> np.ndarray:
    """The derivatives by q of interpolate, the slopes of its polynomials."""
    x = q / DQ
    i0 = np.floor(x).astype(int)
    p = x - i0
    u, v, w = 1 - p, 2 - p, 3 - p
    t = table
    return (t[:, i0] * (-(v * w + u * w + u * v) / 6)
            + t[:, i0 + 1] * ((v * w - p * w - p * v) / 2)
            - t[:, i0 + 2] * ((u * w - p * w - p * u) / 2)
            + t[:, i0 + 3] * ((u * v - p * v - p * u) / 6)) / DQ  # fmt: skip
