# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The one-center parts of PAW for the GPU, as Quantum ESPRESSO 7.6 takes
them: the densities of each atom on its radial mesh as sums of Y_LM(r^) n_LM(r)
/ r^2 from its occupations becsum,

  n_LM(r) = sum_ab becsum_ab G(LM, a, b) phi_a(r) phi_b(r)            (AE)
  n~_LM(r) = sum_ab becsum_ab G(LM, a, b) (phi~_a phi~_b + r^2 Q^L_ab)(r)  (PS)

G the Gaunt coefficients of the real harmonics; their Hartree and XC
energies, XC on directions of Gauss-Legendre in cos(theta) times a uniform
grid in phi of order 3 lmax_rho (and 2 more with gradients), and the
strengths D_ab = dE/d becsum_ab of AE less PS."""

import math

import numpy as np

from mdapy.calculators._dft import angular, radial


def _harmonics(lmax: int, theta, phi) -> np.ndarray:
    """((lmax + 1)^2, n) the real Y_lm of angular.real_harmonics at the angles
    (theta, phi), by operations that take complex angles, for the derivatives
    by the complex step."""
    z, rho = np.cos(theta), np.sin(theta)
    out = np.zeros(((lmax + 1) ** 2, len(z)), dtype=np.result_type(z, phi))
    p = {(0, 0): np.ones_like(z)}
    for m in range(0, lmax + 1):
        if m > 0:
            p[(m, m)] = p[(m - 1, m - 1)] * (2 * m - 1) * rho
        if m + 1 <= lmax:
            p[(m + 1, m)] = z * (2 * m + 1) * p[(m, m)]
        for ell in range(m + 2, lmax + 1):
            p[(ell, m)] = (
                (2 * ell - 1) * z * p[(ell - 1, m)] - (ell + m - 1) * p[(ell - 2, m)]
            ) / (ell - m)
    for ell in range(lmax + 1):
        for m in range(-ell, ell + 1):
            a = abs(m)
            norm = math.sqrt((2 * ell + 1) / (4 * math.pi)
                             * math.factorial(ell - a) / math.factorial(ell + a))  # fmt: skip
            if m == 0:
                value = norm * p[(ell, 0)]
            elif m > 0:
                value = math.sqrt(2) * norm * p[(ell, a)] * np.cos(a * phi)
            else:
                value = math.sqrt(2) * norm * p[(ell, a)] * np.sin(a * phi)
            out[angular.index(ell, m)] = value
    return out


def quadrature(lmax_rho: int, gga: bool) -> dict:
    """The directions of the one-center integrals as QE's PAW_rad_init: order
    3 lmax_rho, 2 more for the gradients of GGA (none for a spherical
    density), Gauss-Legendre in cos(theta) and a uniform grid in phi. Gives
    the weights ww (nx,), the harmonics of l up to lmax_rho + extra, extra 2
    with GGA (else 0), ylm (nx, lh), dY/dtheta (dylmt) and dY/dphi /
    sin(theta) (dylmp), the angles and the sin and cos of theta, l2 =
    (lmax_rho + 1)^2 and lh =
    (lmax_rho + 1 + extra)^2."""
    order, extra = (0, 0) if lmax_rho == 0 else (3 * lmax_rho, 2 if gga else 0)
    total = order + extra
    nphi = total + 1 + total % 2
    x, w = np.polynomial.legendre.leggauss((total + 2) // 2)
    theta = np.repeat(np.arccos(x), nphi)
    phi = np.tile(2 * math.pi * np.arange(nphi) / nphi, len(x))
    ww = np.repeat(w, nphi) * 2 * math.pi / nphi
    top = lmax_rho + extra
    h = 1e-30
    ylm = _harmonics(top, theta, phi).real
    dt = _harmonics(top, theta + 1j * h, phi).imag / h
    dp = _harmonics(top, theta, phi + 1j * h).imag / h / np.sin(theta)
    return {
        "ww": ww, "ylm": np.ascontiguousarray(ylm.T),
        "dylmt": np.ascontiguousarray(dt.T), "dylmp": np.ascontiguousarray(dp.T),
        "theta": theta, "phi": phi, "sin": np.sin(theta), "cos": np.cos(theta),
        "l2": (lmax_rho + 1) ** 2, "lh": (top + 1) ** 2,
    }  # fmt: skip


def radial_pairs(pp) -> list[tuple[int, int]]:
    """The pairs (p, q), p <= q, of the projectors of pp, in QE's order of
    the augmentation charges: q major."""
    n = len(pp.projectors)
    return [(p, q) for q in range(n) for p in range(q + 1)]


def onecenter(pp, gga: bool) -> dict:
    """The one-center data of a PAW species for the GPU:
    - the radial mesh r, rab, dx, its points mesh, kkbeta and the weights of
      Simpson's rule over the mesh (whole) and up to kkbeta (inner);
    - the quadrature of quadrature();
    - ae (Q, M) the products of AE partial waves of the radial pairs, ps (Q,
      lmax_rho + 1, M) those of the PS ones plus r^2 Q^L of each L;
    - coef (P, l2) the Gaunt coefficients G(LM, a, b) of the pairs a <= b of
      projector functions, pair (P,) their radial pair;
    - the AE and PS core charges."""
    data = pp.paw
    if data is None:
        raise ValueError(f"{pp.element}: not a PAW dataset")
    m = len(pp.r)
    lmax_rho = data.lmax_rho
    radial_ = radial_pairs(pp)
    where = {pair: i for i, pair in enumerate(radial_)}
    ae = np.array([data.ae_waves[p] * data.ae_waves[q] for p, q in radial_])
    ps = np.zeros((len(radial_), lmax_rho + 1, m))
    for i, (p, q) in enumerate(radial_):
        ps[i] = data.ps_waves[p] * data.ps_waves[q]
        for ell in range(lmax_rho + 1):
            charge = pp.augmentation.get((p, q, ell))
            if charge is not None:
                ps[i, ell] += charge
    functions = [(p, beta.angular, mm) for p, beta in enumerate(pp.projectors)
                 for mm in range(-beta.angular, beta.angular + 1)]  # fmt: skip
    l2 = (lmax_rho + 1) ** 2
    gaunt = angular.gaunt(pp.lmax)
    coef, pair = [], []
    for a, (pa, la, ma) in enumerate(functions):
        for pb, lb, mb in functions[a:]:
            coef.append(gaunt[angular.index(la, ma), angular.index(lb, mb), :l2])
            pair.append(where[(min(pa, pb), max(pa, pb))])
    kkbeta = pp.kkbeta
    out = quadrature(lmax_rho, gga)
    out.update(
        mesh=m, kkbeta=kkbeta, r=pp.r, rab=pp.rab, dx=data.dx,
        whole=radial.simpson_weights(pp.rab), inner=radial.simpson_weights(pp.rab, kkbeta),
        ae=np.ascontiguousarray(ae), ps=np.ascontiguousarray(ps),
        coef=np.ascontiguousarray(coef), pair=[int(p) for p in pair],
        ae_core=data.ae_core, ps_core=np.zeros(m) if pp.core is None else pp.core,
        lmax_rho=lmax_rho, functions=len(functions),
    )  # fmt: skip
    return out


def initial_becsum(pp, spins: int, moment: float = 0.0) -> np.ndarray:
    """(spins, P) the occupations of the free atom as QE's PAW_atomic_becsum:
    oc / (2l + 1) on the diagonal, shared between the spins by moment (in Bohr
    magnetons) over the valence."""
    occ = []
    for p, beta in enumerate(pp.projectors):
        occ += [pp.paw.occupations[p] / (2 * beta.angular + 1)] * (2 * beta.angular + 1)
    n = len(occ)
    diagonal = np.zeros(n * (n + 1) // 2)
    k = 0
    for a in range(n):
        diagonal[k] = occ[a]
        k += n - a
    if spins == 1:
        return diagonal[None, :]
    share = moment / pp.valence
    return np.stack([0.5 * (1 + share) * diagonal, 0.5 * (1 - share) * diagonal])
