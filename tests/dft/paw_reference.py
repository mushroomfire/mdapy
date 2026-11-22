# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The one-center terms of PAW in NumPy, the reference of the GPU's, from the
equations of Quantum ESPRESSO 7.6 (PAW_potential and what it calls), on the
data of mdapy.calculators._dft.paw.onecenter:
  - the Hartree potential of each LM by Numerov's method on the logarithmic
    mesh, y = sqrt(r) V, y'' - (l + 1/2)^2 y = -(2l + 1) f sqrt(r), y ~ r^(l+1/2)
    at the first point and ~ r^-(l+1/2) at the last, a cubic series at the
    origin for l < 2;
  - radial derivatives by the parabola through three neighbours, zero at the
    last point and extrapolated to the first;
  - XC on the directions of the quadrature, the gradients of GGA from the
    radial derivative and the angular ones of the harmonics, its potential
    by the divergence of the flux expanded to two more l;
  - D_ab = sum_LM G(LM, a, b) int v_LM F_ab,L dr to kkbeta, AE less PS."""

import math

import numpy as np

GGA = ("pbe", "pbe_pw")


def radial_gradient(f: np.ndarray, r: np.ndarray) -> np.ndarray:
    """df/dr along the last axis of f."""
    g = np.zeros_like(f)
    a, b = r[2:] - r[1:-1], r[:-2] - r[1:-1]
    g[..., 1:-1] = (
        a * a * (f[..., :-2] - f[..., 1:-1]) - b * b * (f[..., 2:] - f[..., 1:-1])
    ) / (a * b * (r[2:] - r[:-2]))
    g[..., 0] = g[..., 1] + (g[..., 2] - g[..., 1]) * (r[0] - r[1]) / (r[2] - r[1])
    return g


def _cubic(d: np.ndarray, r: np.ndarray) -> np.ndarray:
    """b0..b3 of the cubic through the four points (r, d), from Newton's
    divided differences."""
    d1 = [(d[i + 1] - d[i]) / (r[i + 1] - r[i]) for i in range(3)]
    d2 = [(d1[i + 1] - d1[i]) / (r[i + 2] - r[i]) for i in range(2)]
    b3 = (d2[1] - d2[0]) / (r[3] - r[0])
    b2 = d2[0] - b3 * (r[0] + r[1] + r[2])
    b1 = d1[0] - b2 * (r[1] + r[0]) - b3 * (r[0] ** 2 + r[1] ** 2 + r[0] * r[1])
    b0 = d[0] - r[0] * (b1 + r[0] * (b2 + r[0] * b3))
    return np.array([b0, b1, b2, b3])


def hartree(ell: int, f: np.ndarray, r: np.ndarray, dx: float) -> np.ndarray:
    """The potential of l of f = 4 pi r^2 n_l(r) / (2l + 1) on the logarithmic
    mesh r of step dx."""
    m = len(r)
    k21, nst = 2 * ell + 1, 2 * ell + 2
    sqr = np.sqrt(r)
    c2 = c3 = 0.0
    if nst - ell - 1 < 3:
        b = _cubic(-k21 * f[:4] / r[:4] ** nst, r[:4])
        e = np.zeros(6)
        e[nst - ell - 2 : nst - ell + 2] = b
        c2, c3 = e[0] / (4 * ell + 6), e[1] / (6 * ell + 12)
    ch = dx * dx / 12
    xkh2 = ch * (ell + 0.5) ** 2
    ei, di = 1.0 - xkh2, -(2.0 + 10.0 * xkh2)
    s = k21 * ch * sqr * f
    rhs = s[:-2] + 10.0 * s[1:-1] + s[2:]
    diag = np.full(m - 2, -di)
    f1 = (sqr[0] / sqr[1]) ** k21
    fn = (sqr[m - 2] / sqr[m - 1]) ** k21
    diag[0] -= ei * f1
    diag[-1] -= ei * fn
    rhs[0] -= (
        ei
        * sqr[0] ** k21
        * (c2 * (r[1] ** 2 - r[0] ** 2) + c3 * (r[1] ** 3 - r[0] ** 3))
    )
    # the symmetric tridiagonal system, off-diagonal -ei, by elimination
    n = m - 2
    c, x = np.empty(n), np.empty(n)
    off = -ei
    w = diag[0]
    c[0], x[0] = off / w, rhs[0] / w
    for i in range(1, n):
        w = diag[i] - off * c[i - 1]
        c[i] = off / w
        x[i] = (rhs[i] - off * x[i - 1]) / w
    for i in range(n - 2, -1, -1):
        x[i] -= c[i] * x[i + 1]
    y = np.empty(m)
    y[1:-1] = x
    c0 = y[1] / sqr[1] ** k21 - c2 * r[1] ** 2 - c3 * r[1] ** 3
    y[0] = sqr[0] ** k21 * (c0 + c2 * r[0] ** 2 + c3 * r[0] ** 3)
    y[-1] = y[-2] * fn
    return y / sqr


def _ell(lm: int) -> int:
    return int(math.isqrt(lm))


def densities(data: dict, becsum: np.ndarray, kind: str) -> np.ndarray:
    """(S, l2, M) the n_LM(r) of AE or PS of becsum (S, P)."""
    l2, pairs = data["l2"], data["pair"]
    count = data["ae"].shape[0]
    c = np.zeros((becsum.shape[0], l2, count))
    for p in range(len(pairs)):
        c[:, :, pairs[p]] += becsum[:, p, None] * data["coef"][p][None, :]
    if kind == "ae":
        return np.einsum("slq,qr->slr", c, data["ae"])
    ells = [_ell(lm) for lm in range(l2)]
    return np.einsum("slq,qlr->slr", c, data["ps"][:, ells, :])


def _xc(data, rho_lm, core, functional, xc):
    """E_xc and v_LM (S, l2, M) of the densities rho_lm (S, l2, M)."""
    spins = rho_lm.shape[0]
    r, l2, lh = data["r"], data["l2"], data["lh"]
    ylm, ww = data["ylm"], data["ww"]
    y = ylm[:, :l2]
    n = np.einsum("xl,slr->sxr", y, rho_lm) / r**2 + core / spins
    gga = functional in GGA
    shape = n.shape[1:]
    if spins == 1:
        rho = np.stack([0.5 * n[0], 0.5 * n[0]], -1).reshape(-1, 2)
    else:
        rho = np.stack([n[0], n[1]], -1).reshape(-1, 2)
    sigma = np.zeros((rho.shape[0], 3))
    grads = None
    if gga:
        gr = radial_gradient(n, r)
        gp = np.einsum("xl,slr->sxr", data["dylmp"][:, :l2], rho_lm) / r**3
        gt = np.einsum("xl,slr->sxr", data["dylmt"][:, :l2], rho_lm) / r**3
        grads = np.stack([gr, gp, gt], 1)  # (S, 3, nx, M)
        if spins == 1:
            g = 0.5 * grads[0]
            s2 = (g * g).sum(0).ravel()
            sigma = np.stack([s2, s2, s2], -1)
        else:
            sigma = np.stack(
                [(grads[0] * grads[0]).sum(0).ravel(), (grads[0] * grads[1]).sum(0).ravel(),
                 (grads[1] * grads[1]).sum(0).ravel()], -1)  # fmt: skip
    f, v, vs = xc(rho, sigma, spins == 2, spins == 2)
    f, v, vs = f.reshape(shape), v.reshape(*shape, 2), vs.reshape(*shape, 3)
    energy = float((ww[:, None] * f * r**2 * data["whole"][None, :]).sum())
    wy = ww[:, None] * y
    v_lm = np.stack([wy.T @ v[..., s] for s in range(spins)])
    if gga:
        yh = ylm[:, :lh]
        wyh = ww[:, None] * yh
        angular = (
            data["dylmt"][:, :lh] * data["sin"][:, None] + 2 * yh * data["cos"][:, None]
        )
        for s in range(spins):
            if spins == 1:
                h = (2 * vs[..., 0] + vs[..., 1])[None] * (0.5 * grads[0])
            else:
                o = 1 - s
                same = vs[..., 0] if s == 0 else vs[..., 2]
                h = 2 * same[None] * grads[s] + vs[..., 1][None] * grads[o]
            h = h * r**2
            h[2] /= data["sin"][:, None]
            big = np.stack([wyh.T @ h[c] for c in range(3)])  # (3, lh, M)
            div = data["dylmp"][:, :lh] @ big[1] + angular @ big[2]
            div_lm = wy.T @ div / r**3 + radial_gradient(big[0][:l2], r) / r**2
            v_lm[s] -= div_lm
    return energy, v_lm


def onecenter(
    data: dict, becsum: np.ndarray, functional: str, xc
) -> tuple[dict, np.ndarray]:
    """The energies (hartree and xc of ae and ps) and D (S, P), AE less PS, of
    one atom of occupations becsum (S, P); xc(rho, sigma, polarized,
    signed_gradient) gives f, v and vs of the points."""
    r, dx = data["r"], data["dx"]
    l2 = data["l2"]
    ells = [_ell(lm) for lm in range(l2)]
    energies = {}
    d = np.zeros(becsum.shape)
    for kind, sign in (("ae", 1.0), ("ps", -1.0)):
        rho_lm = densities(data, becsum, kind)
        total = rho_lm.sum(0)
        vh = np.stack(
            [
                hartree(ells[lm], 4 * math.pi / (2 * ells[lm] + 1) * total[lm], r, dx)
                for lm in range(l2)
            ]
        )
        energies["hartree_" + kind] = 0.5 * float((vh * total * data["whole"]).sum())
        core = data["ae_core"] if kind == "ae" else data["ps_core"]
        energies["xc_" + kind], v_lm = _xc(data, rho_lm, core, functional, xc)
        v = v_lm + vh[None]
        functions = data["ae"][:, None, :] if kind == "ae" else data["ps"][:, ells, :]
        # (S, l2, Q): int v_LM F_q,L(LM) dr to kkbeta
        integral = np.einsum("slr,qlr,r->slq", v, np.broadcast_to(functions, (functions.shape[0], l2, len(r))),
                             data["inner"])  # fmt: skip
        d += sign * np.einsum("pl,slp->sp", data["coef"], integral[:, :, data["pair"]])
    energies["total"] = (
        energies["hartree_ae"]
        + energies["xc_ae"]
        - energies["hartree_ps"]
        - energies["xc_ps"]
    )
    return energies, d


def rounded_gamma(data: dict, becsum: np.ndarray) -> float:
    """The first-order change of E^1 of an atom of occupations becsum (2, P)
    with spin from the rounded gamma of QE's spin-polarized PBE, AE less PS:
    the counterpart of espresso.rounded_gamma for the one-center densities."""
    from dft.espresso import gamma_difference

    r, l2 = data["r"], data["l2"]
    y = data["ylm"][:, :l2]
    total = 0.0
    for kind, sign in (("ae", 1.0), ("ps", -1.0)):
        rho_lm = densities(data, becsum, kind)
        core = data["ae_core"] if kind == "ae" else data["ps_core"]
        both = rho_lm.sum(0)
        n = y @ both / r**2 + core
        m = y @ (rho_lm[0] - rho_lm[1]) / r**2
        gr = radial_gradient(n, r)
        gp = data["dylmp"][:, :l2] @ both / r**3
        gt = data["dylmt"][:, :l2] @ both / r**3
        grad2 = gr * gr + gp * gp + gt * gt
        keep = (n > 1e-6) & (grad2 > 1e-12)
        weights = data["ww"][:, None] * data["whole"][None, :] * r**2
        values = np.zeros(n.shape)
        values[keep] = gamma_difference(n[keep], m[keep], grad2[keep])
        total += sign * float((weights * values).sum())
    return total
