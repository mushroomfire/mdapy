# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The electrostatic energy of the ions, point charges in a uniform background
that makes the cell neutral, by the Ewald sum, in hartree atomic units:

  E = 1/2 sum_ij sum'_R z_i z_j erfc(sqrt(eta) |t_ij + R|) / |t_ij + R|
      + (2 pi / Omega) sum_{G != 0} exp(-G^2 / 4 eta) / G^2 |S(G)|^2
      - sqrt(eta / pi) sum_i z_i^2 - pi (sum_i z_i)^2 / (2 Omega eta),

S(G) = sum_i z_i exp(i G.t_i). It does not depend on eta once both sums are
converged, which the cutoffs here make so to 1e-12 of the terms."""

import itertools
import math

import numpy as np


def ewald(cell: np.ndarray, positions: np.ndarray, charges: np.ndarray, forces=False):
    """The Ewald energy in hartree of the charges at the positions (bohr) in the
    cell of rows (bohr), and with forces the forces on them (N, 3) in hartree
    / bohr."""
    cell = np.asarray(cell, dtype=float)
    x = np.asarray(positions, dtype=float)
    z = np.asarray(charges, dtype=float)
    volume = abs(np.linalg.det(cell))
    recip = 2 * math.pi * np.linalg.inv(cell).T
    # the split: the real-space sum to erfc(sqrt(eta) r) < 1e-16 and the
    # reciprocal one to exp(-G^2 / 4 eta) < 1e-16, eta from the cell's size
    eta = (math.sqrt(math.pi) / volume ** (1 / 3)) ** 2
    rmax = 8.5 / math.sqrt(eta)
    gmax = 2 * math.sqrt(eta) * 6.1
    total = 0.0
    force = np.zeros_like(x)
    # reciprocal space: F_i = (4 pi / Omega) z_i sum_G w(G) G Im[exp(iG.x_i) conj(S)]
    span = [
        int(math.ceil(gmax * np.linalg.norm(cell[i]) / (2 * math.pi))) for i in range(3)
    ]
    m = np.array(list(itertools.product(*(range(-s, s + 1) for s in span))))
    g = m @ recip
    g2 = (g * g).sum(axis=1)
    keep = (g2 > 1e-12) & (g2 <= gmax * gmax)
    g, g2 = g[keep], g2[keep]
    w = np.exp(-g2 / (4 * eta)) / g2
    phase = np.exp(1j * (x @ g.T))  # (atoms, G)
    s = (z[:, None] * phase).sum(axis=0)
    total += 2 * math.pi / volume * float((w * np.abs(s) ** 2).sum())
    force += 4 * math.pi / volume * z[:, None] * ((w * (phase * s.conj()).imag) @ g)
    # real space: F_i = z_i sum_j z_j [erfc/r^2 + 2 sqrt(eta/pi) exp(-eta r^2)/r] d/r
    span = [
        int(math.ceil(rmax * np.linalg.norm(recip[i]) / (2 * math.pi))) + 1
        for i in range(3)
    ]
    images = np.array(list(itertools.product(*(range(-s, s + 1) for s in span)))) @ cell
    root = math.sqrt(eta)
    for i in range(len(z)):
        d = x[i] - x[None, :, :] + images[:, None, :]  # (images, atoms, 3)
        r = np.linalg.norm(d, axis=-1)
        keep = (r > 1e-10) & (r < rmax)
        rk = r[keep]
        erfc = np.vectorize(math.erfc)(root * rk)
        zz = np.broadcast_to(z[None, :], r.shape)[keep]
        total += 0.5 * z[i] * float((zz * erfc / rk).sum())
        radial = zz * (
            erfc / rk**2 + 2 * root / math.sqrt(math.pi) * np.exp(-eta * rk * rk) / rk
        )
        force[i] += z[i] * ((radial / rk)[:, None] * d[keep]).sum(axis=0)
    total -= math.sqrt(eta / math.pi) * float((z * z).sum())
    total -= math.pi * float(z.sum()) ** 2 / (2 * volume * eta)
    return (total, force) if forces else total
