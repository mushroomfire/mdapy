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


def ewald(cell: np.ndarray, positions: np.ndarray, charges: np.ndarray) -> float:
    """The Ewald energy in hartree of the charges at the positions (bohr) in the
    cell of rows (bohr)."""
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
    # reciprocal space
    span = [
        int(math.ceil(gmax * np.linalg.norm(cell[i]) / (2 * math.pi))) for i in range(3)
    ]
    for m in itertools.product(*(range(-s, s + 1) for s in span)):
        if m == (0, 0, 0):
            continue
        g = np.asarray(m) @ recip
        g2 = g @ g
        if g2 > gmax * gmax:
            continue
        s = (z * np.exp(1j * (x @ g))).sum()
        total += 2 * math.pi / volume * math.exp(-g2 / (4 * eta)) / g2 * abs(s) ** 2
    # real space
    span = [
        int(math.ceil(rmax * np.linalg.norm(recip[i]) / (2 * math.pi))) + 1
        for i in range(3)
    ]
    images = np.array(list(itertools.product(*(range(-s, s + 1) for s in span)))) @ cell
    for i in range(len(z)):
        d = x[i] - x[None, :, :] + images[:, None, :]  # (images, atoms, 3)
        r = np.linalg.norm(d, axis=-1)
        keep = (r > 1e-10) & (r < rmax)
        erfc = np.vectorize(math.erfc)(math.sqrt(eta) * r[keep])
        zz = np.broadcast_to(z[None, :], r.shape)[keep]
        total += 0.5 * z[i] * float((zz * erfc / r[keep]).sum())
    total -= math.sqrt(eta / math.pi) * float((z * z).sum())
    total -= math.pi * float(z.sum()) ** 2 / (2 * volume * eta)
    return total
