# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""A Hamiltonian of a k-point on the GPU and its matrix built in NumPy, for the
tests of the Hamiltonian and of the eigensolvers: silicon's cell, projectors of
Gaussians of l from 0 to 3, a random D and a random local potential."""

import math

import numpy as np

from mdapy.calculators._dft import radial

BOHR = 0.529177210903


def ylm(l, m, q):  # noqa: E741
    """The real spherical harmonics of the directions q (n, 3), as the GPU's."""
    r = np.linalg.norm(q, axis=1)
    safe = np.where(r > 1e-12, r, 1.0)
    x, y, z = (q / safe[:, None]).T
    if l == 0:
        return np.full(len(q), 0.5 / math.sqrt(math.pi))
    table = {
        (1, -1): math.sqrt(3 / (4 * math.pi)) * y,
        (1, 0): math.sqrt(3 / (4 * math.pi)) * z,
        (1, 1): math.sqrt(3 / (4 * math.pi)) * x,
        (2, -2): math.sqrt(15 / (4 * math.pi)) * x * y,
        (2, -1): math.sqrt(15 / (4 * math.pi)) * y * z,
        (2, 0): math.sqrt(5 / (16 * math.pi)) * (3 * z * z - 1),
        (2, 1): math.sqrt(15 / (4 * math.pi)) * x * z,
        (2, 2): math.sqrt(15 / (16 * math.pi)) * (x * x - y * y),
        (3, -3): math.sqrt(35 / (32 * math.pi)) * y * (3 * x * x - y * y),
        (3, -2): math.sqrt(105 / (4 * math.pi)) * x * y * z,
        (3, -1): math.sqrt(21 / (32 * math.pi)) * y * (5 * z * z - 1),
        (3, 0): math.sqrt(7 / (16 * math.pi)) * z * (5 * z * z - 3),
        (3, 1): math.sqrt(21 / (32 * math.pi)) * x * (5 * z * z - 1),
        (3, 2): math.sqrt(105 / (16 * math.pi)) * z * (x * x - y * y),
        (3, 3): math.sqrt(35 / (32 * math.pi)) * x * (x * x - 3 * y * y),
    }
    return np.where(r > 1e-12, table[(l, m)], 0.0)


MESH = 0.01 * np.arange(1601)
ANGULAR = [0, 1, 1, 2, 3]


class Gaussians:
    """Projectors r beta = r^(l+1) exp(-r^2 / (2 + i)) of l from 0 to 3, two of
    l = 1."""

    r = MESH
    rab = np.full(1601, 0.01)
    angular = ANGULAR
    projectors = tuple(
        type(
            "B",
            (),
            {
                "angular": ell,
                "r_beta": MESH ** (ell + 1) * np.exp(-(MESH**2) / (2 + i)),
                "cutoff_index": len(MESH),
            },
        )
        for i, ell in enumerate(ANGULAR)
    )


def setup(k, cutoff=3.0, potential=1.0, nonlocal_=1.0):
    """The Hamiltonian of k and its parts, of a cutoff in hartree; potential
    and nonlocal_ scale the local potential and D (0 for free electrons)."""
    from mdapy_dft_cuda12 import _dft

    half = 5.431 / 2 / BOHR
    a = np.array([[0, half, half], [half, 0, half], [half, half, 0]])
    grid = _dft.fft_grid(a, 4 * cutoff)
    volume = abs(np.linalg.det(a))
    table = radial.projector_table(Gaussians, math.sqrt(2 * cutoff) + 1, volume)
    rng = np.random.default_rng(0)
    dij = rng.normal(size=(5, 5))
    dij = nonlocal_ * (dij + dij.T)
    positions = np.array([[0.0, 0.0, 0.0], [0.27, 0.25, 0.25]])
    h = _dft.Hamiltonian(a, grid, cutoff, [table], [Gaussians.angular], [dij], radial.DQ,
                         positions, np.array([0, 0], dtype=np.int32))  # fmt: skip
    h.set_k(k)
    v = potential * rng.normal(size=tuple(grid))
    h.set_potential(v)
    return h, a, grid, table, dij, positions, v


def reference(h, a, grid, table, dij, positions, v, k):
    """The matrix of H in NumPy."""
    m = h.miller
    b = 2 * math.pi * np.linalg.inv(a).T
    q = (np.asarray(k) + m) @ b
    radials = radial.interpolate(table, np.linalg.norm(q, axis=1))
    beta, d_blocks = [], []
    for tau in positions:
        phase = np.exp(-2j * math.pi * ((np.asarray(k) + m) @ tau))
        first = []
        for p, ell in enumerate(Gaussians.angular):
            first.append(len(beta))
            for mm in range(-ell, ell + 1):
                beta.append((-1j) ** ell * ylm(ell, mm, q) * radials[p] * phase)
        d_blocks.append(first)
    beta = np.array(beta)
    d = np.zeros((len(beta), len(beta)))
    for first in d_blocks:
        for p, lp in enumerate(Gaussians.angular):
            for s, ls in enumerate(Gaussians.angular):
                if lp == ls:
                    for mm in range(2 * lp + 1):
                        d[first[p] + mm, first[s] + mm] = dij[p, s]
    vg = np.fft.fftn(v) / v.size
    diff = np.mod(m[:, None, :] - m[None, :, :], grid)
    local = vg[diff[..., 0], diff[..., 1], diff[..., 2]]
    kinetic = 0.5 * (q * q).sum(axis=1)
    return beta, np.diag(kinetic) + local + beta.T @ d @ beta.conj()
