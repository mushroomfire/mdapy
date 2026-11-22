# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The Hamiltonian of a k-point on the GPU against its matrix built in NumPy:
the projectors from their tables, real spherical harmonics, (-i)^l and the
phases of the atoms; the kinetic energy; the local potential as the convolution
of its Fourier components; D between projectors of one l. And H is Hermitian."""

import math

import numpy as np
import pytest

from mdapy.calculators._dft import radial

_dft = pytest.importorskip("mdapy_dft_cuda12._dft")
if _dft.device_count() == 0:
    pytest.skip("no GPU", allow_module_level=True)

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


def test_the_spherical_harmonics_are_orthonormal():
    theta = (np.arange(200) + 0.5) * math.pi / 200
    phi = (np.arange(400) + 0.5) * 2 * math.pi / 400
    t, p = np.meshgrid(theta, phi, indexing="ij")
    q = np.stack([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)], -1).reshape(
        -1, 3
    )
    weight = (np.sin(t) * (math.pi / 200) * (2 * math.pi / 400)).ravel()
    lm = [(ell, m) for ell in range(4) for m in range(-ell, ell + 1)]
    values = np.array([ylm(ell, m, q) for ell, m in lm])
    np.testing.assert_allclose((values * weight) @ values.T, np.eye(len(lm)), atol=1e-4)


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
            },
        )
        for i, ell in enumerate(ANGULAR)
    )


def setup(k):
    half = 5.431 / 2 / BOHR
    a = np.array([[0, half, half], [half, 0, half], [half, half, 0]])
    cutoff = 3.0
    grid = _dft.fft_grid(a, 4 * cutoff)
    volume = abs(np.linalg.det(a))
    table = radial.projector_table(Gaussians, math.sqrt(2 * cutoff) + 1, volume)
    rng = np.random.default_rng(0)
    dij = rng.normal(size=(5, 5))
    dij = dij + dij.T
    positions = np.array([[0.0, 0.0, 0.0], [0.27, 0.25, 0.25]])
    h = _dft.Hamiltonian(a, grid, cutoff, [table], [Gaussians.angular], [dij], radial.DQ,
                         positions, np.array([0, 0], dtype=np.int32))  # fmt: skip
    h.set_k(k)
    v = rng.normal(size=tuple(grid))
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


@pytest.mark.parametrize("k", [[0, 0, 0], [0.125, -0.25, 0.375]])
def test_against_the_matrix_of_numpy(k):
    h, *rest = setup(k)
    beta, matrix = reference(h, *rest, k)
    assert h.projectors == 2 * (1 + 3 + 3 + 5 + 7)
    np.testing.assert_allclose(h.beta, beta, atol=1e-12)
    rng = np.random.default_rng(1)
    psi = rng.normal(size=(4, len(matrix))) + 1j * rng.normal(size=(4, len(matrix)))
    np.testing.assert_allclose(h.apply(psi), psi @ matrix.T, rtol=1e-10, atol=1e-10)


def test_h_is_hermitian():
    h, *_ = setup([0.1, 0.2, 0.3])
    rng = np.random.default_rng(2)
    n = len(h.miller)
    psi, phi = (
        rng.normal(size=(1, n)) + 1j * rng.normal(size=(1, n)) for _ in range(2)
    )
    left = (phi.conj() * h.apply(psi)).sum()
    right = (psi.conj() * h.apply(phi)).sum().conjugate()
    assert left == pytest.approx(right, rel=1e-12)
