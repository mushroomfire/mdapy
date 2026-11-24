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

from dft import gpu
from dft.hamiltonians import reference, setup, ylm

_dft = gpu.module()
if _dft.device_count() == 0:
    pytest.skip("no GPU", allow_module_level=True)


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


@pytest.mark.parametrize("k", [[0, 0, 0], [0.125, -0.25, 0.375]])
def test_against_the_matrix_of_numpy(k):
    h, *rest = setup(k)
    beta, matrix = reference(h, *rest, k)
    assert h.projectors == 2 * (1 + 3 + 3 + 5 + 7)
    np.testing.assert_allclose(h.beta, beta, atol=1e-12)
    rng = np.random.default_rng(1)
    psi = rng.normal(size=(4, len(matrix))) + 1j * rng.normal(size=(4, len(matrix)))
    np.testing.assert_allclose(h.apply(psi), psi @ matrix.T, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(h.diagonal, np.diag(matrix).real, rtol=1e-12, atol=1e-12)


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
