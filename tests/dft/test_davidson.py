# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Block Davidson on the GPU against the eigenvalues of the matrix of H in
NumPy: at Gamma and at a general k, with restarts in a small basis, for free
electrons (degenerate levels), for the whole basis and with buffer bands; the
bands come back orthonormal with small residuals; its errors."""

import numpy as np
import pytest

from dft import gpu
from dft.hamiltonians import reference, setup

_dft = gpu.module()
if _dft.device_count() == 0:
    pytest.skip("no GPU", allow_module_level=True)

K = [0.125, -0.25, 0.375]
# a spectrum led by the kinetic energy, as of pseudopotentials: 286 plane
# waves at K, levels 0.25 to 8 hartree
REALISTIC = {"cutoff": 8.0, "potential": 1.0, "nonlocal_": 1e-3}


def guess(bands, n, seed=3):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(bands, n)) + 1j * rng.normal(size=(bands, n))


def check(h, matrix, bands, tolerance=1e-9, **options):
    e, psi, iterations, converged, residual = _dft.davidson(
        h, guess(bands, len(matrix)), tolerance=tolerance, **options
    )
    np.testing.assert_allclose(e, np.linalg.eigvalsh(matrix)[:bands], atol=1e-10)
    np.testing.assert_allclose(psi.conj() @ psi.T, np.eye(bands), atol=1e-12)
    np.testing.assert_array_less(
        np.linalg.norm(psi @ matrix.T - e[:, None] * psi, axis=1), tolerance
    )
    assert converged == bands
    assert residual < tolerance
    return iterations


@pytest.mark.parametrize("k", [[0, 0, 0], K])
def test_the_lowest_eigenvalues_of_the_matrix(k):
    h, *rest = setup(k, **REALISTIC)
    _, matrix = reference(h, *rest, k)
    assert len(matrix) > 250
    # from random bands to residuals of 1e-9
    assert check(h, matrix, 12) < 60


def test_restarts_in_a_small_basis():
    h, *rest = setup(K, **REALISTIC)
    _, matrix = reference(h, *rest, K)
    few = check(h, matrix, 10, largest_basis=60)
    many = check(h, matrix, 10, largest_basis=20, max_iterations=400)
    assert few < many


def test_degenerate_free_electrons():
    h, *rest = setup(K, cutoff=8.0, potential=0.0, nonlocal_=0.0)
    _, matrix = reference(h, *rest, K)
    np.testing.assert_allclose(matrix, np.diag(np.diag(matrix)).real, atol=1e-14)
    check(h, matrix, 16)


def test_the_whole_basis():
    h, *rest = setup([0, 0, 0])
    _, matrix = reference(h, *rest, [0, 0, 0])
    assert check(h, matrix, len(matrix)) == 1


def test_a_basis_filled_before_it_restarts():
    # 65 plane waves and a basis of all: complete after five steps
    h, *rest = setup(K)
    _, matrix = reference(h, *rest, K)
    assert check(h, matrix, 12, largest_basis=len(matrix)) <= 6


def test_buffer_bands_stop_early():
    h, *rest = setup(K, **REALISTIC)
    _, matrix = reference(h, *rest, K)
    exact = np.linalg.eigvalsh(matrix)
    e, psi, _, converged, residual = _dft.davidson(
        h, guess(16, len(matrix)), tolerance=1e-10, tight=12, loose_tolerance=1e-1
    )
    assert converged == 16
    np.testing.assert_allclose(e[:12], exact[:12], atol=1e-12)
    np.testing.assert_allclose(e[12:], exact[12:16], atol=1e-2)


def test_iterations_run_out():
    h, *rest = setup(K, **REALISTIC)
    n = len(reference(h, *rest, K)[1])
    _, _, iterations, converged, residual = _dft.davidson(
        h, guess(8, n), tolerance=1e-12, max_iterations=2
    )
    assert iterations == 2 and converged < 8 and residual > 1e-12


def test_errors():
    h, *rest = setup(K)
    n = len(reference(h, *rest, K)[1])
    with pytest.raises(ValueError, match="plane waves"):
        _dft.davidson(h, guess(4, n + 1))
    same = guess(1, n)
    with pytest.raises(ValueError, match="linearly dependent"):
        _dft.davidson(h, np.vstack([same, 2 * same]))
    with pytest.raises(ValueError, match="max_iterations"):
        _dft.davidson(h, guess(4, n), max_iterations=0)
