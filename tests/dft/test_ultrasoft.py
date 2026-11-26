# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The parts of ultrasoft pseudopotentials in the Hamiltonian against NumPy:
strengths of the projectors for each atom, the overlap S = 1 + sum |beta> q
<beta|, Davidson for H psi = e S psi against the generalized eigenvalues of
the matrices, its bands orthonormal in S, and becsum."""

import numpy as np
import pytest

from dft import gpu
from dft.hamiltonians import reference, setup

_dft = gpu.module()
if _dft.device_count() == 0:
    pytest.skip("no GPU", allow_module_level=True)

K = [0.125, -0.25, 0.375]
OPTIONS = {"cutoff": 8.0, "potential": 1.0, "nonlocal_": 1e-3}
WIDTH = 1 + 3 + 3 + 5 + 7  # the projector functions of an atom


def blocks(rng, scale, positive=False):
    out = []
    for _ in range(2):
        a = rng.normal(size=(WIDTH, WIDTH))
        out.append(a @ a.T if positive else a + a.T)
    return [scale * b for b in out]


def dense(parts):
    d = np.zeros((2 * WIDTH, 2 * WIDTH))
    for i, b in enumerate(parts):
        d[i * WIDTH : (i + 1) * WIDTH, i * WIDTH : (i + 1) * WIDTH] = b
    return d


def matrices():
    h, *rest = setup(K, **OPTIONS)
    beta, matrix = reference(h, *rest, K)
    rng = np.random.default_rng(5)
    q = blocks(rng, 0.02, positive=True)
    h.set_overlaps(np.concatenate([b.ravel() for b in q]))
    s = np.eye(len(matrix)) + beta.T @ dense(q) @ beta.conj()
    return h, beta, matrix, s, rest


def test_the_strengths_of_each_atom():
    h, *rest = setup(K, **OPTIONS)
    beta, matrix = reference(h, *rest, K)
    rng = np.random.default_rng(3)
    d = blocks(rng, 1e-3)
    h.set_strengths(np.concatenate([b.ravel() for b in d]))
    old = beta.T @ dense_of_species(rest, beta) @ beta.conj()
    new = matrix - old + beta.T @ dense(d) @ beta.conj()
    psi = rng.normal(size=(3, len(matrix))) + 1j * rng.normal(size=(3, len(matrix)))
    np.testing.assert_allclose(h.apply(psi), psi @ new.T, atol=1e-10)
    with pytest.raises(ValueError, match="blocks of the atoms"):
        h.set_strengths(np.zeros(5))


def dense_of_species(rest, beta):
    """D of the species, as setup gives it, in the functions of the atoms."""
    from dft.hamiltonians import Gaussians

    dij = rest[3]
    d = np.zeros((len(beta), len(beta)))
    first = [0, 1, 4, 7, 12]
    for atom in range(2):
        for p, lp in enumerate(Gaussians.angular):
            for t, lt in enumerate(Gaussians.angular):
                if lp == lt:
                    for m in range(2 * lp + 1):
                        d[atom * WIDTH + first[p] + m, atom * WIDTH + first[t] + m] = (
                            dij[p, t]
                        )
    return d


def test_the_overlap_and_the_generalized_problem():
    h, beta, matrix, s, _ = matrices()
    rng = np.random.default_rng(7)
    n = len(matrix)
    psi = rng.normal(size=(12, n)) + 1j * rng.normal(size=(12, n))
    np.testing.assert_allclose(h.apply_overlap(psi), psi @ s.T, atol=1e-10)
    # the generalized eigenvalues through the Cholesky factor of S
    lower = np.linalg.cholesky(s)
    inverse = np.linalg.inv(lower)
    exact = np.linalg.eigvalsh(inverse @ matrix @ inverse.conj().T)
    e, bands, _, converged, _ = _dft.davidson(h, psi, tolerance=1e-9)
    assert converged == 12
    np.testing.assert_allclose(e, exact[:12], atol=1e-10)
    gram = bands.conj() @ h.apply_overlap(bands).T
    np.testing.assert_allclose(gram, np.eye(12), atol=1e-11)
    residual = bands @ matrix.T - e[:, None] * (bands @ s.T)
    assert np.abs(residual).max() < 1e-8


def test_restarts_and_an_orthonormal_guess():
    """In a basis that restarts, and from bands that are orthonormal in S
    already (taken as they are), the same eigenpairs."""
    h, beta, matrix, s, _ = matrices()
    rng = np.random.default_rng(8)
    n = len(matrix)
    lower = np.linalg.cholesky(s)
    inverse = np.linalg.inv(lower)
    exact = np.linalg.eigvalsh(inverse @ matrix @ inverse.conj().T)
    psi = rng.normal(size=(10, n)) + 1j * rng.normal(size=(10, n))
    e, bands, iterations, converged, _ = _dft.davidson(
        h, psi, tolerance=1e-9, largest_basis=20, max_iterations=400
    )
    assert converged == 10 and iterations > 2
    np.testing.assert_allclose(e, exact[:10], atol=1e-10)
    np.testing.assert_allclose(
        bands.conj() @ h.apply_overlap(bands).T, np.eye(10), atol=1e-11
    )
    residual = bands @ matrix.T - e[:, None] * (bands @ s.T)
    assert np.abs(residual).max() < 1e-8
    again, same, iterations, converged, _ = _dft.davidson(h, bands, tolerance=1e-9)
    assert iterations == 1 and converged == 10
    np.testing.assert_allclose(again, e, atol=1e-12)
    np.testing.assert_allclose(
        np.abs(np.sum(same.conj() * (bands @ s.T), axis=1)), 1.0, atol=1e-9
    )


def test_becsum():
    h, beta, *_ = matrices()
    rng = np.random.default_rng(9)
    n = beta.shape[1]
    psi = rng.normal(size=(4, n)) + 1j * rng.normal(size=(4, n))
    w = np.array([2.0, 1.5, -0.1, 0.3])
    pairs = WIDTH * (WIDTH + 1) // 2
    ours = h.becsum(psi, w, 2 * pairs)
    p = beta.conj() @ psi.T  # (functions, bands), <beta|psi>
    expected = []
    for atom in range(2):
        block = p[atom * WIDTH : (atom + 1) * WIDTH]
        c = (block.conj() * w) @ block.T
        for i in range(WIDTH):
            for j in range(i, WIDTH):
                expected.append(c[i, i].real if i == j else 2 * c[i, j].real)
    np.testing.assert_allclose(ours, expected, atol=1e-10)
