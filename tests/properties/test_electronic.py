# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The sums of the electronic density of states over the energies on a mesh,
without the DFT: the linear tetrahedron method against the dense sum over
every tetrahedron and energy of the phonons' code, the Gaussians against a
loop over every state, each band integrating to 1. Against dos.x of Quantum
ESPRESSO in tests/dft/test_dos.py, through the calculator in
tests/dft/test_electronic.py."""

import numpy as np
import pytest

from mdapy.properties import _electronic
from mdapy.properties._phonons import _mesh, _tetrahedra, _tetrahedron_dos

COUNTS = np.array([5, 4, 6])
RECIPROCAL = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])


def bands():
    """(2, K, 5) smooth periodic bands on the mesh, ascending, the second
    spin shifted."""
    k = _mesh(COUNTS)
    e = [
        b + np.cos(2 * np.pi * k[:, 0])
        + 0.5 * (b + 1) * np.cos(2 * np.pi * (k[:, 1] + k[:, 2]))
        for b in range(5)
    ]  # fmt: skip
    up = np.sort(np.stack(e, axis=-1), axis=-1)
    return np.stack([up, up + 0.3 * np.sin(2 * np.pi * k[:, 2:]) ** 2])


@pytest.mark.parametrize("pairs", [1 << 21, 1000])
def test_tetrahedra_against_the_dense_sum(monkeypatch, pairs):
    # a few pairs at a time, as many pieces as that takes
    monkeypatch.setattr(_electronic, "_PAIRS", pairs)
    e = bands()
    grid = np.linspace(-3.0, 9.0, 2001)
    ours = _electronic._tetrahedron_sum(e, COUNTS, RECIPROCAL, grid)
    corners = _tetrahedra(COUNTS, RECIPROCAL)
    dense = np.zeros_like(ours)
    for s in range(2):
        for b in range(5):
            dense[s] += _tetrahedron_dos(e[s][corners, b], grid).sum(axis=0)
    dense /= len(corners)
    np.testing.assert_allclose(ours, dense, rtol=0, atol=1e-13)
    # each band integrates to 1, to the error of the trapezoids
    np.testing.assert_allclose(np.trapezoid(ours, grid, axis=1), 5.0, atol=2e-3)


def test_flat_bands_and_energies_outside():
    e = np.zeros((1, COUNTS.prod(), 2))
    e[..., 1] = 1.0
    grid = np.linspace(-1.0, 2.0, 31)
    # a band of one energy has no width: no density, and no division by zero
    flat = _electronic._tetrahedron_sum(e, COUNTS, RECIPROCAL, grid)
    assert np.isfinite(flat).all() and not flat.any()
    # nothing at energies away from every band
    away = _electronic._tetrahedron_sum(
        bands(), COUNTS, RECIPROCAL, np.linspace(20.0, 30.0, 11)
    )
    assert away.shape == (2, 11) and not away.any()


def test_gaussians_against_every_state():
    e = bands()
    k = e.shape[1]
    rng = np.random.default_rng(3)
    weights = rng.random(k)
    weights /= weights.sum()
    grid = np.linspace(-3.0, 9.0, 1201)
    sigma = 0.15
    ours = _electronic._gaussian_sum(e, weights, grid, sigma)
    brute = np.zeros_like(ours)
    for s in range(2):
        for i in range(k):
            for b in range(5):
                x = (grid - e[s, i, b]) / sigma
                brute[s] += weights[i] * np.exp(-0.5 * x * x)
    brute /= sigma * np.sqrt(2 * np.pi)
    np.testing.assert_allclose(ours, brute, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(np.trapezoid(ours, grid, axis=1), 5.0, atol=1e-6)
