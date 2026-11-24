# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The magnetic moments of the atoms: Gaussians of known total on the grid of
a cubic and of a triclinic cell, across the periodic boundary; a uniform
magnetization; the default radius; the errors. Runs without a GPU."""

import itertools

import numpy as np
import pytest

from mdapy.calculators._dft.moments import default_radius, magnetic_moments
from mdapy.errors import InvalidParameter

CUBIC = np.diag([6.0, 6.0, 6.0])
TRICLINIC = np.array([[5.0, 0.0, 0.0], [1.5, 5.5, 0.0], [0.8, 1.2, 6.0]])
SIGMA = 0.25


def points(cell, shape):
    """The Cartesian positions of the grid points, (n1, n2, n3, 3)."""
    axes = [np.arange(n) / n for n in shape]
    frac = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    return frac @ cell


def gaussians(cell, shape, centres, totals):
    """Gaussians of the given totals (per Å^3 on the grid) at the centres, each
    at the distance of its nearest periodic image."""
    x = points(cell, shape)
    rho = np.zeros(shape)
    for centre, total in zip(centres, totals):
        best = np.full(shape, np.inf)
        for t in itertools.product(range(-2, 3), repeat=3):
            v = x - centre - np.array(t) @ cell
            best = np.minimum(best, np.einsum("...i,...i", v, v))
        rho += total * np.exp(-best / (2 * SIGMA**2)) / (2 * np.pi * SIGMA**2) ** 1.5
    return rho


def test_gaussians_in_a_cubic_cell():
    shape = (48, 48, 48)
    # the first sits across the boundary of the cell
    centres = np.array([[0.1, 5.9, 0.05], [3.0, 2.6, 3.3]])
    rho = gaussians(CUBIC, shape, centres, [2.0, -1.5])
    got = magnetic_moments(rho, CUBIC, centres, default_radius(CUBIC, centres))
    np.testing.assert_allclose(got, [2.0, -1.5], atol=1e-6)
    # positions outside the cell are the same atoms
    moved = centres + np.array([6.0, -12.0, 6.0])
    np.testing.assert_allclose(
        magnetic_moments(rho, CUBIC, moved, 1.5), [2.0, -1.5], atol=1e-6
    )


def test_gaussians_in_a_triclinic_cell():
    shape = (40, 44, 48)
    frac = np.array([[0.02, 0.98, 0.99], [0.5, 0.4, 0.45]])
    centres = frac @ TRICLINIC
    rho = gaussians(TRICLINIC, shape, centres, [2.2, 0.7])
    radius = default_radius(TRICLINIC, centres)
    assert radius > 8 * SIGMA
    got = magnetic_moments(rho, TRICLINIC, centres, radius)
    np.testing.assert_allclose(got, [2.2, 0.7], atol=1e-6)


def test_a_uniform_magnetization():
    shape = (60, 60, 60)
    radius = 1.7
    rho = np.full(shape, 0.3)
    got = magnetic_moments(
        rho, CUBIC, np.array([[0.0, 0.0, 0.0], [3.0, 3, 3.1]]), radius
    )
    expected = 0.3 * 4 / 3 * np.pi * radius**3
    np.testing.assert_allclose(got, expected, rtol=1e-2)
    # a sphere larger than the cell counts each point once, at its nearest image
    whole = magnetic_moments(rho, CUBIC, np.zeros((1, 3)), 100.0)
    assert whole[0] == pytest.approx(0.3 * 6.0**3, rel=1e-12)


def test_the_integral_of_the_whole_cell_is_the_sum_of_the_grid():
    rng = np.random.default_rng(1)
    shape = (20, 22, 24)
    rho = rng.normal(size=shape)
    got = magnetic_moments(rho, TRICLINIC, np.zeros((1, 3)), 1e3)
    dv = abs(np.linalg.det(TRICLINIC)) / rho.size
    assert got[0] == pytest.approx(rho.sum() * dv, rel=1e-12)


def brute_force_radius(cell, positions):
    best = np.inf
    for i, j in itertools.product(range(len(positions)), repeat=2):
        for t in itertools.product(range(-4, 5), repeat=3):
            if i == j and t == (0, 0, 0):
                continue
            v = positions[j] - positions[i] + np.array(t) @ cell
            best = min(best, np.linalg.norm(v))
    return best / 2


def test_the_default_radius():
    assert default_radius(CUBIC, np.zeros((1, 3))) == pytest.approx(3.0)
    # across the boundary: 0.2 Å apart through the image
    two = np.array([[0.1, 0.0, 0.0], [5.9, 0.0, 0.0]])
    assert default_radius(CUBIC, two) == pytest.approx(0.1)
    # one atom of a skewed cell: half its shortest lattice vector
    assert default_radius(TRICLINIC, np.zeros((1, 3))) == pytest.approx(
        brute_force_radius(TRICLINIC, np.zeros((1, 3)))
    )
    rng = np.random.default_rng(5)
    pos = rng.random((6, 3)) @ TRICLINIC
    assert default_radius(TRICLINIC, pos) == pytest.approx(
        brute_force_radius(TRICLINIC, pos)
    )


def test_errors():
    rho = np.ones((8, 8, 8))
    at = np.zeros((1, 3))
    for radius in (0.0, -1.0, float("nan")):
        with pytest.raises(InvalidParameter, match="radius of the moments"):
            magnetic_moments(rho, CUBIC, at, radius)
    with pytest.raises(InvalidParameter, match="grid"):
        magnetic_moments(np.ones((2, 8, 8, 8)), CUBIC, at, 1.0)
