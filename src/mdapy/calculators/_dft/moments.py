# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Magnetic moments of the atoms of a periodic cell: the magnetization density
on the real-space grid integrated within a sphere around each atom. NumPy
only."""

import itertools

import numpy as np

from mdapy.errors import InvalidParameter

_SHIFTS = np.array(list(itertools.product((-1, 0, 1), repeat=3)), dtype=np.float64)
_SHIFTS_FAR = np.array(
    list(itertools.product((-2, -1, 0, 1, 2), repeat=3)), dtype=np.float64
)


def default_radius(cell: np.ndarray, positions: np.ndarray) -> float:
    """Half the shortest distance between two atoms of a periodic cell, the
    periodic images of the atoms included.

    Parameters
    ----------
    cell : (3, 3) float64
        The lattice vectors in rows, in Å.
    positions : (N, 3) float64
        The positions of the atoms, in Å.

    Returns
    -------
    float64
        The radius in Å. With one atom, half the shortest lattice vector.
    """
    cell = np.asarray(cell, dtype=np.float64)
    frac = np.asarray(positions, dtype=np.float64) @ np.linalg.inv(cell)
    best = np.inf
    for i in range(len(frac)):
        d = frac[i:] - frac[i]
        d -= np.round(d)
        v = (d[:, None, :] + _SHIFTS_FAR[None]) @ cell
        dist = np.linalg.norm(v, axis=2)
        dist[0][np.all(_SHIFTS_FAR == 0, axis=1)] = np.inf  # the atom itself
        best = min(best, float(dist.min()))
    return 0.5 * best


def magnetic_moments(
    magnetization: np.ndarray,
    cell: np.ndarray,
    positions: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Integrates a density on the grid of a periodic cell within a sphere
    around each atom.

    The grid point (i, j, k) sits at the fractional coordinates (i / n1,
    j / n2, k / n3); a point counts for an atom when its nearest periodic image
    is within the radius, found over the images of the neighboring cells (the
    cell should not be extremely skewed). Only the grid points of a box around
    each atom are visited.

    Parameters
    ----------
    magnetization : (n1, n2, n3) float64
        The density per Å^3 on the grid, such as spin up minus spin down.
    cell : (3, 3) float64
        The lattice vectors in rows, in Å.
    positions : (N, 3) float64
        The positions of the atoms, in Å, relative to the origin of the grid.
    radius : float64
        The radius of the spheres in Å.

    Returns
    -------
    (N,) float64
        The integral of the density within the sphere of each atom, in the
        unit of the density times Å^3.

    Raises
    ------
    InvalidParameter
        If the radius is not positive, or the grid is not three-dimensional.
    """
    if not radius > 0:
        raise InvalidParameter("the radius of the moments must be positive")
    rho = np.asarray(magnetization, dtype=np.float64)
    if rho.ndim != 3:
        raise InvalidParameter("the magnetization must be a grid (n1, n2, n3)")
    cell = np.asarray(cell, dtype=np.float64)
    inverse = np.linalg.inv(cell)
    frac = np.asarray(positions, dtype=np.float64) @ inverse
    n = np.array(rho.shape)
    dv = abs(np.linalg.det(cell)) / rho.size
    # reach of the sphere along each lattice vector, in fractions of it
    reach = radius * np.linalg.norm(inverse, axis=0)
    out = np.zeros(len(frac))
    for a, f in enumerate(frac):
        lo = np.floor((f - reach) * n).astype(int)
        hi = np.ceil((f + reach) * n).astype(int)
        index = [np.unique(np.arange(lo[x], hi[x] + 1) % n[x]) for x in range(3)]
        grid = np.meshgrid(*(i / n[x] for x, i in enumerate(index)), indexing="ij")
        d = np.stack(grid, axis=-1) - f
        d -= np.round(d)
        inside = np.zeros(d.shape[:3], dtype=bool)
        for shift in _SHIFTS:
            v = (d + shift) @ cell
            inside |= np.einsum("...i,...i", v, v) <= radius * radius
        out[a] = rho[np.ix_(*index)][inside].sum() * dv
    return out
