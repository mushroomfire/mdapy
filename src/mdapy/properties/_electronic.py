# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The electronic structure of a crystal from the DFT calculator: the bands
along a path through the Brillouin zone and the density of states, at the
density of the self-consistent field."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from mdapy.calculators._base import Calculator
from mdapy.core.system import System
from mdapy.errors import InvalidParameter
from mdapy.properties import _paths
from mdapy.properties._phonons import (
    _kinds,
    _mesh,
    _read_only,
    _repeat,
    _tetrahedra,
    _tetrahedron_density,
    primitive_cell,
)

# the largest distance in 1/Å (with 2 pi) between neighbours of the default
# mesh of the density of states along each reciprocal lattice vector
_SPACING = 0.15
# number of energies of the density of states by default
_ENERGIES = 1001
# the largest number of pairs of a tetrahedron and an energy inside it at a
# time
_PAIRS = 1 << 21


@dataclass(frozen=True)
class ElectronicBandStructure:
    """The energies of the electrons along a path through the Brillouin zone.

    Attributes
    ----------
    k : numpy.ndarray
        (P, 3) float64 wave vectors in units of the reciprocal lattice vectors
        of the cell.
    distances : numpy.ndarray
        (P,) float64 distance of each wave vector along the path, in 1/Å.
    energies : numpy.ndarray
        (S, P, B) float64 energies in eV relative to ``fermi``, ascending at
        each wave vector; S = 2 with spin, up then down, else 1.
    fermi : float64
        The Fermi level of the self-consistent field in eV; with fixed
        occupations its highest occupied level. ``energies + fermi`` are the
        eigenvalues.
    labels : tuple of (float64, str)
        The distance and the label of each special point; two points that
        meet at a break of the path share a label such as "U|K".
    """

    k: np.ndarray
    distances: np.ndarray
    energies: np.ndarray
    fermi: float
    labels: tuple[tuple[float, str], ...]

    def __post_init__(self):
        _read_only(self.k, self.distances, self.energies)


@dataclass(frozen=True)
class ElectronicDOS:
    """The electronic density of states of the cell.

    Attributes
    ----------
    energies : numpy.ndarray
        (F,) float64 energies in eV relative to ``fermi``.
    dos : numpy.ndarray
        (S, F) float64 states per eV of the cell; S = 2 with spin, up then
        down, else 1 with both spins together. Below the Fermi level it
        integrates to the number of electrons, to the accuracy of the mesh.
    fermi : float64
        The Fermi level of the self-consistent field in eV; with fixed
        occupations its highest occupied level.
    mesh : numpy.ndarray
        (3,) int32 number of wave vectors along each reciprocal lattice
        vector.
    """

    energies: np.ndarray
    dos: np.ndarray
    fermi: float
    mesh: np.ndarray

    def __post_init__(self):
        _read_only(self.energies, self.dos, self.mesh)


class ElectronicStructure:
    """The electronic structure of a crystal at the density of its
    self-consistent field.

    Made by ``electronic_structure``. Wave vectors are in units of the
    reciprocal lattice vectors of the cell: k = (h, k, l) is h b1 + k b2 +
    l b3 with a_i · b_j = 2π δ_ij.

    Attributes
    ----------
    system : System
        The crystal.
    fermi : float64
        The Fermi level of the self-consistent field in eV; with fixed
        occupations its highest occupied level.
    spins : int32
        2 with spin, else 1.
    bands : int32
        Number of bands of the self-consistent field at each wave vector.
    """

    def __init__(self, system: System, calculator: Calculator):
        from mdapy.calculators._dft.calculator import DFT

        if not isinstance(calculator, DFT):
            raise InvalidParameter(
                f"the electronic structure needs the calculator DFT, not {calculator!r}"
            )
        self.__calculator = calculator
        self.__fixed = calculator._fixed_density(system)
        self.system = system
        self.fermi = float(self.__fixed["fermi"])
        self.spins = int(self.__fixed["spins"])
        self.bands = int(self.__fixed["bands"])

    def __repr__(self) -> str:
        atoms = f"{len(self.system)} atom" + ("s" if len(self.system) > 1 else "")
        return f"ElectronicStructure({atoms}, {self.spins} spin, {self.bands} bands)"

    def energies(self, k: ArrayLike, *, bands: int | None = None) -> np.ndarray:
        """The energies of the electrons at wave vectors.

        Parameters
        ----------
        k : array_like of float64
            (K, 3) wave vectors, or one (3,).
        bands : int32, optional
            Number of bands, from the lowest. Default that of the
            self-consistent field.

        Returns
        -------
        numpy.ndarray
            (S, K, B) float64 energies in eV relative to ``fermi``, ascending
            at each wave vector; (S, B) for one wave vector.

        Raises
        ------
        InvalidParameter
            For wave vectors that do not have 3 values, or bands that are
            not positive.
        CalculationFailed
            If a band does not converge.
        """
        k = np.asarray(k, dtype=float)
        single = k.ndim == 1
        k = np.atleast_2d(k)
        if k.ndim != 2 or k.shape[1] != 3 or len(k) == 0:
            raise InvalidParameter("wave vectors must be (K, 3) or (3,)")
        if bands is not None and int(bands) < 1:
            raise InvalidParameter(f"bands must be at least 1, not {bands!r}")
        e = self.__calculator._bands(self.__fixed, k, bands) - self.fermi
        return e[:, 0] if single else e

    def band_structure(
        self,
        path: str | Sequence[Sequence[tuple[str, ArrayLike]]],
        *,
        points: int = 51,
        bands: int | None = None,
    ) -> ElectronicBandStructure:
        """The energies of the electrons along a path through the Brillouin
        zone.

        Parameters
        ----------
        path : str or sequence of sequences of (str, array_like)
            "auto" for the standard path of the crystal by its symmetry,
            from seekpath; "sc", "fcc", "bcc" or "hexagonal" for the standard
            path of that lattice, the cell having the axes of its
            conventional cell along x, y and z, or a along x and c along z;
            or the sections of a path, each a sequence of labeled wave
            vectors in units of the reciprocal lattice vectors of the cell,
            joined by straight lines.
        points : int32, optional
            Number of wave vectors on each line, its ends included.
            Default 51.
        bands : int32, optional
            Number of bands, from the lowest. Default that of the
            self-consistent field.

        Returns
        -------
        ElectronicBandStructure
            The wave vectors, their distances along the path, the energies
            relative to the Fermi level, the Fermi level and the labels of
            the special points.

        Raises
        ------
        ImportError
            For path "auto" without seekpath.
        InvalidParameter
            For a path that is none of these, fewer than 2 points, or bands
            that are not positive.
        CalculationFailed
            If a band does not converge.
        """
        if path == "auto":
            sections = _paths.auto_path(self.system, _kinds(self.system))
        elif isinstance(path, str):
            sections = _paths.standard_path(
                self.system.box.matrix, primitive_cell(self.system).box.matrix, path
            )
        else:
            sections = _paths.given_path(path)
        k, distances, labels, _ = _paths.sample(
            sections, points, self.__fixed["reciprocal"]
        )
        return ElectronicBandStructure(
            k=k,
            distances=distances,
            energies=self.energies(k, bands=bands),
            fermi=self.fermi,
            labels=labels,
        )

    def dos(
        self,
        mesh: int | ArrayLike | None = None,
        *,
        energies: ArrayLike | None = None,
        sigma: float | None = None,
        bands: int | None = None,
    ) -> ElectronicDOS:
        """The electronic density of states, from the energies on a mesh of
        wave vectors, by the linear tetrahedron method or with Gaussians.

        Parameters
        ----------
        mesh : int32 or array_like, optional
            Number of wave vectors along each reciprocal lattice vector, of
            a mesh through Γ. Default the fewest that leave at most
            0.15 1/Å between neighbours along each.
        energies : array_like of float64, optional
            The energies in eV relative to the Fermi level at which to give
            the density. Default 1001 from 0.5 eV (and 3 sigma) below the
            lowest band to the lowest energy of the highest band, above which
            states are missing.
        sigma : float64, optional
            Width in eV of a Gaussian for each state; by default the linear
            tetrahedron method.
        bands : int32, optional
            Number of bands, from the lowest. Default that of the
            self-consistent field.

        Returns
        -------
        ElectronicDOS
            The energies relative to the Fermi level, the density of states
            of the cell, the Fermi level and the mesh.

        Raises
        ------
        InvalidParameter
            For a mesh that is not one or three positive integers, or a
            sigma or bands that are not positive.
        CalculationFailed
            If a band does not converge.
        """
        from mdapy.calculators._dft import symmetry as symmetries

        if sigma is not None and not sigma > 0:
            raise InvalidParameter(f"sigma must be positive, not {sigma!r}")
        fixed = self.__fixed
        reciprocal = fixed["reciprocal"]
        if mesh is None:
            lengths = np.linalg.norm(reciprocal, axis=1)
            counts = np.maximum(1, np.ceil(lengths / _SPACING - 1e-9)).astype(int)
        else:
            counts = _repeat(mesh, "mesh")
        k = _mesh(counts)
        # the wave vectors that the operations of the field, or -1, relate
        # have the same energies: those of one of each star are solved
        rotations = np.eye(3, dtype=np.int32)[None]
        if fixed["symmetric"]:
            rotations = symmetries.operations(
                fixed["cell"], fixed["positions"], fixed["kinds"], fixed["moments"]
            ).rotations
        first, where = np.unique(symmetries.stars(k, rotations), return_inverse=True)
        e = self.energies(k[first], bands=bands)
        if energies is None:
            margin = 0.5 + (0.0 if sigma is None else 3 * sigma)
            grid = np.linspace(e.min() - margin, e[..., -1].min(), _ENERGIES)
        else:
            grid = np.asarray(energies, dtype=float).reshape(-1)
        if sigma is None:
            dos = _tetrahedron_sum(e[:, where], counts, reciprocal, grid)
        else:
            dos = _gaussian_sum(e, np.bincount(where) / len(k), grid, float(sigma))
        # without spin each state holds two electrons
        return ElectronicDOS(
            energies=grid,
            dos=dos * (2 / self.spins),
            fermi=self.fermi,
            mesh=counts.astype(np.int32),
        )


def _tetrahedron_sum(
    energies: np.ndarray, counts: np.ndarray, reciprocal: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    """(S, F) density of states of energies (S, K, B) on the mesh of counts
    (in the order of _mesh) at the energies grid (F,), ascending: of each
    band the mean of those of the tetrahedra of the mesh, summed over the
    bands; each band integrates to 1."""
    tetrahedra = _tetrahedra(counts, reciprocal)
    out = np.zeros((len(energies), len(grid)))
    # the energies of the grid inside each tetrahedron alone, at most about
    # _PAIRS pairs at a time
    for s, values in enumerate(energies):
        corners = np.sort(values[tetrahedra], axis=1).transpose(0, 2, 1).reshape(-1, 4)
        low = np.searchsorted(grid, corners[:, 0], side="right")
        width = np.maximum(np.searchsorted(grid, corners[:, 3], side="left") - low, 0)
        ends = np.cumsum(width)
        if len(ends) == 0 or ends[-1] == 0:
            continue
        cuts = np.searchsorted(ends, np.arange(_PAIRS, ends[-1], _PAIRS), side="right")
        bounds = np.unique(np.concatenate([[0], cuts, [len(corners)]]))
        for start, stop in zip(bounds[:-1], bounds[1:]):
            w = width[start:stop]
            row = np.repeat(np.arange(start, stop), w)
            offset = np.arange(len(row)) - np.repeat(np.cumsum(w) - w, w)
            index = low[row] + offset
            value = _tetrahedron_density(
                *(corners[row, j] for j in range(4)), grid[index]
            )
            out[s] += np.bincount(index, weights=value, minlength=len(grid))
    return out / len(tetrahedra)


def _gaussian_sum(
    energies: np.ndarray, weights: np.ndarray, grid: np.ndarray, sigma: float
) -> np.ndarray:
    """(S, F) density of states of energies (S, K, B) at wave vectors of
    weights (K,) adding up to one, at the energies grid (F,): a Gaussian of
    width sigma for each state, summed over the bands."""
    out = np.zeros((len(energies), len(grid)))
    for s, values in enumerate(energies):
        for k, w in enumerate(weights):
            x = (grid[None, :] - values[k][:, None]) / sigma
            out[s] += w * np.exp(-0.5 * x * x).sum(axis=0)
    return out / (sigma * math.sqrt(2 * math.pi))


def electronic_structure(system: System, calculator: Calculator) -> ElectronicStructure:
    """The electronic structure of a crystal: its self-consistent field, of
    which the bands at any wave vectors then follow.

    Parameters
    ----------
    system : System
        Periodic along a, b and c, with an ``element`` column.
    calculator : Calculator
        The calculator ``DFT``.

    Returns
    -------
    ElectronicStructure
        The state of the self-consistent field, with ``band_structure``
        and ``dos``.

    Raises
    ------
    InvalidParameter
        For another calculator, a ``DFT`` with DFT+U or with a communicator
        of several processes, or a system it does not take.
    CalculationFailed
        If the self-consistent field does not converge.
    """
    return ElectronicStructure(system, calculator)
