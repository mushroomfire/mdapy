# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The electronic structure of a crystal from the DFT calculator: the bands
along a path through the Brillouin zone, at the density of the
self-consistent field."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from mdapy.calculators._base import Calculator
from mdapy.core.system import System
from mdapy.errors import InvalidParameter
from mdapy.properties import _paths
from mdapy.properties._phonons import _kinds, _read_only, primitive_cell


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
        The state of the self-consistent field, with ``band_structure``.

    Raises
    ------
    InvalidParameter
        For another calculator, a ``DFT`` with DFT+U or with a communicator
        of several processes, or a system it does not take.
    CalculationFailed
        If the self-consistent field does not converge.
    """
    return ElectronicStructure(system, calculator)
