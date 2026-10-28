# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Smooth overlap of atomic positions (SOAP), as DScribe computes it."""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from mdapy._core import analysis as _native
from mdapy.core.system import System
from mdapy.core.threads import _native_token
from mdapy.elements import SYMBOLS
from mdapy.errors import InvalidParameter, MissingProperty
from mdapy.neighbor import NeighborList
from mdapy.neighbor._neighbor_list import _given_or_built

# the value the Gaussians of the radial functions and of the density
# decay to at their ends, as in DScribe
_THRESHOLD = 1e-3


def _gaussian_basis(
    r_cut: float, n_max: int, l_max: int
) -> tuple[np.ndarray, np.ndarray]:
    """The exponents of the Gaussian radial functions of each l, which decay
    to the threshold at lengths spaced evenly from 1 Å to the cutoff, and
    the Löwdin matrices that make each set orthonormal."""
    ends = np.linspace(1.0, r_cut, n_max)
    alphas = np.zeros((l_max + 1, n_max))
    betas = np.zeros((l_max + 1, n_max, n_max))
    for degree in range(l_max + 1):
        alpha = -np.log(_THRESHOLD / ends**degree) / ends**2
        pair = alpha[:, None] + alpha[None, :]
        overlap = 0.5 * math.gamma(degree + 1.5) * pair ** (-degree - 1.5)
        values, vectors = np.linalg.eigh(overlap)
        if values.min() <= 0:
            raise InvalidParameter(
                "the radial functions are linearly dependent: use fewer (n_max) "
                "or a longer cutoff (r_cut)"
            )
        alphas[degree] = alpha
        betas[degree] = (vectors / np.sqrt(values)) @ vectors.T
    return alphas, betas


class SOAP:
    """The smooth overlap of atomic positions (SOAP) of each atom: the power
    spectrum of the Gaussian density of its neighbors, in Gaussian radial
    functions and spherical harmonics, the same as DScribe's SOAP with its
    default Gaussian radial functions.

    Parameters
    ----------
    species : sequence of str
        The elements, each a part of the descriptor; every atom must be one
        of them, so that systems of other compositions share the same
        layout.
    r_cut : float, optional
        The cutoff of the radial functions in Å, above 1. Default 5.
    n_max : int, optional
        The number of radial functions. Default 8.
    l_max : int, optional
        The highest degree of the spherical harmonics, at most 20. Default 6.
    sigma : float, optional
        The width in Å of the Gaussian on each neighbor. Default 0.5.
    crossover : bool, optional
        Keep the products of the densities of two different elements.
        Default True.
    """

    def __init__(
        self,
        species: Sequence[str],
        r_cut: float = 5.0,
        n_max: int = 8,
        l_max: int = 6,
        sigma: float = 0.5,
        *,
        crossover: bool = True,
    ):
        names = [species] if isinstance(species, str) else list(species)
        if not names:
            raise InvalidParameter("give at least one element in species")
        for name in names:
            if name not in SYMBOLS[1:]:
                raise InvalidParameter(f"unknown element {name!r} in species")
        if len(set(names)) != len(names):
            raise InvalidParameter("an element is given twice in species")
        if not r_cut > 1:
            raise InvalidParameter(f"r_cut must be above 1 Å, not {r_cut!r}")
        for name, value, low, high in (
            ("n_max", n_max, 1, None),
            ("l_max", l_max, 0, 20),
        ):
            if (
                isinstance(value, bool)
                or int(value) != value
                or value < low
                or (high is not None and value > high)
            ):
                limit = (
                    f"from {low} to {high}" if high is not None else f"at least {low}"
                )
                raise InvalidParameter(
                    f"{name} must be an integer {limit}, not {value!r}"
                )
        if not sigma > 0:
            raise InvalidParameter(f"sigma must be positive, not {sigma!r}")
        # by atomic number, as DScribe orders them
        self.species = tuple(sorted(names, key=SYMBOLS.index))
        self.r_cut = float(r_cut)
        self.n_max = int(n_max)
        self.l_max = int(l_max)
        self.sigma = float(sigma)
        self.crossover = bool(crossover)
        self.__alphas, self.__betas = _gaussian_basis(
            self.r_cut, self.n_max, self.l_max
        )

    @property
    def cutoff(self) -> float:
        """The reach of the neighbors in Å: the cutoff and the distance over
        which the Gaussian of a neighbor decays to a thousandth."""
        return self.r_cut + self.sigma * math.sqrt(-2 * math.log(_THRESHOLD))

    @property
    def dimension(self) -> int:
        """The number of values per atom."""
        s, n, degrees = len(self.species), self.n_max, self.l_max + 1
        if self.crossover:
            return (s * n) * (s * n + 1) // 2 * degrees
        return s * degrees * n * (n + 1) // 2

    def descriptors(
        self,
        system: System,
        *,
        neighbors: NeighborList | None = None,
        progress: Any = None,
        cancel: Any = None,
    ) -> np.ndarray:
        """The SOAP of each atom.

        Parameters
        ----------
        system : System
            The atoms, with an ``element`` column of the species.
        neighbors : NeighborList, optional
            A neighbor list of this system reaching at least ``cutoff``.

        Returns
        -------
        numpy.ndarray
            (N, dimension) the power spectrum of each atom: for each pair of
            species in the order of their atomic numbers, for each degree,
            for each pair of radial functions, the second from the first on
            within one species.

        Raises
        ------
        MissingProperty
            Without an element column.
        InvalidParameter
            For an atom of an element not among the species.
        """
        if "element" not in system.table.columns:
            raise MissingProperty("SOAP needs the element column")
        elements = np.asarray(system["element"])
        index = {name: k for k, name in enumerate(self.species)}
        missing = sorted(set(elements.tolist()) - set(index))
        if missing:
            raise InvalidParameter(
                f"element {missing[0]} is not among the species {self.species}"
            )
        species = np.array([index[e] for e in elements], dtype=np.int32)
        neighbors = _given_or_built(system, self.cutoff, neighbors)
        with neighbors._reading():
            # the pairs within the cutoff, of a list that may reach further
            keep = neighbors.distances < self.cutoff
            owner = np.repeat(np.arange(len(system)), np.diff(neighbors.offsets))
            counts = np.bincount(owner[keep], minlength=len(system))
            offsets = np.concatenate([[0], np.cumsum(counts)])
            return _native.soap(
                species,
                len(self.species),
                np.ascontiguousarray(system["positions"]),
                np.ascontiguousarray(system.box.matrix),
                offsets.astype(np.int64),
                np.ascontiguousarray(neighbors.j[keep]),
                np.ascontiguousarray(neighbors.shifts[keep]),
                self.n_max,
                self.l_max,
                1.0 / (2.0 * self.sigma**2),
                np.ascontiguousarray(self.__alphas.ravel()),
                np.ascontiguousarray(self.__betas.ravel()),
                self.crossover,
                progress,
                _native_token(cancel),
            )

    def __repr__(self) -> str:
        return (
            f"SOAP({self.species}, r_cut={self.r_cut}, n_max={self.n_max}, "
            f"l_max={self.l_max}, sigma={self.sigma})"
        )
