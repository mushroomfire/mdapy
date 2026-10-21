# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Radial distribution functions and coordination numbers."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from mdapy._core import analysis as _native
from mdapy.analysis._bonds import _Bonds, _elements
from mdapy.core.result import Result
from mdapy.core.system import System
from mdapy.core.threads import _native_token
from mdapy.errors import InvalidParameter
from mdapy.neighbor import NeighborList
from mdapy.neighbor._neighbor_list import _given_or_built


@dataclass(frozen=True)
class RadialDistributionResult(Result):
    """A radial distribution function and its partial functions.

    Attributes
    ----------
    r : numpy.ndarray
        (bins,) centers of the bins in Å.
    edges : numpy.ndarray
        (bins + 1,) edges of the bins in Å.
    g : numpy.ndarray
        (bins,) the radial distribution function.
    coordination : numpy.ndarray
        (bins,) mean number of neighbors closer than the upper edge of each bin.
    partial : dict
        (bins,) partial function of each pair of elements ``(a, b)``, ``a``
        not after ``b`` by atomic number; empty without an element column.
    """

    r: np.ndarray
    edges: np.ndarray
    g: np.ndarray
    coordination: np.ndarray
    partial: dict[tuple[str, str], np.ndarray]

    def pair(self, a: str, b: str) -> np.ndarray:
        """The partial function of elements a and b, in either order."""
        if (a, b) in self.partial:
            return self.partial[a, b]
        if (b, a) in self.partial:
            return self.partial[b, a]
        raise InvalidParameter(f"no partial function for {(a, b)}")


class RadialDistributionFunction:
    """The radial distribution function g(r) and its partial functions.

    Parameters
    ----------
    r_max : float
        Largest distance in Å.
    bins : int, optional
        Number of bins of equal width. Default 200.
    """

    def __init__(self, r_max: float, bins: int = 200):
        if not r_max > 0:
            raise InvalidParameter(f"r_max must be positive, not {r_max!r}")
        if bins < 1:
            raise InvalidParameter(f"bins must be at least 1, not {bins!r}")
        self.r_max = float(r_max)
        self.bins = int(bins)

    def compute(
        self,
        system: System,
        *,
        neighbors: NeighborList | None = None,
        progress: Any = None,
        cancel: Any = None,
    ) -> RadialDistributionResult:
        """The functions of a system.

        Parameters
        ----------
        system : System
            The atoms; partial functions need an ``element`` column.
        neighbors : NeighborList, optional
            A neighbor list of this system with a cutoff of at least ``r_max``.

        Returns
        -------
        RadialDistributionResult
            The functions, normalized by the number density of the box.
        """
        n, bins = len(system), self.bins
        found = _elements(system)
        names, index = found if found is not None else ((), np.zeros(n, dtype=np.int64))
        types = max(len(names), 1)
        if neighbors is None:
            # counted during the search, without storing the pairs
            box = system.box
            table = _native.pair_histogram(
                np.ascontiguousarray(system.positions),
                np.ascontiguousarray(box.matrix),
                np.ascontiguousarray(box.origin),
                np.ascontiguousarray(box.pbc),
                self.r_max,
                bins,
                index.astype(np.int32),
                types,
                progress,
                _native_token(cancel),
            )
        else:
            nl = _given_or_built(system, self.r_max, neighbors)
            keep = nl.distances < self.r_max
            which = np.minimum(
                (nl.distances[keep] / self.r_max * bins).astype(np.int64), bins - 1
            )
            first, second = index[nl.i[keep]], index[nl.j[keep]]
            table = np.bincount(
                (first * types + second) * bins + which, minlength=types * types * bins
            ).reshape(types, types, bins)
        edges = np.linspace(0.0, self.r_max, bins + 1)
        shells = 4.0 / 3.0 * np.pi * np.diff(edges**3)
        volume = system.box.volume
        counts = table.sum(axis=(0, 1)).astype(np.float64)
        g = counts * volume / (n * n * shells) if n else np.zeros(bins)
        coordination = np.cumsum(counts) / n if n else np.zeros(bins)
        partial = {}
        if found is not None and n:
            sizes = np.bincount(index, minlength=types)
            for a in range(types):
                for b in range(a, types):
                    partial[names[a], names[b]] = (
                        table[a, b] * volume / (sizes[a] * sizes[b] * shells)
                    )
        return RadialDistributionResult(
            r=0.5 * (edges[1:] + edges[:-1]),
            edges=edges,
            g=g,
            coordination=coordination,
            partial=partial,
        )

    def __repr__(self) -> str:
        return f"RadialDistributionFunction(r_max={self.r_max!r}, bins={self.bins})"


@dataclass(frozen=True)
class CoordinationResult(Result):
    """Number of neighbors of each atom.

    Attributes
    ----------
    coordination : numpy.ndarray
        (N,) int64 number of neighbors.
    partial : dict
        (N,) int64 number of neighbors of each element, by element; empty
        without an element column.
    """

    coordination: np.ndarray
    partial: dict[str, np.ndarray]

    def columns(self) -> dict[str, np.ndarray]:
        columns = {"coordination": self.coordination}
        for element, counts in self.partial.items():
            columns[f"coordination_{element}"] = counts
        return columns


class CoordinationNumber:
    """The number of neighbors of each atom within a cutoff.

    Parameters
    ----------
    cutoff : float, optional
        Cutoff in Å for every pair.
    cutoffs : dict, optional
        Cutoff in Å for each pair of elements ``(a, b)``, in either order;
        pairs not given are never neighbors. Give ``cutoff`` or ``cutoffs``.
    """

    def __init__(
        self,
        cutoff: float | None = None,
        *,
        cutoffs: Mapping[tuple[str, str], float] | None = None,
    ):
        self._bonds = _Bonds(cutoff, cutoffs)
        self.cutoff, self.cutoffs = self._bonds.cutoff, self._bonds.cutoffs

    def compute(
        self,
        system: System,
        *,
        neighbors: NeighborList | None = None,
        progress: Any = None,
        cancel: Any = None,
    ) -> CoordinationResult:
        """The counts of a system.

        Parameters
        ----------
        system : System
            The atoms; per-element cutoffs and counts need an ``element`` column.
        neighbors : NeighborList, optional
            A neighbor list of this system with at least the largest cutoff.

        Returns
        -------
        CoordinationResult
            The counts of each atom.
        """
        bonds, found = self._bonds.select(system, neighbors)
        n = len(system)
        partial = {}
        if found is not None:
            names, index = found
            types = len(names)
            table = np.bincount(
                bonds.i * types + index[bonds.j], minlength=n * types
            ).reshape(n, types)
            partial = {name: table[:, t].copy() for t, name in enumerate(names)}
        return CoordinationResult(bonds.counts, partial)

    def __repr__(self) -> str:
        return f"CoordinationNumber({self._bonds})"


@dataclass(frozen=True)
class StructureFactorResult(Result):
    """A static structure factor and its partial factors.

    Attributes
    ----------
    q : numpy.ndarray
        (bins,) centers of the bins of |q| in 1/Å.
    edges : numpy.ndarray
        (bins + 1,) edges of the bins in 1/Å.
    s : numpy.ndarray
        (bins,) the structure factor averaged over the vectors of each bin;
        NaN in bins without vectors.
    vectors : numpy.ndarray
        (bins,) int64 number of vectors in each bin, one of each pair q, -q.
    partial : dict
        (bins,) partial factor of each pair of elements ``(a, b)``, ``a``
        not after ``b`` by atomic number; empty without an element column.
    """

    q: np.ndarray
    edges: np.ndarray
    s: np.ndarray
    vectors: np.ndarray
    partial: dict[tuple[str, str], np.ndarray]

    def pair(self, a: str, b: str) -> np.ndarray:
        """The partial factor of elements a and b, in either order."""
        if (a, b) in self.partial:
            return self.partial[a, b]
        if (b, a) in self.partial:
            return self.partial[b, a]
        raise InvalidParameter(f"no partial factor for {(a, b)}")


class StructureFactor:
    """The static structure factor S(q) of a periodic system, summed over the
    wave vectors that fit the box.

    Parameters
    ----------
    q_max : float
        Largest |q| in 1/Å.
    bins : int, optional
        Number of bins of |q| of equal width. Default 200.
    """

    def __init__(self, q_max: float, bins: int = 200):
        if not q_max > 0:
            raise InvalidParameter(f"q_max must be positive, not {q_max!r}")
        if bins < 1:
            raise InvalidParameter(f"bins must be at least 1, not {bins!r}")
        self.q_max = float(q_max)
        self.bins = int(bins)

    def compute(
        self, system: System, *, progress: Any = None, cancel: Any = None
    ) -> StructureFactorResult:
        """The structure factor of a system.

        Parameters
        ----------
        system : System
            The atoms in a box periodic in every direction; partial factors
            need an ``element`` column.

        Returns
        -------
        StructureFactorResult
            The factor averaged over shells of |q|.
        """
        if not system.box.pbc.all():
            raise InvalidParameter(
                "the structure factor needs a box periodic in every direction"
            )
        n, bins = len(system), self.bins
        found = _elements(system)
        names, index = found if found is not None else ((), np.zeros(n, dtype=np.int64))
        types = max(len(names), 1)
        sums, vectors = _native.structure_factor(
            np.ascontiguousarray(system.positions),
            np.ascontiguousarray(system.box.matrix),
            index.astype(np.int32),
            types,
            self.q_max,
            bins,
            0,
            progress,
            _native_token(cancel),
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            s = sums.sum(axis=(0, 1)) / n / vectors if n else np.full(bins, np.nan)
            partial = {}
            if found is not None and n:
                sizes = np.bincount(index, minlength=types)
                for a in range(types):
                    for b in range(a, types):
                        partial[names[a], names[b]] = (
                            sums[a, b] / np.sqrt(sizes[a] * sizes[b]) / vectors
                        )
        edges = np.linspace(0.0, self.q_max, bins + 1)
        return StructureFactorResult(
            q=0.5 * (edges[1:] + edges[:-1]),
            edges=edges,
            s=s,
            vectors=vectors,
            partial=partial,
        )

    def __repr__(self) -> str:
        return f"StructureFactor(q_max={self.q_max!r}, bins={self.bins})"
