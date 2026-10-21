# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Radial distribution functions and coordination numbers."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from mdapy.core.result import Result
from mdapy.core.system import System
from mdapy.elements import SYMBOLS
from mdapy.errors import InvalidParameter, MissingProperty
from mdapy.neighbor import NeighborList
from mdapy.neighbor._neighbor_list import _given_or_built


def _elements(system: System) -> tuple[tuple[str, ...], np.ndarray] | None:
    """The elements present, by atomic number, and each atom's index among them."""
    if "element" not in system.table.columns:
        return None
    numbers = system.table["element"].to_physical().to_numpy()
    present, index = np.unique(numbers, return_inverse=True)
    return tuple(SYMBOLS[z] for z in present), index.astype(np.int64)


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
        nl = _given_or_built(system, self.r_max, neighbors)
        n, bins = len(system), self.bins
        edges = np.linspace(0.0, self.r_max, bins + 1)
        keep = nl.distances < self.r_max
        which = np.minimum(
            (nl.distances[keep] / self.r_max * bins).astype(np.int64), bins - 1
        )
        shells = 4.0 / 3.0 * np.pi * np.diff(edges**3)
        volume = system.box.volume
        counts = np.bincount(which, minlength=bins).astype(np.float64)
        g = counts * volume / (n * n * shells) if n else np.zeros(bins)
        coordination = np.cumsum(counts) / n if n else np.zeros(bins)
        partial = {}
        found = _elements(system)
        if found is not None and n:
            names, index = found
            types = len(names)
            first = np.repeat(index, nl.counts)[keep]
            second = index[nl.j[keep]]
            table = np.bincount(
                (first * types + second) * bins + which, minlength=types * types * bins
            ).reshape(types, types, bins)
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
        if (cutoff is None) == (cutoffs is None):
            raise InvalidParameter("give either cutoff or cutoffs")
        if cutoff is not None and not cutoff > 0:
            raise InvalidParameter(f"cutoff must be positive, not {cutoff!r}")
        self.cutoff = None if cutoff is None else float(cutoff)
        self.cutoffs = None
        if cutoffs is not None:
            self.cutoffs = {}
            for (a, b), value in cutoffs.items():
                for element in (a, b):
                    if element not in SYMBOLS[1:]:
                        raise InvalidParameter(
                            f"unknown element {element!r} in cutoffs"
                        )
                if not value > 0:
                    raise InvalidParameter(f"the cutoff of {(a, b)} must be positive")
                self.cutoffs[a, b] = self.cutoffs[b, a] = float(value)

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
        found = _elements(system)
        largest = self.cutoff if self.cutoffs is None else max(self.cutoffs.values())
        nl = _given_or_built(system, largest, neighbors)
        if self.cutoffs is not None and found is None:
            raise MissingProperty(
                "cutoffs per pair of elements need the element column"
            )
        first = nl.i
        if self.cutoffs is None:
            keep = nl.distances < self.cutoff
        else:
            names, index = found
            limits = np.array(
                [[self.cutoffs.get((a, b), 0.0) for b in names] for a in names]
            )
            keep = nl.distances < limits[index[first], index[nl.j]]
        n = len(system)
        coordination = np.bincount(first[keep], minlength=n)
        partial = {}
        if found is not None:
            names, index = found
            types = len(names)
            table = np.bincount(
                first[keep] * types + index[nl.j[keep]], minlength=n * types
            ).reshape(n, types)
            partial = {name: table[:, t].copy() for t, name in enumerate(names)}
        return CoordinationResult(coordination, partial)

    def __repr__(self) -> str:
        if self.cutoffs is None:
            return f"CoordinationNumber(cutoff={self.cutoff!r})"
        return f"CoordinationNumber(cutoffs={self.cutoffs!r})"
