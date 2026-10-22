# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Bonds between atoms: clusters, bond angles and chemical short-range order."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from mdapy._core import analysis as _native
from mdapy.core.result import Result
from mdapy.core.system import System
from mdapy.core.threads import _native_token
from mdapy.elements import SYMBOLS
from mdapy.errors import InvalidParameter, MissingProperty
from mdapy.neighbor import NeighborList
from mdapy.neighbor._neighbor_list import _given_or_built


def _elements(system: System) -> tuple[tuple[str, ...], np.ndarray] | None:
    """The elements present, by atomic number, and each atom's index among them."""
    if "element" not in system.table.columns:
        return None
    numbers = system.table["element"].to_physical().to_numpy()
    present = np.flatnonzero(np.bincount(numbers))
    lookup = np.zeros(len(SYMBOLS), dtype=np.int64)
    lookup[present] = np.arange(len(present))
    return tuple(SYMBOLS[z] for z in present), lookup[numbers]


class _Bonds:
    """Bonded pairs: closer than one cutoff, or than a cutoff per pair of
    elements (pairs not given are never bonded)."""

    def __init__(
        self, cutoff: float | None, cutoffs: Mapping[tuple[str, str], float] | None
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

    def select(
        self, system: System, neighbors: NeighborList | None
    ) -> tuple[NeighborList, tuple[tuple[str, ...], np.ndarray] | None]:
        """The bonds as a neighbor list, and the elements of the system."""
        found = _elements(system)
        largest = self.cutoff if self.cutoffs is None else max(self.cutoffs.values())
        nl = _given_or_built(system, largest, neighbors)
        if self.cutoffs is None:
            return nl.within(self.cutoff), found
        limits = self._limits(found)
        index = found[1]
        keep = nl.distances < limits[index[nl.i], index[nl.j]]
        return nl._subset(keep, largest), found

    def _limits(self, found: tuple[tuple[str, ...], np.ndarray] | None) -> np.ndarray:
        """The bond length of each pair of the elements found, 0 for none."""
        if found is None:
            raise MissingProperty(
                "cutoffs per pair of elements need the element column"
            )
        names = found[0]
        return np.array([[self.cutoffs.get((a, b), 0.0) for b in names] for a in names])

    def counts(
        self,
        system: System,
        neighbors: NeighborList | None,
        progress: Any = None,
        cancel: Any = None,
    ) -> tuple[np.ndarray, tuple[tuple[str, ...], np.ndarray] | None]:
        """Each atom's bonds by the element of the other atom, (N, elements)
        or (N, 1) without an element column, and the elements."""
        if neighbors is not None:
            bonds, found = self.select(system, neighbors)
            n = len(system)
            index = found[1] if found is not None else np.zeros(n, dtype=np.int64)
            types = len(found[0]) if found is not None else 1
            table = np.bincount(bonds.i * types + index[bonds.j], minlength=n * types)
            return table.reshape(n, types), found
        *grid, found = self._grid(system)
        table = _native.count_neighbors(*grid, progress, _native_token(cancel))
        return table, found

    def _grid(self, system: System) -> tuple:
        """The arguments of the searches on the cell grid: positions, box,
        each atom's element, the bond length of each pair of elements and
        whether the lengths are per pair; and the elements."""
        found = _elements(system)
        if self.cutoffs is None:
            names = found[0] if found is not None else ()
            limits = np.full((max(len(names), 1),) * 2, self.cutoff)
        else:
            limits = self._limits(found)
        index = found[1] if found is not None else np.zeros(len(system), dtype=np.int64)
        box = system.box
        return (
            np.ascontiguousarray(system.positions),
            np.ascontiguousarray(box.matrix),
            np.ascontiguousarray(box.origin),
            np.ascontiguousarray(box.pbc),
            index.astype(np.int32),
            len(limits),
            np.ascontiguousarray(limits, dtype=np.float64),
            self.cutoffs is not None,
            found,
        )

    def __repr__(self) -> str:
        if self.cutoffs is None:
            return f"cutoff={self.cutoff!r}"
        return f"cutoffs={self.cutoffs!r}"


@dataclass(frozen=True)
class ClusterResult(Result):
    """Clusters of bonded atoms.

    Attributes
    ----------
    cluster : numpy.ndarray
        (N,) int64 cluster of each atom, numbered from 0 by decreasing size.
    sizes : numpy.ndarray
        (clusters,) int64 number of atoms of each cluster.
    """

    cluster: np.ndarray
    sizes: np.ndarray

    def columns(self) -> dict[str, np.ndarray]:
        return {"cluster": self.cluster}


class ClusterAnalysis:
    """Clusters of atoms joined by bonds shorter than a cutoff.

    Parameters
    ----------
    cutoff : float, optional
        Bond length in Å for every pair.
    cutoffs : dict, optional
        Bond length in Å for each pair of elements ``(a, b)``, in either
        order; pairs not given are never bonded. Give ``cutoff`` or
        ``cutoffs``.
    """

    def __init__(
        self,
        cutoff: float | None = None,
        *,
        cutoffs: Mapping[tuple[str, str], float] | None = None,
    ):
        self._bonds = _Bonds(cutoff, cutoffs)

    def compute(
        self,
        system: System,
        *,
        neighbors: NeighborList | None = None,
        progress: Any = None,
        cancel: Any = None,
    ) -> ClusterResult:
        """The clusters of a system.

        Parameters
        ----------
        system : System
            The atoms; per-element cutoffs need an ``element`` column.
        neighbors : NeighborList, optional
            A neighbor list of this system with at least the largest cutoff.

        Returns
        -------
        ClusterResult
            The cluster of each atom; ties in size are ordered by their first
            atom.
        """
        if neighbors is None:
            # bonds found on the grid, without a list
            *grid, _ = self._bonds._grid(system)
            cluster, sizes = _native.clusters_search(
                *grid, progress, _native_token(cancel)
            )
        else:
            bonds, _ = self._bonds.select(system, neighbors)
            cluster, sizes = _native.clusters(bonds.offsets, bonds.j)
        return ClusterResult(cluster, sizes)

    def __repr__(self) -> str:
        return f"ClusterAnalysis({self._bonds})"


@dataclass(frozen=True)
class BondAngleResult(Result):
    """The distribution of angles between bonds of the same atom.

    Attributes
    ----------
    angle : numpy.ndarray
        (bins,) centers of the bins in degrees.
    edges : numpy.ndarray
        (bins + 1,) edges of the bins in degrees.
    counts : numpy.ndarray
        (bins,) int64 number of angles in each bin.
    density : numpy.ndarray
        (bins,) the counts normalized to unit area, per degree.
    partial : dict
        (bins,) int64 counts of the angles b-a-c for each triplet of elements
        ``(a, b, c)``: the atom ``a`` at the vertex, its neighbors ``b`` and
        ``c``, ``b`` not after ``c`` by atomic number; empty without an
        element column.
    """

    angle: np.ndarray
    edges: np.ndarray
    counts: np.ndarray
    density: np.ndarray
    partial: dict[tuple[str, str, str], np.ndarray]

    def triplet(self, vertex: str, b: str, c: str) -> np.ndarray:
        """The counts of the angles b-vertex-c, b and c in either order."""
        if (vertex, b, c) in self.partial:
            return self.partial[vertex, b, c]
        if (vertex, c, b) in self.partial:
            return self.partial[vertex, c, b]
        raise InvalidParameter(f"no partial distribution for {(vertex, b, c)}")


class BondAngleDistribution:
    """The distribution of the angles between the bonds of each atom, in
    total and for each triplet of elements.

    Parameters
    ----------
    cutoff : float, optional
        Bond length in Å for every pair.
    bins : int, optional
        Number of bins from 0 to 180 degrees. Default 180.
    cutoffs : dict, optional
        Bond length in Å for each pair of elements ``(a, b)``, in either
        order; pairs not given are never bonded. Give ``cutoff`` or
        ``cutoffs``.
    """

    def __init__(
        self,
        cutoff: float | None = None,
        bins: int = 180,
        *,
        cutoffs: Mapping[tuple[str, str], float] | None = None,
    ):
        if bins < 1:
            raise InvalidParameter(f"bins must be at least 1, not {bins!r}")
        self._bonds = _Bonds(cutoff, cutoffs)
        self.bins = int(bins)

    def compute(
        self,
        system: System,
        *,
        neighbors: NeighborList | None = None,
        progress: Any = None,
        cancel: Any = None,
    ) -> BondAngleResult:
        """The distribution of a system.

        Parameters
        ----------
        system : System
            The atoms; per-element cutoffs need an ``element`` column.
        neighbors : NeighborList, optional
            A neighbor list of this system with at least the largest cutoff.

        Returns
        -------
        BondAngleResult
            Counts of the angles between every two bonds of an atom.
        """
        if neighbors is None:
            # bonds found on the grid, without a list
            *grid, found = self._bonds._grid(system)
            counts = _native.bond_angles_search(
                *grid, self.bins, progress, _native_token(cancel)
            )
            names = found[0] if found is not None else ()
            types = max(len(names), 1)
        else:
            bonds, found = self._bonds.select(system, neighbors)
            names, index = (
                found if found is not None else ((), np.zeros(len(system), int))
            )
            types = max(len(names), 1)
            counts = _native.bond_angles(
                bonds.offsets,
                bonds.j,
                np.ascontiguousarray(bonds.vectors),
                index.astype(np.int32),
                types,
                self.bins,
                progress,
                _native_token(cancel),
            )
        pairs = [(b, c) for b in range(types) for c in range(b, types)]
        partial = {
            (names[a], names[b], names[c]): counts[a, p]
            for a in range(len(names))
            for p, (b, c) in enumerate(pairs)
        }
        total = counts.sum(axis=(0, 1))
        edges = np.linspace(0.0, 180.0, self.bins + 1)
        density = (
            total / (total.sum() * (180.0 / self.bins))
            if total.sum()
            else np.zeros(self.bins)
        )
        return BondAngleResult(
            0.5 * (edges[1:] + edges[:-1]), edges, total, density, partial
        )

    def __repr__(self) -> str:
        return f"BondAngleDistribution({self._bonds}, bins={self.bins})"


@dataclass(frozen=True)
class WarrenCowleyResult(Result):
    """Warren-Cowley short-range order parameters.

    Attributes
    ----------
    elements : tuple of str
        The elements, by atomic number.
    alpha : numpy.ndarray
        (elements, elements) parameter of each pair: row a, column b is
        1 - p(b | a) / c_b.
    """

    elements: tuple[str, ...]
    alpha: np.ndarray

    def __getitem__(self, pair: tuple[str, str]) -> float:
        """The parameter of a pair of elements (a, b)."""
        a, b = pair
        try:
            return float(self.alpha[self.elements.index(a), self.elements.index(b)])
        except ValueError:
            raise InvalidParameter(f"no parameter for {pair}") from None


class WarrenCowley:
    """Warren-Cowley parameters of chemical short-range order.

    Parameters
    ----------
    cutoff : float, optional
        Neighbors closer than cutoff, in Å.
    num_neighbors : int, optional
        The num_neighbors nearest neighbors. Give ``cutoff`` or
        ``num_neighbors``.
    """

    def __init__(
        self, cutoff: float | None = None, *, num_neighbors: int | None = None
    ):
        if (cutoff is None) == (num_neighbors is None):
            raise InvalidParameter("give either cutoff or num_neighbors")
        if cutoff is not None and not cutoff > 0:
            raise InvalidParameter(f"cutoff must be positive, not {cutoff!r}")
        if num_neighbors is not None and num_neighbors < 1:
            raise InvalidParameter("num_neighbors must be at least 1")
        self.cutoff = None if cutoff is None else float(cutoff)
        self.num_neighbors = None if num_neighbors is None else int(num_neighbors)

    def compute(
        self,
        system: System,
        *,
        neighbors: NeighborList | None = None,
        progress: Any = None,
        cancel: Any = None,
    ) -> WarrenCowleyResult:
        """The parameters of a system.

        Parameters
        ----------
        system : System
            The atoms, with an ``element`` column.
        neighbors : NeighborList, optional
            A neighbor list of this system: with at least the cutoff, or at
            least num_neighbors per atom.

        Returns
        -------
        WarrenCowleyResult
            The parameter of each pair of elements; NaN for an element whose
            atoms have no neighbors.
        """
        found = _elements(system)
        if found is None:
            raise MissingProperty("Warren-Cowley parameters need the element column")
        names, index = found
        types = len(names)
        if self.cutoff is not None:
            # each atom's neighbors by element, summed by the atom's element
            table, _ = _Bonds(self.cutoff, None).counts(
                system, neighbors, progress, cancel
            )
            pairs = np.zeros((types, types), dtype=np.int64)
            np.add.at(pairs, index, table)
        else:
            if neighbors is None:
                nl = NeighborList.nearest(system, self.num_neighbors)
            else:
                nl = _given_or_built(system, 0.0, neighbors)._nearest(
                    self.num_neighbors
                )
            pairs = np.bincount(
                index[nl.i] * types + index[nl.j], minlength=types * types
            ).reshape(types, types)
        fractions = np.bincount(index, minlength=types) / len(system)
        with np.errstate(invalid="ignore", divide="ignore"):
            probability = pairs / pairs.sum(axis=1, keepdims=True)
        return WarrenCowleyResult(names, 1.0 - probability / fractions[None, :])

    def __repr__(self) -> str:
        if self.cutoff is not None:
            return f"WarrenCowley(cutoff={self.cutoff!r})"
        return f"WarrenCowley(num_neighbors={self.num_neighbors})"
