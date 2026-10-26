# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Bonds between atoms: clusters, bond angles and chemical short-range order."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mdapy._core import analysis as _native
from mdapy.core.result import Result
from mdapy.core.system import System
from mdapy.core.threads import _native_token
from mdapy.elements import SYMBOLS, atomic_masses
from mdapy.errors import InvalidParameter, MissingProperty
from mdapy.neighbor import NeighborList
from mdapy.neighbor._neighbor_list import (
    _given_or_built,
    _nearest_given_or_built,
    _view,
)


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

    def listed(
        self, system: System, neighbors: NeighborList | None
    ) -> tuple[NeighborList, float, np.ndarray | None, np.ndarray | None, Any]:
        """The bonds in a list, read where it is: the list, the cutoff (0
        with cutoffs per pair of elements), each atom's element and the bond
        length of each pair of elements (None with one cutoff); and the
        elements of the system."""
        found = _elements(system)
        largest = self.cutoff if self.cutoffs is None else max(self.cutoffs.values())
        nl = _given_or_built(system, largest, neighbors)
        if self.cutoffs is None:
            return nl, self.cutoff, None, None, found
        limits = self.__limits(found)
        return nl, 0.0, found[1].astype(np.int32), limits, found

    def __limits(self, found: tuple[tuple[str, ...], np.ndarray] | None) -> np.ndarray:
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
        *grid, found = self._grid(system)
        if neighbors is not None:
            largest = (
                self.cutoff if self.cutoffs is None else max(self.cutoffs.values())
            )
            nl = _given_or_built(system, largest, neighbors)
            index, types, limits = grid[4], grid[5], grid[6]
            with nl._reading():
                table = _native.count_neighbors_list(
                    nl.offsets,
                    nl.j,
                    nl.distances,
                    index,
                    types,
                    limits,
                    progress,
                    _native_token(cancel),
                )
            return table, found
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
            limits = self.__limits(found)
        index = found[1] if found is not None else np.zeros(len(system), dtype=np.int64)
        box = system.box
        return (
            np.ascontiguousarray(system["positions"]),
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
    spanning : numpy.ndarray
        (clusters,) bool, whether the cluster joins another periodic image of
        itself, spanning the box; its unwrapped positions, center of mass and
        gyration then depend on the order of the bonds.
    unwrapped : numpy.ndarray or None
        (N, 3) positions in Å with each cluster unbroken by the periodic
        boundaries, the first atom of each cluster at its own position, when
        computed.
    masses : numpy.ndarray or None
        (clusters,) total mass of each cluster in amu, or its number of atoms
        when the system has neither masses nor elements, with the center of
        mass.
    center_of_mass : numpy.ndarray or None
        (clusters, 3) center of mass of each cluster in Å, from the unwrapped
        positions, when computed.
    radius_of_gyration : numpy.ndarray or None
        (clusters,) radius of gyration of each cluster in Å, when computed.
    gyration_tensor : numpy.ndarray or None
        (clusters, 3, 3) gyration tensor of each cluster in Å², with the
        radius of gyration.
    """

    cluster: np.ndarray
    sizes: np.ndarray
    spanning: np.ndarray
    unwrapped: np.ndarray | None = None
    masses: np.ndarray | None = None
    center_of_mass: np.ndarray | None = None
    radius_of_gyration: np.ndarray | None = None
    gyration_tensor: np.ndarray | None = None

    def columns(self) -> dict[str, np.ndarray]:
        return {"cluster": self.cluster}


def _weights(system: System) -> np.ndarray:
    """The mass of each atom in amu: the mass column, else that of its
    element, else 1."""
    table = system.table
    if "mass" in table.columns:
        return table["mass"].to_numpy().astype(np.float64)
    if "element" in table.columns:
        return atomic_masses(table["element"].to_physical().to_numpy())
    return np.ones(len(system))


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
    unwrap : bool, optional
        Give the positions with each cluster unbroken by the periodic
        boundaries. Default False.
    center_of_mass : bool, optional
        Compute the mass and the center of mass of each cluster. Default
        False.
    gyration : bool, optional
        Compute the gyration tensor and the radius of gyration of each
        cluster. Default False.
    """

    def __init__(
        self,
        cutoff: float | None = None,
        *,
        cutoffs: Mapping[tuple[str, str], float] | None = None,
        unwrap: bool = False,
        center_of_mass: bool = False,
        gyration: bool = False,
    ):
        self.__bonds = _Bonds(cutoff, cutoffs)
        self.unwrap = bool(unwrap)
        self.center_of_mass = bool(center_of_mass)
        self.gyration = bool(gyration)

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
            *grid, _ = self.__bonds._grid(system)
            cluster, sizes, images, spanning = _native.clusters_search(
                *grid, progress, _native_token(cancel)
            )
        else:
            nl, cutoff, type, limits, _ = self.__bonds.listed(system, neighbors)
            with nl._reading():
                cluster, sizes, images, spanning = _native.clusters(
                    *_view(system, nl), cutoff, type, limits
                )
        found = {"cluster": cluster, "sizes": sizes, "spanning": spanning.astype(bool)}
        if not (self.unwrap or self.center_of_mass or self.gyration):
            return ClusterResult(**found)
        n, count = len(system), len(sizes)
        unwrapped = system["positions"] + images.reshape(n, 3) @ system.box.matrix
        if self.unwrap:
            found["unwrapped"] = unwrapped
        if not (self.center_of_mass or self.gyration):
            return ClusterResult(**found)
        weights = _weights(system)
        masses = np.bincount(cluster, weights=weights, minlength=count)
        center = (
            np.stack(
                [
                    np.bincount(
                        cluster, weights=weights * unwrapped[:, e], minlength=count
                    )
                    for e in range(3)
                ],
                axis=1,
            )
            / masses[:, None]
        )
        if self.center_of_mass:
            found.update(masses=masses, center_of_mass=center)
        if self.gyration:
            delta = unwrapped - center[cluster]
            gyration = np.empty((count, 3, 3))
            for a in range(3):
                for b in range(a, 3):
                    gyration[:, a, b] = gyration[:, b, a] = (
                        np.bincount(
                            cluster,
                            weights=weights * delta[:, a] * delta[:, b],
                            minlength=count,
                        )
                        / masses
                    )
            found.update(
                radius_of_gyration=np.sqrt(np.trace(gyration, axis1=1, axis2=2)),
                gyration_tensor=gyration,
            )
        return ClusterResult(**found)

    def __repr__(self) -> str:
        return f"ClusterAnalysis({self.__bonds})"


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
        self.__bonds = _Bonds(cutoff, cutoffs)
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
            *grid, found = self.__bonds._grid(system)
            counts = _native.bond_angles_search(
                *grid, self.bins, progress, _native_token(cancel)
            )
            names = found[0] if found is not None else ()
            types = max(len(names), 1)
        else:
            nl, cutoff, _, limits, found = self.__bonds.listed(system, neighbors)
            names, index = (
                found if found is not None else ((), np.zeros(len(system), int))
            )
            types = max(len(names), 1)
            with nl._reading():
                counts = _native.bond_angles(
                    *_view(system, nl),
                    cutoff,
                    index.astype(np.int32),
                    types,
                    self.bins,
                    limits,
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
        return f"BondAngleDistribution({self.__bonds}, bins={self.bins})"


@dataclass(frozen=True)
class WarrenCowleyResult(Result):
    """Warren-Cowley short-range order parameters.

    Attributes
    ----------
    elements : tuple of str
        The elements, by atomic number.
    alpha : numpy.ndarray
        (elements, elements) parameter of each pair: row a, column b is
        1 - p(b | a) / c_b; (shells, elements, elements) for shells of
        neighbors.
    """

    elements: tuple[str, ...]
    alpha: np.ndarray

    def __getitem__(self, key: tuple) -> Any:
        """The parameter of a pair of elements (a, b), an array over the
        shells for shells of neighbors, or that of one shell (a, b, shell)."""
        a, b, *shell = key
        try:
            pair = (self.elements.index(a), self.elements.index(b))
        except ValueError:
            raise InvalidParameter(f"no parameter for {(a, b)}") from None
        if self.alpha.ndim == 2:
            if shell:
                raise InvalidParameter("the parameters are of one shell")
            return float(self.alpha[pair])
        values = self.alpha[(slice(None), *pair)]
        return values.copy() if not shell else float(values[shell[0]])


class WarrenCowley:
    """Warren-Cowley parameters of chemical short-range order.

    Parameters
    ----------
    cutoff : float or sequence of float, optional
        Neighbors closer than cutoff, in Å; increasing cutoffs make shells
        of neighbors, each from the cutoff before it to its own.
    num_neighbors : int or sequence of int, optional
        The num_neighbors nearest neighbors; a sequence makes shells of the
        nearest neighbors, the first so many nearest, then the next so many.
        Give ``cutoff`` or ``num_neighbors``.
    """

    def __init__(
        self,
        cutoff: float | Sequence[float] | None = None,
        *,
        num_neighbors: int | Sequence[int] | None = None,
    ):
        if (cutoff is None) == (num_neighbors is None):
            raise InvalidParameter("give either cutoff or num_neighbors")
        self.shells = None
        if cutoff is not None:
            values = np.atleast_1d(np.asarray(cutoff, dtype=float))
            if values.ndim != 1 or not (values > 0).all():
                raise InvalidParameter(f"cutoff must be positive, not {cutoff!r}")
            if (np.diff(values) <= 0).any():
                raise InvalidParameter("the cutoffs of shells must increase")
            if np.ndim(cutoff):
                self.shells = values
            cutoff = float(values[-1])
        if num_neighbors is not None:
            counts = np.atleast_1d(np.asarray(num_neighbors))
            if counts.ndim != 1 or not np.issubdtype(counts.dtype, np.integer):
                raise InvalidParameter("num_neighbors must be integers")
            if (counts < 1).any():
                raise InvalidParameter("num_neighbors must be at least 1")
            if np.ndim(num_neighbors):
                self.shells = np.cumsum(counts)
            num_neighbors = int(counts.sum())
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
        if self.cutoff is not None and self.shells is None:
            # each atom's neighbors by element, summed by the atom's element
            table, _ = _Bonds(self.cutoff, None).counts(
                system, neighbors, progress, cancel
            )
            pairs = np.zeros((types, types), dtype=np.int64)
            np.add.at(pairs, index, table)
            pairs = pairs[None]
        else:
            if self.cutoff is not None:
                nl = _given_or_built(system, self.cutoff, neighbors)
            else:
                nl = _nearest_given_or_built(system, self.num_neighbors, neighbors)
            with nl._reading():
                if self.cutoff is not None:
                    kept = nl.distances < self.cutoff
                    shell = np.searchsorted(
                        self.shells, nl.distances[kept], side="right"
                    )
                else:
                    # the rank of each pair among the nearest of its atom
                    rank = np.arange(len(nl.j)) - np.repeat(nl.offsets[:-1], nl.counts)
                    kept = rank < self.num_neighbors
                    shell = np.zeros(int(kept.sum()), dtype=np.int64)
                    if self.shells is not None:
                        shell = np.searchsorted(self.shells, rank[kept], side="right")
                first, second = index[nl.i[kept]], index[nl.j[kept]]
            count = 1 if self.shells is None else len(self.shells)
            pairs = np.bincount(
                (shell * types + first) * types + second,
                minlength=count * types * types,
            ).reshape(count, types, types)
        fractions = np.bincount(index, minlength=types) / len(system)
        with np.errstate(invalid="ignore", divide="ignore"):
            probability = pairs / pairs.sum(axis=2, keepdims=True)
        alpha = 1.0 - probability / fractions[None, None, :]
        return WarrenCowleyResult(names, alpha if self.shells is not None else alpha[0])

    def __repr__(self) -> str:
        if self.cutoff is not None:
            value = self.cutoff if self.shells is None else self.shells.tolist()
            return f"WarrenCowley(cutoff={value!r})"
        if self.shells is not None:
            sizes = np.diff(np.concatenate([[0], self.shells])).tolist()
            return f"WarrenCowley(num_neighbors={sizes})"
        return f"WarrenCowley(num_neighbors={self.num_neighbors})"
