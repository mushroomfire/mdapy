# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Neighbor lists: pairs of atoms within a cutoff, including periodic images."""

import math
from typing import Any

import numpy as np

from mdapy._core import neighbor as _native
from mdapy.core.system import System
from mdapy.core.threads import _native_token
from mdapy.errors import InvalidParameter


class NeighborList:
    """Pairs of atoms within a cutoff, including periodic images.

    Build one with :meth:`build` or :meth:`nearest`. Pairs are grouped by the
    first atom: the pairs of atom ``i`` are ``offsets[i]:offsets[i + 1]``.
    The vector from atom ``i`` to the image of atom ``j`` is
    ``x[j] - x[i] + shifts @ box.matrix``. Every pair appears in both
    directions.

    Attributes
    ----------
    system : System
        The system the list was built for.
    cutoff : float
        Cutoff radius in Å; every pair is closer than it.
    offsets : numpy.ndarray
        (N + 1,) int64.
    j : numpy.ndarray
        (P,) int32, the second atom of each pair.
    shifts : numpy.ndarray
        (P, 3) int32 periodic image of the second atom.
    distances : numpy.ndarray
        (P,) float64 distances in Å.
    """

    def __init__(
        self,
        system: System,
        cutoff: float,
        offsets: np.ndarray,
        j: np.ndarray,
        shifts: np.ndarray,
        distances: np.ndarray,
    ):
        self.system = system
        self.cutoff = float(cutoff)
        self.offsets = offsets
        self.j = j
        self.shifts = shifts
        self.distances = distances
        for array in (offsets, j, shifts, distances):
            array.flags.writeable = False
        self._i: np.ndarray | None = None
        self._vectors: np.ndarray | None = None

    @classmethod
    def build(
        cls,
        system: System,
        cutoff: float,
        *,
        progress: Any = None,
        cancel: Any = None,
    ) -> "NeighborList":
        """All pairs of atoms closer than cutoff.

        Parameters
        ----------
        system : System
            The atoms and the box. Any box works; a cutoff larger than the box
            gives several images of the same neighbor, and images of the atom
            itself.
        cutoff : float
            Cutoff radius in Å, positive.
        progress : callable, optional
            Called as ``progress(done, total)`` while the list is built.
        cancel : CancelToken, optional
            Stops the search.

        Returns
        -------
        NeighborList
            The pairs, sorted by the first atom, then the second, then the
            image.
        """
        if not cutoff > 0 or not math.isfinite(cutoff):
            raise InvalidParameter(f"cutoff must be a positive number, not {cutoff!r}")
        box = system.box
        offsets, j, shifts, distances = _native.build(
            np.ascontiguousarray(system.positions),
            np.ascontiguousarray(box.matrix),
            np.ascontiguousarray(box.origin),
            np.ascontiguousarray(box.pbc),
            float(cutoff),
            progress,
            _native_token(cancel),
        )
        return cls(system, cutoff, offsets, j, shifts, distances)

    @classmethod
    def nearest(cls, system: System, num_neighbors: int) -> "NeighborList":
        """The num_neighbors nearest neighbors of every atom, nearest first.

        Parameters
        ----------
        system : System
            The atoms and the box.
        num_neighbors : int
            Number of neighbors of each atom, positive.

        Returns
        -------
        NeighborList
            Exactly num_neighbors pairs per atom, sorted by distance (then by
            the second atom and the image). Its cutoff is the largest of
            these distances, slightly enlarged.

        Raises
        ------
        InvalidParameter
            If an atom of a non-periodic system has fewer neighbors.
        """
        k = int(num_neighbors)
        if k < 1:
            raise InvalidParameter("num_neighbors must be at least 1")
        n = len(system)
        if n == 0:
            empty = np.zeros(0)
            return cls(
                system,
                0.0,
                np.zeros(1, np.int64),
                empty.astype(np.int32),
                np.zeros((0, 3), np.int32),
                empty,
            )
        periodic = bool(system.box.pbc.any())
        if not periodic and n - 1 < k:
            raise InvalidParameter(
                f"num_neighbors is {k}, but the system has only {n} atoms and no periodic images"
            )
        box = system.box
        offsets, j, shifts, distances = _native.nearest(
            np.ascontiguousarray(system.positions),
            np.ascontiguousarray(box.matrix),
            np.ascontiguousarray(box.origin),
            np.ascontiguousarray(box.pbc),
            k,
        )
        return cls(
            system,
            float(np.nextafter(distances.max(), np.inf)),
            offsets,
            j,
            shifts,
            distances,
        )

    @property
    def counts(self) -> np.ndarray:
        """(N,) int64 number of pairs of each atom."""
        return np.diff(self.offsets)

    @property
    def i(self) -> np.ndarray:
        """(P,) int32 first atom of each pair."""
        if self._i is None:
            self._i = np.repeat(
                np.arange(len(self.counts), dtype=np.int32), self.counts
            )
            self._i.flags.writeable = False
        return self._i

    @property
    def vectors(self) -> np.ndarray:
        """(P, 3) float64 vectors from the first atom to the second, in Å."""
        if self._vectors is None:
            self._vectors = _native.pair_vectors(
                np.ascontiguousarray(self.system.positions),
                self.offsets,
                self.j,
                self.shifts,
                np.ascontiguousarray(self.system.box.matrix),
            )
            self._vectors.flags.writeable = False
        return self._vectors

    def __len__(self) -> int:
        return len(self.j)

    def _subset(self, mask: np.ndarray, cutoff: float) -> "NeighborList":
        kept = np.concatenate([[0], np.cumsum(mask)])
        return NeighborList(
            self.system,
            cutoff,
            kept[self.offsets].astype(np.int64),
            self.j[mask],
            self.shifts[mask],
            self.distances[mask],
        )

    def within(self, cutoff: float) -> "NeighborList":
        """The pairs closer than a smaller cutoff, without a new search."""
        if not 0 < cutoff <= self.cutoff:
            raise InvalidParameter(
                f"cutoff must be positive and at most {self.cutoff}, not {cutoff!r}"
            )
        return self._subset(self.distances < cutoff, cutoff)

    def _nearest(self, k: int) -> "NeighborList":
        """The k nearest pairs of each atom, nearest first (ties keep their
        order), or all of an atom with fewer."""
        order, offsets = _native.order_by_distance(self.offsets, self.distances, k)
        return NeighborList(
            self.system,
            self.cutoff,
            offsets,
            self.j[order],
            self.shifts[order],
            self.distances[order],
        )

    def sorted(self) -> "NeighborList":
        """The same pairs, each atom's sorted by distance (ties keep their order)."""
        order, _ = _native.order_by_distance(self.offsets, self.distances)
        return NeighborList(
            self.system,
            self.cutoff,
            self.offsets,
            self.j[order],
            self.shifts[order],
            self.distances[order],
        )

    def __repr__(self) -> str:
        return (
            f"NeighborList(atoms={len(self.counts)}, pairs={len(self)}, "
            f"cutoff={self.cutoff!r})"
        )


def _given_or_built(
    system: System, cutoff: float, neighbors: "NeighborList | None"
) -> NeighborList:
    """The given neighbor list after checking that it fits, or a new one."""
    if neighbors is None:
        return NeighborList.build(system, cutoff)
    if neighbors.system is not system:
        raise InvalidParameter("the neighbor list was built for another system")
    if neighbors.cutoff < cutoff:
        raise InvalidParameter(
            f"the neighbor list has cutoff {neighbors.cutoff}, below the required {cutoff}"
        )
    return neighbors
