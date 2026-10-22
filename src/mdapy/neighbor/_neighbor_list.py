# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Neighbor lists: pairs of atoms within a cutoff, the nearest neighbors of
each atom, or the atoms that share a Voronoi face, including periodic
images."""

import math
from typing import Any

import numpy as np

from mdapy._core import analysis as _analysis
from mdapy._core import neighbor as _native
from mdapy.core.indices import index_dtype
from mdapy.core.system import System
from mdapy.core.threads import _native_token
from mdapy.errors import InvalidParameter


def _run(kernel: Any, *args: Any) -> Any:
    """Calls a compiled kernel, its overflow errors as InvalidParameter."""
    try:
        return kernel(*args)
    except OverflowError as error:
        raise InvalidParameter(str(error)) from None


def _voronoi_thresholds(
    face_threshold: float, relative_face_threshold: float, edge_threshold: float
) -> tuple[float, float, float]:
    """The thresholds of the faces of Voronoi cells that count, checked."""
    if not face_threshold >= 0.0:
        raise InvalidParameter(
            f"face_threshold must not be negative, not {face_threshold!r}"
        )
    if not 0.0 <= relative_face_threshold < 1.0:
        raise InvalidParameter(
            "relative_face_threshold must be from 0 to below 1, "
            f"not {relative_face_threshold!r}"
        )
    if not edge_threshold >= 0.0:
        raise InvalidParameter(
            f"edge_threshold must not be negative, not {edge_threshold!r}"
        )
    return float(face_threshold), float(relative_face_threshold), float(edge_threshold)


class NeighborList:
    """Pairs of atoms within a cutoff, the nearest neighbors of each atom, or
    the atoms that share a Voronoi face, including periodic images.

    Build one with :meth:`build`. Pairs are grouped by the first atom: the
    pairs of atom ``i`` are ``offsets[i]:offsets[i + 1]``. The vector from
    atom ``i`` to the image of atom ``j`` is
    ``x[j] - x[i] + shifts @ box.matrix``.

    Attributes
    ----------
    system : System
        The system the list was built for.
    cutoff : float
        Every pair is closer than it, in Å.
    num_neighbors : int or None
        The number of pairs of each atom in a list of nearest neighbors,
        None otherwise.
    voronoi : bool
        Whether the pairs are the atoms that share a Voronoi face.
    offsets : numpy.ndarray
        (N + 1,) int64.
    j : numpy.ndarray
        (P,) :data:`mdapy.index_dtype`, the second atom of each pair.
    shifts : numpy.ndarray
        (P, 3) int8 periodic image of the second atom.
    distances : numpy.ndarray
        (P,) float64 distances in Å.
    areas : numpy.ndarray or None
        (P,) float64 area in Å² of the Voronoi face of each pair in a list of
        Voronoi neighbors, None otherwise.
    """

    def __init__(
        self,
        system: System,
        cutoff: float,
        offsets: np.ndarray,
        j: np.ndarray,
        shifts: np.ndarray,
        distances: np.ndarray,
        num_neighbors: int | None = None,
        areas: np.ndarray | None = None,
    ):
        self.system = system
        self.cutoff = float(cutoff)
        self.num_neighbors = num_neighbors
        self.voronoi = areas is not None
        self.areas = areas
        self.offsets = offsets
        self.j = j
        self.shifts = shifts
        self.distances = distances
        for array in (offsets, j, shifts, distances, areas):
            if array is not None:
                array.flags.writeable = False
        self._i: np.ndarray | None = None
        self._vectors: np.ndarray | None = None

    @classmethod
    def build(
        cls,
        system: System,
        cutoff: float | None = None,
        *,
        num_neighbors: int | None = None,
        voronoi: bool = False,
        face_threshold: float = 0.0,
        relative_face_threshold: float = 0.0,
        edge_threshold: float = 0.0,
        progress: Any = None,
        cancel: Any = None,
    ) -> "NeighborList":
        """All pairs of atoms closer than a cutoff, the nearest neighbors of
        each atom, or the atoms that share a Voronoi face.

        Parameters
        ----------
        system : System
            The atoms and the box. Any box works; a box thinner than the
            cutoff gives several images of the same neighbor, and images of
            the atom itself.
        cutoff : float, optional
            Cutoff radius in Å, positive.
        num_neighbors : int, optional
            Number of nearest neighbors of each atom, positive.
        voronoi : bool, optional
            The neighbors of an atom are the atoms across the faces of its
            Voronoi cell that count. Give one of cutoff, num_neighbors and
            voronoi.
        face_threshold, relative_face_threshold, edge_threshold : float, optional
            With voronoi, the faces that count, as in
            :class:`mdapy.analysis.VoronoiAnalysis`. Default 0.
        progress : callable, optional
            Called as ``progress(done, total)`` while the list is built.
        cancel : CancelToken, optional
            Stops the search.

        Returns
        -------
        NeighborList
            With a cutoff, every pair closer than it, grouped by the first
            atom. With num_neighbors, that many pairs per atom, nearest first
            (ties by the second atom, then the image). With voronoi, a pair
            for each face that counts of the cell of the first atom, with its
            area; the walls of the box are left out. Lists of nearest and of
            Voronoi neighbors have as cutoff the largest distance, slightly
            enlarged.

        Raises
        ------
        InvalidParameter
            For an invalid cutoff, num_neighbors or threshold, when a system
            without periodic directions has num_neighbors atoms or fewer,
            when a periodic image is more than 127 boxes away, or, with
            voronoi, for two atoms at the same position.
        """
        modes = (cutoff is not None) + (num_neighbors is not None) + bool(voronoi)
        if modes != 1:
            raise InvalidParameter("give one of cutoff, num_neighbors and voronoi")
        thresholds = (face_threshold, relative_face_threshold, edge_threshold)
        if voronoi:
            return cls._voronoi(
                system, _voronoi_thresholds(*thresholds), progress, cancel
            )
        if any(thresholds):
            raise InvalidParameter("the face and edge thresholds need voronoi")
        box = system.box
        arrays = (
            np.ascontiguousarray(system.positions),
            np.ascontiguousarray(box.matrix),
            np.ascontiguousarray(box.origin),
            np.ascontiguousarray(box.pbc),
        )
        if num_neighbors is None:
            if not cutoff > 0 or not math.isfinite(cutoff):
                raise InvalidParameter(
                    f"cutoff must be a positive number, not {cutoff!r}"
                )
            offsets, j, shifts, distances = _run(
                _native.build,
                *arrays,
                float(cutoff),
                progress,
                _native_token(cancel),
            )
            return cls(system, cutoff, offsets, j, shifts, distances)

        if (
            isinstance(num_neighbors, bool)
            or not isinstance(num_neighbors, int | np.integer)
            or num_neighbors < 1
        ):
            raise InvalidParameter(
                f"num_neighbors must be a positive integer, not {num_neighbors!r}"
            )
        k, n = int(num_neighbors), len(system)
        if n > 0 and not box.pbc.any() and n - 1 < k:
            raise InvalidParameter(
                f"num_neighbors is {k}, but the system has only {n} atoms "
                "and no periodic images"
            )
        offsets, j, shifts, distances = _run(
            _native.nearest, *arrays, k, progress, _native_token(cancel)
        )
        reach = float(np.nextafter(distances.max(), np.inf)) if n else 0.0
        return cls(system, reach, offsets, j, shifts, distances, num_neighbors=k)

    @classmethod
    def _voronoi(
        cls,
        system: System,
        thresholds: tuple[float, float, float],
        progress: Any,
        cancel: Any,
    ) -> "NeighborList":
        """The atoms across the faces that count of the Voronoi cells."""
        box = system.box
        positions = np.ascontiguousarray(system.positions)
        try:
            *_, offsets, j, shifts, areas, _, _ = _analysis.voronoi(
                positions,
                np.ascontiguousarray(box.matrix),
                np.ascontiguousarray(box.origin),
                np.ascontiguousarray(box.pbc),
                *thresholds,
                False,
                progress,
                _native_token(cancel),
            )
        except (ValueError, OverflowError) as error:
            raise InvalidParameter(str(error)) from None
        atoms = j >= 0  # not a wall
        kept = np.concatenate([[0], np.cumsum(atoms)])
        offsets = kept[offsets].astype(np.int64)
        j, shifts = j[atoms], shifts.reshape(-1, 3)[atoms]
        vectors = _native.pair_vectors(
            positions, offsets, j, shifts, np.ascontiguousarray(box.matrix)
        )
        distances = np.sqrt((vectors * vectors).sum(axis=1))
        reach = float(np.nextafter(distances.max(), np.inf)) if len(j) else 0.0
        voronoi = cls(system, reach, offsets, j, shifts, distances, areas=areas[atoms])
        vectors.flags.writeable = False
        voronoi._vectors = vectors
        return voronoi

    @property
    def counts(self) -> np.ndarray:
        """(N,) int64 number of pairs of each atom."""
        return np.diff(self.offsets)

    @property
    def i(self) -> np.ndarray:
        """(P,) :data:`mdapy.index_dtype` first atom of each pair."""
        if self._i is None:
            self._i = np.repeat(
                np.arange(len(self.counts), dtype=index_dtype), self.counts
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

    def _cut(
        self, cutoff: float, limits: np.ndarray, types: np.ndarray | None = None
    ) -> "NeighborList":
        """The pairs closer than their limit, in their order: limits[0, 0]
        for every pair, or limits[types[i], types[j]] with types."""
        offsets, j, shifts, distances = _native.within(
            self.offsets,
            self.j,
            np.ascontiguousarray(self.shifts),
            self.distances,
            None if types is None else np.ascontiguousarray(types, dtype=np.int32),
            np.ascontiguousarray(limits, dtype=np.float64),
        )
        return NeighborList(self.system, cutoff, offsets, j, shifts, distances)

    def within(self, cutoff: float) -> "NeighborList":
        """The pairs closer than a smaller cutoff, without a new search.

        Parameters
        ----------
        cutoff : float
            Cutoff radius in Å, positive and at most the cutoff of this list.

        Returns
        -------
        NeighborList
            The pairs closer than cutoff, in their order; this list itself
            when cutoff is its own.
        """
        if self.voronoi:
            raise InvalidParameter("a list of Voronoi neighbors has no pairs to cut")
        if not 0 < cutoff <= self.cutoff:
            raise InvalidParameter(
                f"cutoff must be positive and at most {self.cutoff}, not {cutoff!r}"
            )
        if cutoff == self.cutoff:
            return self  # every pair is closer already
        return self._cut(cutoff, np.array([[cutoff]]))

    def _order(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        # the first k pairs of each atom (all when k < 0) in the order of a
        # search of nearest neighbors, and their offsets
        return _native.order_by_distance(
            np.ascontiguousarray(self.system.positions),
            self.offsets,
            self.j,
            self.shifts,
            np.ascontiguousarray(self.system.box.matrix),
            k,
        )

    def _nearest(self, k: int) -> "NeighborList":
        """The k nearest pairs of each atom, nearest first with ties by atom
        and image, or all of an atom with fewer."""
        order, offsets = self._order(k)
        return NeighborList(
            self.system,
            self.cutoff,
            offsets,
            self.j[order],
            self.shifts[order],
            self.distances[order],
        )

    def _follow(self, system: System, cutoff: float) -> "NeighborList | None":
        """The pairs of NeighborList.build(system, cutoff), in another order,
        for the atoms of this list moved to their places in system, from the
        pairs of this list without a new search; None for a list of nearest
        neighbors, when the box changed or when an atom moved too far, half
        the difference of the cutoffs or more."""
        old = self.system
        if (
            self.num_neighbors is not None
            or self.voronoi
            or len(system) != len(old)
            or not np.array_equal(system.box.matrix, old.box.matrix)
            or not np.array_equal(system.box.origin, old.box.origin)
            or not np.array_equal(system.box.pbc, old.box.pbc)
        ):
            return None
        step = system.positions - old.positions
        moved = math.sqrt(float((step * step).sum(axis=1).max(initial=0.0)))
        if not cutoff + 2.0 * moved < self.cutoff:
            return None
        offsets, j, shifts, distances, vectors = _native.follow(
            np.ascontiguousarray(system.positions),
            self.offsets,
            self.j,
            self.shifts,
            np.ascontiguousarray(system.box.matrix),
            float(cutoff),
        )
        followed = NeighborList(system, cutoff, offsets, j, shifts, distances)
        vectors.flags.writeable = False
        followed._vectors = vectors
        return followed

    def sorted(self) -> "NeighborList":
        """The same pairs, each atom's sorted by distance, ties by atom and
        then image as in a list of nearest neighbors.

        Returns
        -------
        NeighborList
            A list with the pairs of each atom nearest first.
        """
        order, _ = self._order(-1)
        return NeighborList(
            self.system,
            self.cutoff,
            self.offsets,
            self.j[order],
            self.shifts[order],
            self.distances[order],
            self.num_neighbors,
            None if self.areas is None else self.areas[order],
        )

    def __repr__(self) -> str:
        size = f"NeighborList(atoms={len(self.counts)}, pairs={len(self)}"
        if self.num_neighbors is not None:
            return f"{size}, num_neighbors={self.num_neighbors})"
        if self.voronoi:
            return f"{size}, voronoi=True)"
        return f"{size}, cutoff={self.cutoff!r})"


def _given_or_built(
    system: System, cutoff: float, neighbors: "NeighborList | None"
) -> NeighborList:
    """The given neighbor list after checking that it holds every pair of
    system within cutoff, or a new one."""
    if neighbors is None:
        return NeighborList.build(system, cutoff)
    if neighbors.system is not system:
        raise InvalidParameter("the neighbor list was built for another system")
    if neighbors.num_neighbors is not None or neighbors.voronoi:
        kind = "nearest" if neighbors.num_neighbors is not None else "Voronoi"
        raise InvalidParameter(
            f"a list of {kind} neighbors does not hold every pair within a cutoff; "
            "build the list with a cutoff"
        )
    if neighbors.cutoff < cutoff:
        raise InvalidParameter(
            f"the neighbor list has cutoff {neighbors.cutoff}, below the required {cutoff}"
        )
    return neighbors


def _nearest_given_or_built(
    system: System, k: int, neighbors: "NeighborList | None"
) -> NeighborList:
    """The k nearest neighbors of each atom, nearest first with ties by atom
    and image, from the given list after checking that it holds them, or
    from a new search."""
    if neighbors is None:
        return NeighborList.build(system, num_neighbors=k)
    if neighbors.system is not system:
        raise InvalidParameter("the neighbor list was built for another system")
    if neighbors.voronoi:
        raise InvalidParameter(
            "a list of Voronoi neighbors does not hold the nearest neighbors; "
            "build the list with a cutoff or num_neighbors"
        )
    if neighbors.num_neighbors == k:
        return neighbors  # in the order of a search of nearest neighbors already
    fewest = int(neighbors.counts.min(initial=k))
    if fewest < k:
        raise InvalidParameter(
            f"an atom has {fewest} neighbors in the list, fewer than {k}"
        )
    return neighbors._nearest(k)


def _voronoi_given_or_built(
    system: System, neighbors: "NeighborList | None"
) -> NeighborList:
    """The given list of Voronoi neighbors after checking it, or a new one."""
    if neighbors is None:
        return NeighborList.build(system, voronoi=True)
    if neighbors.system is not system:
        raise InvalidParameter("the neighbor list was built for another system")
    if not neighbors.voronoi:
        raise InvalidParameter("the neighbor list does not hold Voronoi neighbors")
    return neighbors
