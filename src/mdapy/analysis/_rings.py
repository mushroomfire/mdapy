# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Rings of bonds: closed paths without shortcuts."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from mdapy._core import analysis as _native
from mdapy.analysis._bonds import _Bonds
from mdapy.core.mesh import TriangleMesh
from mdapy.core.result import Result
from mdapy.core.system import System
from mdapy.core.threads import _native_token
from mdapy.errors import InvalidParameter, MissingProperty

# the largest ring size
_LARGEST = 36


@dataclass(frozen=True)
class RingResult(Result):
    """The rings of a network of bonds.

    Attributes
    ----------
    atoms : numpy.ndarray
        (M, max_size) int64 the atoms of each ring in order around it, from
        its atom of the smallest index on toward the smaller of its two
        neighbors; -1 after the last.
    shifts : numpy.ndarray
        (M, max_size, 3) int32 the periodic image of each atom of a ring
        relative to its first atom.
    sizes : numpy.ndarray
        (M,) int64 the number of atoms of each ring.
    counts : numpy.ndarray
        (max_size + 1,) int64 the number of rings of each size.
    per_atom : numpy.ndarray
        (N,) int64 the number of rings each atom is in.
    """

    atoms: np.ndarray
    shifts: np.ndarray
    sizes: np.ndarray
    counts: np.ndarray
    per_atom: np.ndarray

    def columns(self) -> dict[str, np.ndarray]:
        return {"rings": self.per_atom}

    def polygons(self, system: System) -> tuple[TriangleMesh, np.ndarray]:
        """The rings as polygons, for pictures.

        Parameters
        ----------
        system : System
            The system of the rings.

        Returns
        -------
        mesh : TriangleMesh
            A fan of triangles about the center of each ring, across the
            periodic boundaries of the box.
        ring : numpy.ndarray
            (F,) the ring of each triangle, for colors such as by its size.
        """
        box = system.box
        positions = system["positions"]
        rings = np.flatnonzero(self.sizes)
        size = self.sizes
        corners = []
        owner = []
        for k in rings:
            n = int(size[k])
            atoms = self.atoms[k, :n]
            points = positions[atoms] + self.shifts[k, :n] @ box.matrix
            center = points.mean(axis=0)
            nxt = np.roll(points, -1, axis=0)
            corners.append(
                np.stack([np.broadcast_to(center, points.shape), points, nxt], 1)
            )
            owner.append(np.full(n, k))
        if not corners:
            empty = TriangleMesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
            return empty, np.zeros(0, dtype=np.int64)
        triangles = np.concatenate(corners)
        # each triangle moved whole into the box by its first corner, the
        # center of its ring
        fractional = box.to_fractional(triangles[:, 0])
        move = np.where(box.pbc, np.floor(fractional), 0.0)
        triangles = triangles - (move @ box.matrix)[:, None, :]
        mesh = TriangleMesh(
            triangles.reshape(-1, 3),
            np.arange(3 * len(triangles)).reshape(-1, 3),
            np.zeros((len(triangles), 3, 3), dtype=np.int64),
            box,
        )
        return mesh, np.concatenate(owner)


class RingStatistics:
    """The rings of a network of bonds: closed paths through distinct atoms
    in which no two atoms are closer through the network than along the
    ring (shortest-path rings).

    Parameters
    ----------
    max_size : int, optional
        The largest ring, at most 36. Default 8.
    min_size : int, optional
        The smallest ring, at least 3. Default 3.
    cutoff : float, optional
        Atoms closer than it, in Å, are bonded.
    cutoffs : dict, optional
        The bond length in Å of each pair of elements ``(a, b)``, in either
        order; pairs not given are never bonded. Without cutoff or cutoffs,
        the bonds of the topology of the system.
    """

    def __init__(
        self,
        max_size: int = 8,
        *,
        min_size: int = 3,
        cutoff: float | None = None,
        cutoffs: Mapping[tuple[str, str], float] | None = None,
    ):
        for name, value in (("max_size", max_size), ("min_size", min_size)):
            if isinstance(value, bool) or int(value) != value:
                raise InvalidParameter(f"{name} must be an integer, not {value!r}")
        if not 3 <= min_size <= max_size <= _LARGEST:
            raise InvalidParameter(
                f"the ring sizes must satisfy 3 <= min_size <= max_size <= {_LARGEST}, "
                f"not {min_size} and {max_size}"
            )
        self.max_size = int(max_size)
        self.min_size = int(min_size)
        self.__bonds = (
            None if cutoff is None and cutoffs is None else _Bonds(cutoff, cutoffs)
        )

    def __adjacency(self, system: System) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The bonds of each atom, both ways: offsets, the other atoms and
        their periodic images."""
        n = len(system)
        if self.__bonds is None:
            topology = system.topology
            if topology is None or not len(topology.bonds):
                raise MissingProperty(
                    "the system has no bonds: give a cutoff, cutoffs or a topology"
                )
            pairs = topology.bonds["atoms"].to_numpy().astype(np.int64)
            box = system.box
            step = box.to_fractional(
                system["positions"][pairs[:, 1]]
            ) - box.to_fractional(system["positions"][pairs[:, 0]])
            image = -np.where(box.pbc, np.round(step), 0.0).astype(np.int32)
            i = np.concatenate([pairs[:, 0], pairs[:, 1]])
            j = np.concatenate([pairs[:, 1], pairs[:, 0]])
            shifts = np.concatenate([image, -image])
        else:
            nl, cutoff, types, limits, _ = self.__bonds.listed(system, None)
            i = np.repeat(np.arange(n), np.diff(nl.offsets))
            j = nl.j.astype(np.int64)
            shifts = nl.shifts.astype(np.int32)
            if types is not None:
                keep = nl.distances < limits[types[i], types[j]]
                i, j, shifts = i[keep], j[keep], shifts[keep]
            elif cutoff:
                keep = nl.distances < cutoff
                i, j, shifts = i[keep], j[keep], shifts[keep]
        order = np.lexsort((j, i))
        i, j, shifts = i[order], j[order], shifts[order]
        offsets = np.concatenate([[0], np.cumsum(np.bincount(i, minlength=n))])
        return offsets.astype(np.int64), j, np.ascontiguousarray(shifts)

    def compute(
        self, system: System, *, progress: Any = None, cancel: Any = None
    ) -> RingResult:
        """The rings of a system.

        Parameters
        ----------
        system : System
            The atoms, with an element column for cutoffs per pair of
            elements, or bonds in their topology without a cutoff.

        Returns
        -------
        RingResult
            The rings, their sizes and how many each atom is in.

        Raises
        ------
        MissingProperty
            Without a cutoff, for a system without bonds in its topology;
            with cutoffs per pair of elements, without an element column.
        """
        offsets, j, shifts = self.__adjacency(system)
        atoms, images, starts = _native.find_rings(
            offsets,
            np.ascontiguousarray(j),
            shifts,
            self.min_size,
            self.max_size,
            progress,
            _native_token(cancel),
        )
        sizes = np.diff(starts)
        count = len(sizes)
        padded = np.full((count, self.max_size), -1, dtype=np.int64)
        padded_shifts = np.zeros((count, self.max_size, 3), dtype=np.int32)
        column = np.arange(len(atoms)) - np.repeat(starts[:-1], sizes)
        row = np.repeat(np.arange(count), sizes)
        padded[row, column] = atoms
        padded_shifts[row, column] = images.reshape(-1, 3)
        return RingResult(
            atoms=padded,
            shifts=padded_shifts,
            sizes=sizes,
            counts=np.bincount(sizes, minlength=self.max_size + 1),
            per_atom=np.bincount(atoms, minlength=len(system)),
        )

    def __repr__(self) -> str:
        bonds = "" if self.__bonds is None else f", {self.__bonds!r}"
        return f"RingStatistics({self.max_size}, min_size={self.min_size}{bonds})"
