# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Voronoi cells of atoms."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from mdapy._core import analysis as _native
from mdapy.core.result import Result
from mdapy.core.system import System
from mdapy.core.threads import _native_token
from mdapy.errors import InvalidParameter
from mdapy.neighbor import NeighborList
from mdapy.neighbor._neighbor_list import _given_or_built


@dataclass(frozen=True)
class VoronoiResult(Result):
    """Voronoi cells of the atoms.

    Attributes
    ----------
    volume : numpy.ndarray
        (N,) volume of each cell in Å³.
    coordination : numpy.ndarray
        (N,) int64 number of faces shared with other atoms.
    radius : numpy.ndarray
        (N,) distance from each atom to the farthest vertex of its cell, in Å.
    offsets : numpy.ndarray
        (N + 1,) int64: the faces of atom ``i`` are ``offsets[i]:offsets[i + 1]``.
    neighbors : numpy.ndarray
        (F,) int64 atom across each face, or -1 for a wall of the box.
    shifts : numpy.ndarray
        (F, 3) int32 periodic image of that atom.
    areas : numpy.ndarray
        (F,) area of each face in Å².
    vertices : numpy.ndarray
        (F,) int32 number of vertices of each face.
    """

    volume: np.ndarray
    coordination: np.ndarray
    radius: np.ndarray
    offsets: np.ndarray
    neighbors: np.ndarray
    shifts: np.ndarray
    areas: np.ndarray
    vertices: np.ndarray

    def index(self, largest: int = 8) -> np.ndarray:
        """(N, largest - 2) Voronoi index: the number of faces with 3, 4,
        ..., largest vertices of each cell (faces with more are not counted)."""
        atom = np.repeat(np.arange(len(self.volume)), np.diff(self.offsets))
        keep = (self.neighbors >= 0) & (self.vertices >= 3) & (self.vertices <= largest)
        width = largest - 2
        counts = np.bincount(
            atom[keep] * width + (self.vertices[keep] - 3),
            minlength=len(self.volume) * width,
        )
        return counts.reshape(len(self.volume), width)

    def columns(self) -> dict[str, np.ndarray]:
        return {
            "voronoi_volume": self.volume,
            "voronoi_coordination": self.coordination,
            "cavity_radius": self.radius,
        }


class VoronoiAnalysis:
    """Voronoi tessellation: the region of space closer to each atom than to
    any other, including periodic images.

    Parameters
    ----------
    area_threshold : float, optional
        Faces smaller than this fraction of their cell's surface do not
        count as neighbors, in the coordination and in the faces. Default 0.
    """

    def __init__(self, *, area_threshold: float = 0.0):
        if not 0.0 <= area_threshold < 1.0:
            raise InvalidParameter(
                f"area_threshold must be from 0 to below 1, not {area_threshold!r}"
            )
        self.area_threshold = float(area_threshold)

    def compute(
        self,
        system: System,
        *,
        neighbors: NeighborList | None = None,
        progress: Any = None,
        cancel: Any = None,
    ) -> VoronoiResult:
        """The cells of a system.

        Parameters
        ----------
        system : System
            The atoms. Directions that are not periodic are closed by the
            faces of the box, moved out as far as needed to include every atom.
        neighbors : NeighborList, optional
            A neighbor list of this system to start from; a larger one is built
            when it does not reach far enough.

        Returns
        -------
        VoronoiResult
            The cell of each atom.
        """
        n = len(system)
        box = system.box
        if neighbors is not None:
            nl = _given_or_built(system, 0.0, neighbors)
            reach = nl.cutoff
        else:
            nl = None
            reach = 2.5 * (box.volume / max(n, 1)) ** (1.0 / 3.0)
        while True:
            if nl is None:
                nl = NeighborList.build(system, reach)
            volume, radius, complete, offsets, pairs, areas, vertices = _native.voronoi(
                np.ascontiguousarray(system.positions),
                np.ascontiguousarray(box.matrix),
                np.ascontiguousarray(box.origin),
                np.ascontiguousarray(box.pbc),
                nl.offsets,
                np.ascontiguousarray(nl.vectors),
                reach,
                progress,
                _native_token(cancel),
            )
            if complete.all():
                break
            reach *= 1.5
            nl = None
        atom = np.repeat(np.arange(n), np.diff(offsets))
        surface = np.bincount(atom, weights=areas, minlength=n)
        keep = areas > self.area_threshold * surface[atom]
        wall = pairs < 0
        safe = np.where(wall, 0, pairs)
        neighbor = np.where(wall, -1, nl.j[safe].astype(np.int64))
        shifts = np.where(wall[:, None], 0, nl.shifts[safe])
        kept = np.concatenate([[0], np.cumsum(keep)])[offsets]
        return VoronoiResult(
            volume=volume,
            coordination=np.bincount(atom[keep & ~wall], minlength=n),
            radius=radius,
            offsets=kept.astype(np.int64),
            neighbors=neighbor[keep],
            shifts=shifts[keep].astype(np.int32),
            areas=areas[keep],
            vertices=vertices[keep],
        )

    def __repr__(self) -> str:
        return f"VoronoiAnalysis(area_threshold={self.area_threshold!r})"
