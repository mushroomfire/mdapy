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
from mdapy.neighbor._neighbor_list import _voronoi_thresholds


@dataclass(frozen=True)
class VoronoiPolyhedra(Result):
    """The shapes of the Voronoi cells: their vertices and all their faces.

    Attributes
    ----------
    vertices : numpy.ndarray
        (V, 3) positions of the vertices in Å, each cell with its own.
    vertex_offsets : numpy.ndarray
        (N + 1,) int64: the vertices of the cell of atom ``i`` are
        ``vertex_offsets[i]:vertex_offsets[i + 1]``.
    face_offsets : numpy.ndarray
        (N + 1,) int64: the faces of the cell of atom ``i`` are
        ``face_offsets[i]:face_offsets[i + 1]``.
    corner_offsets : numpy.ndarray
        (F + 1,) int64: the corners of face ``f`` are
        ``corners[corner_offsets[f]:corner_offsets[f + 1]]``.
    corners : numpy.ndarray
        (C,) int64 vertex of each corner, counterclockwise seen from outside
        the cell.
    j : numpy.ndarray
        (F,) the atom across each face, or -1 for a wall of the box, of
        ``mdapy.index_dtype``.
    areas : numpy.ndarray
        (F,) area of each face in Å².
    """

    vertices: np.ndarray
    vertex_offsets: np.ndarray
    face_offsets: np.ndarray
    corner_offsets: np.ndarray
    corners: np.ndarray
    j: np.ndarray
    areas: np.ndarray


@dataclass(frozen=True)
class VoronoiResult(Result):
    """Voronoi cells of the atoms.

    Attributes
    ----------
    volume : numpy.ndarray
        (N,) volume of each cell in Å³.
    coordination : numpy.ndarray
        (N,) int64 number of faces that count shared with other atoms.
    radius : numpy.ndarray
        (N,) distance from each atom to the farthest vertex of its cell, in Å.
    offsets : numpy.ndarray
        (N + 1,) int64: the faces that count of atom ``i`` are
        ``offsets[i]:offsets[i + 1]``.
    j : numpy.ndarray
        (F,) the atom across each face, or -1 for a wall of the box, of
        ``mdapy.index_dtype``.
    shifts : numpy.ndarray
        (F, 3) int8 periodic image of that atom.
    areas : numpy.ndarray
        (F,) area of each face in Å².
    edges : numpy.ndarray
        (F,) int32 number of edges of each face that count.
    polyhedra : VoronoiPolyhedra or None
        The shapes of the cells, when asked for.
    """

    volume: np.ndarray
    coordination: np.ndarray
    radius: np.ndarray
    offsets: np.ndarray
    j: np.ndarray
    shifts: np.ndarray
    areas: np.ndarray
    edges: np.ndarray
    polyhedra: VoronoiPolyhedra | None = None

    def index(self, largest: int = 8) -> np.ndarray:
        """The Voronoi index of each cell.

        Parameters
        ----------
        largest : int, optional
            The largest number of edges counted. Default 8.

        Returns
        -------
        numpy.ndarray
            (N, largest - 2) int64 number of faces with 3, 4, ..., largest
            edges shared with other atoms.
        """
        if largest < 3:
            raise InvalidParameter(f"largest must be at least 3, not {largest!r}")
        n = len(self.volume)
        atom = np.repeat(np.arange(n), np.diff(self.offsets))
        keep = (self.j >= 0) & (self.edges <= largest)
        width = largest - 2
        counts = np.bincount(
            atom[keep] * width + (self.edges[keep] - 3), minlength=n * width
        )
        return counts.reshape(n, width)

    def columns(self) -> dict[str, np.ndarray]:
        return {
            "voronoi_volume": self.volume,
            "voronoi_coordination": self.coordination,
            "cavity_radius": self.radius,
        }


class VoronoiAnalysis:
    """Voronoi cells: the region of space closer to each atom than to any
    other, periodic images included.

    Parameters
    ----------
    face_threshold : float, optional
        Faces of this area or less, in Å², do not count. Default 0.
    relative_face_threshold : float, optional
        Faces of this fraction of the surface of their cell or less do not
        count, from 0 to below 1. Default 0.
    edge_threshold : float, optional
        Edges shorter than this, in Å, do not count; faces with fewer than
        three edges that count do not count. Default 0.
    polyhedra : bool, optional
        Also give the shapes of the cells. Default False.
    """

    def __init__(
        self,
        *,
        face_threshold: float = 0.0,
        relative_face_threshold: float = 0.0,
        edge_threshold: float = 0.0,
        polyhedra: bool = False,
    ):
        self.face_threshold, self.relative_face_threshold, self.edge_threshold = (
            _voronoi_thresholds(face_threshold, relative_face_threshold, edge_threshold)
        )
        self.polyhedra = bool(polyhedra)

    def compute(
        self, system: System, *, progress: Any = None, cancel: Any = None
    ) -> VoronoiResult:
        """The cells of a system.

        Parameters
        ----------
        system : System
            The atoms, no two at the same position. Directions that are not
            periodic are closed by the faces of the box, moved out as far as
            needed to include every atom.

        Returns
        -------
        VoronoiResult
            The cell of each atom.
        """
        box = system.box
        try:
            volume, radius, coordination, offsets, j, shifts, areas, edges, shapes = (
                _native.voronoi(
                    np.ascontiguousarray(system.positions),
                    np.ascontiguousarray(box.matrix),
                    np.ascontiguousarray(box.origin),
                    np.ascontiguousarray(box.pbc),
                    self.face_threshold,
                    self.relative_face_threshold,
                    self.edge_threshold,
                    self.polyhedra,
                    progress,
                    _native_token(cancel),
                )
            )
        except (ValueError, OverflowError) as error:
            raise InvalidParameter(str(error)) from None
        polyhedra = None
        if shapes is not None:
            vertices, *rest = shapes
            polyhedra = VoronoiPolyhedra(vertices.reshape(-1, 3), *rest)
        return VoronoiResult(
            volume=volume,
            coordination=coordination,
            radius=radius,
            offsets=offsets,
            j=j,
            shifts=shifts.reshape(-1, 3),
            areas=areas,
            edges=edges,
            polyhedra=polyhedra,
        )

    def __repr__(self) -> str:
        return (
            f"VoronoiAnalysis(face_threshold={self.face_threshold!r}, "
            f"relative_face_threshold={self.relative_face_threshold!r}, "
            f"edge_threshold={self.edge_threshold!r}, polyhedra={self.polyhedra})"
        )
