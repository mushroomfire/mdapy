# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Coordination polyhedra: the convex hull of the neighbors of an atom."""

import itertools
from dataclasses import dataclass
from functools import cache

import numpy as np
from numpy.typing import ArrayLike

from mdapy.core._atoms import _mask
from mdapy.core.mesh import TriangleMesh
from mdapy.core.result import Result
from mdapy.core.system import System
from mdapy.errors import InvalidParameter, MissingProperty
from mdapy.neighbor import NeighborList


@cache
def _triples(k: int) -> np.ndarray:
    return np.array(list(itertools.combinations(range(k), 3)), dtype=np.int64)


def _hull(points: np.ndarray) -> np.ndarray:
    """(T, 3) triangles of the convex hull of a few points, counterclockwise
    seen from outside; the points of a face in one plane make one polygon,
    cut into a fan."""
    k = len(points)
    if k < 4:
        return np.zeros((0, 3), dtype=np.int64)
    scale = np.abs(points - points.mean(0)).max()
    eps = 1e-6 * scale
    triples = _triples(k)
    a = points[triples[:, 0]]
    normals = np.cross(points[triples[:, 1]] - a, points[triples[:, 2]] - a)
    lengths = np.linalg.norm(normals, axis=1)
    good = lengths > 1e-6 * scale * scale
    normals, a = normals[good] / lengths[good, None], a[good]
    distance = np.einsum("tkd,td->tk", points[None] - a[:, None], normals)
    below, above = (distance <= eps).all(1), (distance >= -eps).all(1)
    triangles, seen = [], set()
    for t in np.flatnonzero(below | above):
        on = np.flatnonzero(np.abs(distance[t]) <= eps)
        if tuple(on) in seen:
            continue
        seen.add(tuple(on))
        normal = normals[t] if below[t] else -normals[t]  # outward
        middle = points[on].mean(0)
        u = points[on[0]] - middle
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        angle = np.arctan2((points[on] - middle) @ v, (points[on] - middle) @ u)
        loop = on[np.argsort(angle)]  # counterclockwise about the normal
        triangles += [(loop[0], loop[i], loop[i + 1]) for i in range(1, len(loop) - 1)]
    return np.array(triangles, dtype=np.int64).reshape(-1, 3)


@dataclass(frozen=True)
class CoordinationPolyhedraResult(Result):
    """Coordination polyhedra.

    Attributes
    ----------
    mesh : TriangleMesh
        The faces of every polyhedron, its vertices at the neighbors next to
        the atom, whole, even where they stick out of the box.
    atoms : numpy.ndarray
        (M,) the atom at the middle of each polyhedron.
    owners : numpy.ndarray
        (F,) the polyhedron of each triangle of the mesh, an index of atoms.
    volumes : numpy.ndarray
        (M,) the volume of each polyhedron in Å³; 0 for fewer than four
        neighbors or neighbors in a plane.
    """

    mesh: TriangleMesh
    atoms: np.ndarray
    owners: np.ndarray
    volumes: np.ndarray


class CoordinationPolyhedra:
    """The convex hull of the neighbors of atoms, as OVITO's coordination
    polyhedra.

    Parameters
    ----------
    cutoff : float, optional
        The neighbors closer than this, in Å.
    num_neighbors : int, optional
        The nearest neighbors, this many.
    bonds : bool, optional
        The atoms bonded to each, by the bonds of the topology.

    Give exactly one of cutoff, num_neighbors and bonds.
    """

    def __init__(
        self,
        cutoff: float | None = None,
        *,
        num_neighbors: int | None = None,
        bonds: bool = False,
    ):
        if (cutoff is not None) + (num_neighbors is not None) + bool(bonds) != 1:
            raise InvalidParameter(
                "give exactly one of cutoff, num_neighbors and bonds"
            )
        self.cutoff, self.num_neighbors, self.bonds = cutoff, num_neighbors, bool(bonds)

    def compute(
        self, system: System, atoms: ArrayLike | None = None
    ) -> CoordinationPolyhedraResult:
        """The polyhedra about some atoms of a system.

        Parameters
        ----------
        system : System
            The atoms.
        atoms : array_like, optional
            A boolean mask or indices of the atoms at the middle of the
            polyhedra. Default all.

        Returns
        -------
        CoordinationPolyhedraResult
            The mesh of the polyhedra and their volumes.
        """
        n = len(system)
        centers = np.arange(n) if atoms is None else np.flatnonzero(_mask(atoms, n))
        positions = system["positions"]
        if self.bonds:
            if system.topology is None or not len(system.topology.bonds):
                raise MissingProperty("the system has no bonds")
            pairs = np.asarray(system.topology.bonds["atoms"].to_list(), dtype=np.int64)
            i = np.concatenate([pairs[:, 0], pairs[:, 1]])
            j = np.concatenate([pairs[:, 1], pairs[:, 0]])
            order = np.argsort(i, kind="stable")
            i, j = i[order], j[order]
            offsets = np.searchsorted(i, np.arange(n + 1))
            step = system.box.to_fractional(positions[j]) - system.box.to_fractional(
                positions[i]
            )
            periodic = system.box.pbc
            step[:, periodic] -= np.round(step[:, periodic])
            vectors = step @ system.box.matrix
        else:
            nl = (
                NeighborList.build(system, float(self.cutoff))
                if self.cutoff is not None
                else NeighborList.build(system, num_neighbors=int(self.num_neighbors))
            )
            offsets, vectors = np.asarray(nl.offsets), nl.vectors
        vertices, faces, owners, volumes = [], [], [], []
        start = 0
        for m, center in enumerate(centers):
            points = vectors[offsets[center] : offsets[center + 1]]
            triangles = _hull(points)
            if len(triangles):
                a, b, c = (points[triangles[:, k]] for k in range(3))
                volumes.append(np.einsum("td,td->t", a, np.cross(b, c)).sum() / 6)
            else:
                volumes.append(0.0)
            vertices.append(positions[center] + points)
            faces.append(triangles + start)
            owners.append(np.full(len(triangles), m))
            start += len(points)
        mesh = TriangleMesh(
            np.concatenate(vertices) if vertices else np.zeros((0, 3)),
            np.concatenate(faces) if faces else np.zeros((0, 3), dtype=np.int64),
        )
        return CoordinationPolyhedraResult(
            mesh=mesh,
            atoms=centers,
            owners=np.concatenate(owners) if owners else np.zeros(0, dtype=np.int64),
            volumes=np.array(volumes, dtype=np.float64),
        )

    def __repr__(self) -> str:
        if self.bonds:
            return "CoordinationPolyhedra(bonds=True)"
        if self.cutoff is not None:
            return f"CoordinationPolyhedra({self.cutoff!r})"
        return f"CoordinationPolyhedra(num_neighbors={self.num_neighbors})"
