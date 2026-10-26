# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Graphene on the grain boundaries of a polycrystal."""

import numpy as np
import polars as pl

from mdapy._core import neighbor as _search
from mdapy.core.system import System
from mdapy.errors import InvalidParameter, MissingProperty
from mdapy.neighbor import NeighborList


def _boundaries(
    seeds: np.ndarray, box
) -> list[tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]]:
    """Each face between two grains once: its corners (in order around it),
    its unit normal, from the grain of the lower number, its area, and the
    seeds of the two grains, the image of each nearest the face."""
    from mdapy.analysis import VoronoiAnalysis

    cells = VoronoiAnalysis(polyhedra=True).compute(System(seeds, box)).polyhedra
    faces = []
    for i in range(len(seeds)):
        for f in range(cells.face_offsets[i], cells.face_offsets[i + 1]):
            j = int(cells.j[f])
            if j < i:
                continue
            corners = cells.vertices[
                cells.corners[cells.corner_offsets[f] : cells.corner_offsets[f + 1]]
            ]
            # the corners go counterclockwise seen from outside the cell:
            # Newell's normal points out of it
            normal = np.cross(corners, np.roll(corners, -1, axis=0)).sum(axis=0)
            normal /= np.linalg.norm(normal)
            if j == i:  # a grain facing its own image: each such face twice
                first = normal[np.flatnonzero(np.abs(normal) > 1e-9)[0]]
                if first < 0:
                    continue
            center = corners.mean(axis=0)
            apart = (
                box.to_fractional(center[None])[0]
                - box.to_fractional(seeds[i][None])[0]
            )
            inner = center - (apart - np.round(apart)) @ box.matrix
            outer = inner + 2 * ((center - inner) @ normal) * normal
            faces.append((corners, normal, float(cells.areas[f]), inner, outer))
    return faces


def _sheet(
    corners: np.ndarray,
    normal: np.ndarray,
    layers: int,
    bond: float,
    spacing: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Graphene of some layers, AB stacked and centered on the plane of a
    convex polygon whose corners go counterclockwise around normal, in a
    random orientation within it, cut to it: the positions of the atoms and
    the height of each above the plane."""
    center = corners.mean(axis=0)
    u = np.cross(normal, rng.normal(size=3))
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    local = (corners - center) @ np.stack([u, v]).T  # (k, 2), counterclockwise
    a = bond * np.sqrt(3)
    cell = np.array([[a, 0.0], [a / 2, a * np.sqrt(3) / 2]])
    basis = np.array([[0.0, 0.0], [a / 2, bond / 2]])  # the two atoms of a cell
    reach = np.linalg.norm(local, axis=1).max()
    span = int(np.ceil(2 * reach / a)) + 2
    grid = np.stack(
        np.meshgrid(*[np.arange(-span, span + 1)] * 2, indexing="ij"), -1
    ).reshape(-1, 2)
    lattice = (grid @ cell)[:, None, :] + basis[None]
    lattice = lattice.reshape(-1, 2) + rng.random(2) @ cell  # anywhere in the plane
    edges = np.roll(local, -1, axis=0) - local
    positions, heights = [], []
    for layer in range(layers):
        points = lattice + (layer % 2) * basis[1]  # AB stacking
        inside = np.ones(len(points), dtype=bool)
        for start, edge in zip(local, edges):
            relative = points - start
            inside &= edge[0] * relative[:, 1] - edge[1] * relative[:, 0] >= 0
        height = (layer - (layers - 1) / 2) * spacing
        kept = points[inside]
        positions.append(center + kept[:, :1] * u + kept[:, 1:] * v + height * normal)
        heights.append(np.full(len(kept), height))
    return np.concatenate(positions), np.concatenate(heights)


def _mitred(
    points: np.ndarray,
    heights: np.ndarray,
    own: np.ndarray,
    other: np.ndarray,
    seeds: np.ndarray,
    box,
) -> np.ndarray:
    """Whether each point of a layer a height off its face, on the side of
    the seed own, with other across the face, lies nearer that face than
    any other face of the cell of own: the layers of two faces meet where
    the planes halfway between them cross, instead of passing through each
    other."""
    keep = np.ones(len(points), dtype=bool)
    off = np.abs(heights) > 0
    if not off.any():
        return keep
    wrapped, _ = box.wrap(seeds)
    tree = _search.KdTree(
        np.ascontiguousarray(wrapped),
        np.ascontiguousarray(box.matrix),
        np.ascontiguousarray(box.origin),
        np.ascontiguousarray(box.pbc),
    )
    p = np.ascontiguousarray(points[off])
    j, shifts, _ = tree.nearest(p, min(16, 27 * len(seeds)))
    images = wrapped[j] + shifts @ box.matrix  # (m, k, 3)
    own, other = own[off][:, None, :], other[off][:, None, :]
    gap = np.linalg.norm(images - own, axis=2)
    face = np.linalg.norm(images - other, axis=2)
    # distance from the plane halfway between own and each other seed
    reach = (
        np.sum((p[:, None] - images) ** 2, axis=2)
        - np.sum((p[:, None] - own) ** 2, axis=2)
    ) / (2 * np.where(gap > 1e-6, gap, 1.0))
    reach[(gap <= 1e-6) | (face <= 1e-6)] = np.inf
    keep[off] = reach.min(axis=1) >= np.abs(heights[off]) - 1e-6
    return keep


def add_graphene(
    polycrystal: System,
    *,
    layers: int = 1,
    coverage: float = 1.0,
    clearance: float = 3.0,
    bond: float = 1.42,
    spacing: float = 3.35,
    seed: int | np.random.Generator | None = None,
) -> System:
    """Graphene on the grain boundaries of a polycrystal.

    Parameters
    ----------
    polycrystal : System
        A polycrystal from :func:`build_polycrystal`, with the seeds of its
        grains in ``info["seeds"]`` and a ``grain`` column.
    layers : int, optional
        Number of graphene layers on each boundary, AB stacked and centered
        on it; where two faces meet, each layer ends where the planes
        halfway between the faces cross. Default 1.
    coverage : float, optional
        Fraction of the area of the boundaries covered, from 0 to 1: whole
        faces between two grains are covered, in random order, until the
        fraction is reached. Default 1, every face.
    clearance : float, optional
        Atoms of the polycrystal closer than this to a carbon atom, in Å,
        are removed. Default 3, about the distance of graphene from copper.
    bond : float, optional
        Carbon-carbon bond length of graphene in Å. Default 1.42.
    spacing : float, optional
        Distance between graphene layers in Å. Default 3.35.
    seed : int or numpy.random.Generator, optional
        Seed of the random faces and orientations of the graphene.

    Returns
    -------
    System
        The atoms of the polycrystal left and the carbon atoms of the
        graphene, whose ``grain`` is -1; of two carbon atoms of different
        faces closer than 0.85 bond, where faces meet, the later is left
        out.

    Raises
    ------
    MissingProperty
        If the polycrystal has no seeds in its info.
    InvalidParameter
        For layers below 1, a coverage outside 0 to 1, or lengths that are
        not positive.
    """
    if "seeds" not in polycrystal.info or "grain" not in polycrystal.table.columns:
        raise MissingProperty("the polycrystal needs info['seeds'] and a grain column")
    if int(layers) < 1:
        raise InvalidParameter(f"layers must be at least 1, not {layers!r}")
    if not 0 <= coverage <= 1:
        raise InvalidParameter(f"coverage must be from 0 to 1, not {coverage!r}")
    for name, value in (("clearance", clearance), ("bond", bond), ("spacing", spacing)):
        if not value > 0:
            raise InvalidParameter(f"{name} must be positive, not {value!r}")
    rng = np.random.default_rng(seed)
    box = polycrystal.box
    seeds = np.asarray(polycrystal.info["seeds"], dtype=float)
    faces = _boundaries(seeds, box) if len(seeds) > 1 else []
    total = sum(face[2] for face in faces)
    chosen, covered = [], 0.0
    for k in rng.permutation(len(faces)):
        if covered >= coverage * total - 1e-9 * total:
            break
        chosen.append(faces[k])
        covered += faces[k][2]
    if not chosen:
        return polycrystal
    sheets = []
    for corners, normal, _, inner, outer in chosen:
        points, heights = _sheet(corners, normal, int(layers), bond, spacing, rng)
        own = np.where((heights > 0)[:, None], outer, inner)
        across = np.where((heights > 0)[:, None], inner, outer)
        sheets.append(points[_mitred(points, heights, own, across, seeds, box)])
    carbon, _ = box.wrap(np.concatenate(sheets))
    face = np.repeat(np.arange(len(sheets)), [len(s) for s in sheets])
    # where faces meet, the carbon atoms of the later face too close go
    keep = np.ones(len(carbon), dtype=bool)
    nl = NeighborList.build(System(carbon, box), 0.85 * bond)
    for i, j in zip(nl.i, nl.j):
        if face[i] < face[j] and keep[i]:
            keep[j] = False
    carbon = carbon[keep]
    # the atoms of the polycrystal too close to the carbon go
    n = len(polycrystal)
    both = np.concatenate([polycrystal["positions"], carbon])
    nl = NeighborList.build(System(both, box), clearance)
    near = (nl.i < n) & (nl.j >= n)
    left = np.ones(n, dtype=bool)
    left[nl.i[near]] = False
    metal = polycrystal.gather(np.flatnonzero(left))
    graphene = System(carbon, box, elements="C").with_columns(
        grain=pl.Series(np.full(len(carbon), -1), dtype=metal.table["grain"].dtype)
    )
    table = pl.concat(
        [metal.table, graphene.table.select(metal.table.columns)], how="vertical"
    )
    return System(table=table, box=box, info=polycrystal.info)
