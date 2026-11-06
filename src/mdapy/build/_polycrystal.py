# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Polycrystals of Voronoi grains, and crystals of twin lamellae."""

import numpy as np
import polars as pl
from numpy.typing import ArrayLike

from mdapy._core import build as _native
from mdapy.build._defects import _axis
from mdapy.core.box import Box
from mdapy.core.system import VECTOR, System
from mdapy.core.topology import Topology
from mdapy.errors import InvalidParameter
from mdapy.neighbor import NeighborList, find_overlapping
from mdapy.neighbor._overlapping import _molecules_of

# points along the box vector of a gradient, and random places tried for each seed
_GRID = 4097
_CANDIDATES = 16
# what does not follow the atoms of the unit into the grains, or is made anew
_NOT_CARRIED = {
    "positions",
    "element",
    "velocities",
    "forces",
    "id",
    "image",
    "molecule",
    "grain",
}


def random_rotations(
    count: int, *, seed: int | np.random.Generator | None = None
) -> np.ndarray:
    """Rotation matrices drawn uniformly over all orientations.

    Parameters
    ----------
    count : int32
        Number of rotations.
    seed : int64 or numpy.random.Generator, optional
        Seed of the random rotations.

    Returns
    -------
    numpy.ndarray
        (count, 3, 3) float64 rotation matrices.
    """
    q = np.random.default_rng(seed).normal(size=(count, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    ).transpose(2, 0, 1)


def rotations_about(axis: int | str, angles: ArrayLike) -> np.ndarray:
    """Rotation matrices about one Cartesian axis, as the turns of the grains of a
    columnar polycrystal.

    Parameters
    ----------
    axis : int32 or str
        The axis: x, y, z or 0, 1, 2.
    angles : array_like of float64
        The angles in degrees, counterclockwise seen from the end of the axis.

    Returns
    -------
    numpy.ndarray
        (G, 3, 3) float64 rotation matrices.
    """
    k = _axis(axis)
    turn = np.radians(np.atleast_1d(np.asarray(angles, dtype=float)))
    if turn.ndim != 1:
        raise InvalidParameter("angles must be one number or a list of them")
    c, s = np.cos(turn), np.sin(turn)
    i, j = (k + 1) % 3, (k + 2) % 3
    r = np.zeros((len(turn), 3, 3))
    r[:, k, k] = 1
    r[:, i, i], r[:, i, j], r[:, j, i], r[:, j, j] = c, -s, s, c
    return r


def _whole_molecules(
    unit: System, sites: np.ndarray, topology: Topology
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """The molecule of each site numbered from 0, the fractional coordinates with each
    molecule in one piece and its center in the box, the centers of the molecules, and
    the nearest distance between atoms of different molecules."""
    molecule = _molecules_of(unit)
    sizes = np.bincount(molecule)
    leaders = np.unique(molecule, return_index=True)[1]
    pieces = np.rint(sites[leaders][molecule] - sites)
    whole = sites + pieces
    centers = np.zeros((len(sizes), 3))
    np.add.at(centers, molecule, whole)
    centers /= sizes[:, None]
    moved = np.floor(centers)
    for kind, table in topology.tables().items():
        if len(table):
            atoms = table["atoms"].to_numpy()
            if (molecule[atoms] != molecule[atoms[:, :1]]).any():
                raise InvalidParameter(f"{kind} must lie within one molecule")
    # the nearest of more neighbors than a molecule has atoms is of another molecule
    neighbors = NeighborList.build(unit, num_neighbors=int(sizes.max()))
    i, j = neighbors.i, neighbors.j
    same = molecule[i] == molecule[j]
    same &= (np.asarray(neighbors.shifts) == pieces[j] - pieces[i]).all(axis=1)
    return (
        molecule,
        whole - moved[molecule],
        centers - moved,
        float(neighbors.distances[~same].min()),
    )


def _copied(table: pl.DataFrame, number: np.ndarray) -> pl.DataFrame:
    """The terms of a topology in the copies of the unit, from the (C, A) new number of
    each atom of each copy, -1 for those left out: molecules are in or out whole."""
    atoms = number[:, table["atoms"].to_numpy()]
    found = (atoms >= 0).all(axis=2)
    width = atoms.shape[2]
    return table[np.nonzero(found)[1]].with_columns(
        pl.Series("atoms", atoms[found], dtype=pl.Array(pl.Int64, width))
    )


def build_polycrystal(
    unit: System,
    box: Box | ArrayLike,
    grains: int | ArrayLike,
    *,
    orientations: ArrayLike | None = None,
    min_distance: float | None = None,
    seed: int | np.random.Generator | None = None,
) -> System:
    """A polycrystal: each grain the crystal of a unit cell, rotated, filling
    the Voronoi cell of its seed in a periodic box. With a ``molecule`` column in the
    unit, the molecules stay whole.

    Parameters
    ----------
    unit : System
        The crystal of every grain, in a periodic box: a unit cell or any
        periodic arrangement, such as twin lamellae. Its other columns, such as
        charges, are kept. With a ``molecule`` column, a molecule belongs to the grain
        that holds its center, the atoms too close are removed by whole molecules, and
        the topology is kept; a molecule must be shorter than half the box of the unit.
    box : Box or array_like
        The box of the polycrystal, periodic in every direction: a Box or
        its lattice vectors. Along an axis no longer than the box of unit,
        the polycrystal is columnar: every grain spans that axis, its box
        takes the length of the unit there, and the grains turn about it.
    grains : int32 or array_like
        Number of grains, with seeds at random positions, or the (G, 3)
        positions of the seeds in Å.
    orientations : array_like, optional
        (G, 3, 3) float64 rotation of each grain, taking the axes of unit to those
        of the box; in a columnar polycrystal rotations about its axis.
        Uniformly random by default.
    min_distance : float64, optional
        No two atoms stay closer than this, in Å, with as few removed as possible, as
        ``mdapy.neighbor.find_overlapping`` does: atoms of neighboring grains that come
        too close, and the copies of atoms on the faces of a cell. Default 0.7 times
        the nearest distance in unit, between different molecules if there are any.
    seed : int64 or numpy.random.Generator, optional
        Seed of the random seeds and orientations.

    Returns
    -------
    System
        The atoms with a ``grain`` column, numbered as the seeds; each seed
        is a site of the unit in its grain. ``info["seeds"]`` holds the
        (G, 3) positions of the seeds. With molecules, ``molecule`` numbers them from
        1, and ``image`` tells how to put together those the box wraps.
    """
    if not unit.box.pbc.all() or len(unit) == 0:
        raise InvalidParameter("unit must be a periodic box with atoms")
    molecular = "molecule" in unit.table.columns
    topology = unit.topology if unit.topology is not None else Topology()
    if len(topology) and not molecular:
        raise InvalidParameter("a unit with a topology needs the molecule column")
    if not isinstance(box, Box):
        box = Box(box)
    if not box.pbc.all():
        raise InvalidParameter("a polycrystal needs a box periodic in every direction")
    rng = np.random.default_rng(seed)
    # a columnar polycrystal along an axis no longer than the unit
    unit_lengths = np.linalg.norm(unit.box.matrix, axis=1)
    lengths = np.linalg.norm(box.matrix, axis=1)
    thin = [k for k in range(3) if lengths[k] <= unit_lengths[k] + 1e-9]
    if len(thin) > 1:
        raise InvalidParameter("the box is no longer than the unit along two axes")
    axis = thin[0] if thin else None
    matrix = box.matrix.copy()
    if axis is not None:
        for m in (unit.box.matrix, box.matrix):
            if not np.allclose(m, np.diag(np.diag(m))):
                raise InvalidParameter("a columnar polycrystal needs orthogonal boxes")
        matrix[axis] = unit.box.matrix[axis]
        box = Box(matrix, origin=box.origin)
    if np.ndim(grains) == 0:
        if int(grains) < 1:
            raise InvalidParameter(f"grains must be at least 1, not {grains!r}")
        seeds = box.to_cartesian(rng.random((int(grains), 3)))
        if axis is not None:
            seeds[:, axis] = box.origin[axis]
    else:
        seeds = np.asarray(grains, dtype=float)
        if seeds.ndim != 2 or seeds.shape[1] != 3 or len(seeds) < 1:
            raise InvalidParameter("grains must be a number or (G, 3) seed positions")
    count = len(seeds)
    if orientations is None:
        if axis is None:
            rotations = random_rotations(count, seed=rng)
        else:
            rotations = rotations_about(axis, rng.uniform(0, 360, count))
    else:
        rotations = np.asarray(orientations, dtype=float)
        if rotations.shape != (count, 3, 3):
            raise InvalidParameter(f"orientations must have shape ({count}, 3, 3)")
        identity = np.einsum("gij,gkj->gik", rotations, rotations)
        if (
            not np.allclose(identity, np.eye(3), atol=1e-10)
            or not (np.linalg.det(rotations) > 0).all()
        ):
            raise InvalidParameter("orientations must be rotation matrices")
        if axis is not None and not np.allclose(
            rotations[:, axis, axis], 1.0, atol=1e-10
        ):
            raise InvalidParameter(
                "the grains of a columnar polycrystal turn about its axis"
            )
    cell = unit.box.matrix
    sites = unit.box.to_fractional(unit["positions"])
    molecule = centers = None
    if molecular:
        molecule, sites, centers, nearest = _whole_molecules(unit, sites, topology)
    else:
        nearest = NeighborList.build(unit, num_neighbors=1).distances.min()
        sites = sites - np.floor(sites)
    if min_distance is None:
        min_distance = 0.7 * nearest
    elif not min_distance >= 0:
        raise InvalidParameter(
            f"min_distance must not be negative, not {min_distance!r}"
        )

    from mdapy.analysis import VoronoiAnalysis

    # the cell of each grain: the seeds across its faces, and how far it reaches from
    # its seed, to the farthest corner, and a cell of unit more
    placed = box.wrap(seeds)[0]
    voronoi = VoronoiAnalysis(faces=True).compute(System(placed, box))
    owner = np.repeat(np.arange(count), np.diff(voronoi.offsets))
    faces = placed[voronoi.j] - placed[owner] + voronoi.shifts @ box.matrix
    reach = voronoi.radius + np.linalg.norm(cell.sum(axis=0))
    # the copies of unit within reach: 1 / |column of inv(cell)| is the spacing of the
    # lattice planes of each vector
    spans = np.ceil(reach[:, None] * np.linalg.norm(np.linalg.inv(cell), axis=0))
    spans = spans.astype(np.int64)
    if axis is not None:
        spans[:, axis] = 0
    # atoms of different grains too close are both this near a face, and atoms of one
    # grain are too close only below the nearest distance; molecules and the columns of
    # a columnar polycrystal reach out of their cells, and all their atoms are searched
    search_edges = 0.0 < min_distance < nearest and not molecular and axis is None
    # a grain takes the atoms, or the molecules, closer to its seed than to others; in
    # a columnar polycrystal, the distances across the axis only. The first atom of
    # unit is at the seed.
    points, grain, site, copy, edge = _native.fill_grains(
        np.ascontiguousarray(seeds),
        np.ascontiguousarray(rotations),
        reach,
        spans,
        voronoi.offsets,
        faces,
        cell,
        sites[0] @ cell,
        sites,
        np.arange(len(sites)) if molecule is None else molecule,
        sites if centers is None else centers,
        -1 if axis is None else axis,
        1e-9 * nearest,
        molecular,
        1.000001 * min_distance if search_edges else -1.0,
    )
    wrapped, image = box.wrap(points)
    names = (
        unit.table["element"]
        if "element" in unit.table.columns
        else pl.Series(["X"] * len(unit))
    )
    columns = {
        "positions": pl.Series(wrapped, dtype=VECTOR),
        "element": names.gather(site),
        "grain": grain,
    }
    for name in unit.table.columns:
        if name not in _NOT_CARRIED:
            columns[name] = unit.table[name].gather(site)
    crystal = System(table=pl.DataFrame(columns), box=box)
    if molecular:
        crystal = crystal.with_columns(
            molecule=copy * len(centers) + molecule[site], image=image
        )
        if len(topology):
            # the new number of each atom of each copy, -1 for those left out
            number = np.full((int(copy.max(initial=-1)) + 1, len(sites)), -1)
            number[copy, site] = np.arange(len(site))
            crystal = crystal.with_topology(
                Topology(
                    **{
                        kind: _copied(table, number)
                        for kind, table in topology.tables().items()
                        if len(table)
                    }
                )
            )
    if search_edges:
        close = np.flatnonzero(edge)
        close = close[find_overlapping(System(wrapped[close], box), min_distance)]
        if len(close):
            selected = np.zeros(len(crystal), dtype=bool)
            selected[close] = True
            crystal = crystal.filter(~selected)
    elif min_distance > 0:
        crystal = crystal.filter(
            ~find_overlapping(crystal, min_distance, molecules=molecular)
        )
    if molecular:
        numbers = np.unique(crystal["molecule"], return_inverse=True)[1]
        crystal = crystal.with_columns(molecule=numbers + 1)
    return crystal.with_info(seeds=seeds)


def gradient_seeds(
    box: Box | ArrayLike,
    sizes: ArrayLike,
    *,
    axis: int | str = 2,
    columns: int | str | None = None,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """The seeds of a polycrystal whose grain size changes along one box vector.

    Parameters
    ----------
    box : Box or array_like
        The box of the polycrystal: a Box or its lattice vectors.
    sizes : array_like
        (K, 2) float64, the positions along the box vector, as fractions from 0 to
        1, and the grain size in Å there, linear between them; the first at 0 and
        the last at 1, in increasing order.
    axis : int32 or str, optional
        The box vector along which the size changes. Default z.
    columns : int32 or str, optional
        The box vector of the columns of a columnar polycrystal: the seeds lie in
        one plane, and a grain size is the diameter of the circle of the same area.
        By default the seeds fill the box, and a grain size is the diameter of the
        sphere of the same volume.
    seed : int64 or numpy.random.Generator, optional
        Seed of the random seeds.

    Returns
    -------
    numpy.ndarray
        (G, 3) float64 positions of the seeds in Å, for ``build_polycrystal``.
    """
    if not isinstance(box, Box):
        box = Box(box)
    k = _axis(axis)
    thin = None if columns is None else _axis(columns)
    if thin == k:
        raise InvalidParameter("the size cannot change along the columns")
    table = np.asarray(sizes, dtype=float)
    if table.ndim != 2 or table.shape[1] != 2 or len(table) < 2:
        raise InvalidParameter("sizes must be pairs of a position and a grain size")
    where, size = table.T
    if where[0] != 0 or where[-1] != 1 or (np.diff(where) <= 0).any():
        raise InvalidParameter("the positions of sizes must increase from 0 to 1")
    if not (size > 0).all():
        raise InvalidParameter("the grain sizes must be positive")
    # the grains per fraction of the box vector, and their running total
    fractions = np.linspace(0.0, 1.0, _GRID)
    diameters = np.interp(fractions, where, size)
    if thin is None:
        density = 6.0 * box.volume / (np.pi * diameters**3)
    else:
        i, j = [m for m in range(3) if m != thin]
        area = np.linalg.norm(np.cross(box.matrix[i], box.matrix[j]))
        density = 4.0 * area / (np.pi * diameters**2)
    total = np.concatenate(
        [[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(fractions))]
    )
    count = max(int(round(total[-1])), 1)
    rng = np.random.default_rng(seed)
    # each grain in its own equal share of the total along the vector, in random
    # order; across it, the best of some random places, far from the grains so far
    along = np.interp(
        (rng.permutation(count) + rng.random(count)) / count,
        total / total[-1],
        fractions,
    )
    diameter = np.interp(along, where, size)
    free = [m for m in range(3) if m not in (k, thin)]
    placed = np.zeros((count, 3))
    placed[:, k] = along
    for g in range(count):
        tries = np.tile(placed[g], (_CANDIDATES, 1))
        tries[:, free] = rng.random((_CANDIDATES, len(free)))
        if g:
            apart = tries[:, None, :] - placed[:g]
            apart -= np.rint(apart)
            distance = np.linalg.norm(apart @ box.matrix, axis=2)
            room = (distance / (diameter[g] + diameter[:g])).min(axis=1)
            tries = tries[[int(room.argmax())]]
        placed[g] = tries[0]
    return box.to_cartesian(placed)


def twin_lamellae(cell: System, layers: int) -> System:
    """A crystal of twin lamellae: lamellae of layers atomic planes normal to
    z, each the mirror image of the next across the plane between them.

    Parameters
    ----------
    cell : System
        A periodic crystal whose third box vector is along z and whose atoms
        lie on equally spaced planes normal to z, such as fcc with z along
        [111].
    layers : int32
        Number of atomic planes in each lamella, at least 2.

    Returns
    -------
    System
        Two lamellae, 2 layers planes along z, with the box of cell in x and
        y; repeated along z, the twin boundaries are equally spaced.
    """
    matrix = cell.box.matrix
    if not cell.box.pbc.all() or abs(matrix[2, 0]) > 1e-9 or abs(matrix[2, 1]) > 1e-9:
        raise InvalidParameter("cell must be periodic, its third vector along z")
    if abs(matrix[0, 2]) > 1e-9 or abs(matrix[1, 2]) > 1e-9:
        raise InvalidParameter("the first two vectors of cell must lie in the xy plane")
    if int(layers) < 2:
        raise InvalidParameter(f"layers must be at least 2, not {layers!r}")
    layers = int(layers)
    fractional = cell.box.to_fractional(cell["positions"])
    fractional -= np.floor(fractional)
    heights = fractional[:, 2] * matrix[2, 2]
    planes = np.unique(np.round(heights, 6))
    spacing = matrix[2, 2] / len(planes)
    if not np.allclose(planes, planes[0] + spacing * np.arange(len(planes)), atol=1e-6):
        raise InvalidParameter(
            "the atoms of cell must lie on equally spaced planes normal to z"
        )
    names = (
        cell.table["element"].to_list()
        if "element" in cell.table.columns
        else ["X"] * len(cell)
    )
    # enough periods of cell along z for layers + 1 planes from the lowest
    periods = int(np.ceil((layers + 1) / len(planes)))
    points = np.concatenate([(fractional + [0, 0, p]) @ matrix for p in range(periods)])
    kinds = names * periods
    level = np.round((points[:, 2] - planes[0]) / spacing).astype(int)
    lower = level <= layers
    mirrored = (level >= 1) & (level <= layers - 1)
    top = planes[0] + layers * spacing
    image = points[mirrored].copy()
    image[:, 2] = 2 * top - image[:, 2]
    positions = np.concatenate([points[lower], image])
    elements = [k for k, keep in zip(kinds, lower) if keep] + [
        k for k, keep in zip(kinds, mirrored) if keep
    ]
    box = Box(
        [matrix[0], matrix[1], [0.0, 0.0, 2 * layers * spacing]], origin=cell.box.origin
    )
    wrapped, _ = box.wrap(positions)
    return System(wrapped, box, elements=elements)
