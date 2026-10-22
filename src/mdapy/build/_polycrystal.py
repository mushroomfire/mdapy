# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Polycrystals of Voronoi grains, and crystals of twin lamellae."""

import numpy as np
from numpy.typing import ArrayLike

from mdapy._core import neighbor as _search
from mdapy.core.box import Box
from mdapy.core.system import System
from mdapy.errors import InvalidParameter
from mdapy.neighbor import NeighborList


def random_rotations(
    count: int, seed: int | np.random.Generator | None = None
) -> np.ndarray:
    """Rotation matrices drawn uniformly over all orientations.

    Parameters
    ----------
    count : int
        Number of rotations.
    seed : int or numpy.random.Generator, optional
        Seed of the random rotations.

    Returns
    -------
    numpy.ndarray
        (count, 3, 3) rotation matrices.
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


def _rotations_about(axis: int, angles: np.ndarray) -> np.ndarray:
    c, s = np.cos(angles), np.sin(angles)
    i, j = [k for k in range(3) if k != axis]
    r = np.zeros((len(angles), 3, 3))
    r[:, axis, axis] = 1
    r[:, i, i], r[:, i, j], r[:, j, i], r[:, j, j] = c, -s, s, c
    return r


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
    the Voronoi cell of its seed in a periodic box.

    Parameters
    ----------
    unit : System
        The crystal of every grain, in a periodic box: a unit cell or any
        periodic arrangement, such as twin lamellae.
    box : Box or array_like
        The box of the polycrystal, periodic in every direction: a Box or
        its lattice vectors. Along an axis no longer than the box of unit,
        the polycrystal is columnar: every grain spans that axis, its box
        takes the length of the unit there, and the grains turn about it.
    grains : int or array_like
        Number of grains, with seeds at random positions, or the (G, 3)
        positions of the seeds in Å.
    orientations : array_like, optional
        (G, 3, 3) rotation of each grain, taking the axes of unit to those
        of the box; in a columnar polycrystal rotations about its axis.
        Uniformly random by default.
    min_distance : float, optional
        Of two atoms closer than this, in Å, the later is removed: atoms of
        neighboring grains that come too close, and the copies of atoms on
        the faces of a cell. Default 0.7 times the nearest distance in unit.
    seed : int or numpy.random.Generator, optional
        Seed of the random seeds and orientations.

    Returns
    -------
    System
        The atoms with a ``grain`` column, numbered as the seeds; each seed
        is a site of the unit in its grain.
    """
    if not unit.box.pbc.all() or len(unit) == 0:
        raise InvalidParameter("unit must be a periodic box with atoms")
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
            rotations = random_rotations(count, rng)
        else:
            rotations = _rotations_about(axis, rng.uniform(0, 2 * np.pi, count))
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
    nearest = NeighborList.build(unit, num_neighbors=1).distances.min()
    if min_distance is None:
        min_distance = 0.7 * nearest
    elif not min_distance >= 0:
        raise InvalidParameter(
            f"min_distance must not be negative, not {min_distance!r}"
        )

    frame = (
        np.ascontiguousarray(box.matrix),
        np.ascontiguousarray(box.origin),
        np.ascontiguousarray(box.pbc),
    )
    tree = _search.KdTree(np.ascontiguousarray(seeds), *frame)
    # how far a grain may reach: beyond the farthest seed of its neighbors
    if count > 1:
        _, _, far = tree.nearest(np.ascontiguousarray(seeds), min(count, 27))
        reach = far.max(axis=1)
    else:
        reach = np.full(1, 0.0)
    reach = np.maximum(reach, 0.5 * np.linalg.norm(box.matrix.sum(axis=0)))
    reach += np.linalg.norm(unit.box.matrix.sum(axis=0))
    cell = unit.box.matrix
    sites = unit.box.to_fractional(unit.positions)
    sites -= np.floor(sites)
    names = (
        unit.table["element"].to_list()
        if "element" in unit.table.columns
        else ["X"] * len(unit)
    )
    positions, grain, elements = [], [], []
    for g in range(count):
        # the cells of unit within reach: 1 / |column of inv(cell)| is the
        # spacing of the lattice planes of each vector
        span = np.ceil(reach[g] * np.linalg.norm(np.linalg.inv(cell), axis=0)).astype(
            int
        )
        if axis is not None:
            span[axis] = 0
        cells = np.stack(
            np.meshgrid(*[np.arange(-s, s + 1) for s in span], indexing="ij"), -1
        ).reshape(-1, 1, 3)
        local = ((cells + sites) @ cell).reshape(-1, 3)
        local -= sites[0] @ cell  # the first atom of unit at the seed
        points = seeds[g] + local @ rotations[g].T
        # in a columnar polycrystal, the distances across the axis only
        query = points.copy()
        if axis is not None:
            query[:, axis] = seeds[g, axis]
        _, _, closest = tree.nearest(np.ascontiguousarray(query), 1)
        own = np.linalg.norm(query - seeds[g], axis=1)
        inside = own <= closest[:, 0] + 1e-9 * nearest
        positions.append(points[inside])
        grain.append(np.full(int(inside.sum()), g))
        elements += [names[k % len(names)] for k in np.flatnonzero(inside)]
    wrapped, _ = box.wrap(np.concatenate(positions))
    grain = np.concatenate(grain)
    keep = np.ones(len(wrapped), dtype=bool)
    if min_distance > 0:
        nl = NeighborList.build(System(wrapped, box), min_distance)
        first, second = nl.i, nl.j
        for i, j in zip(first[first < second], second[first < second]):
            if keep[i] and keep[j]:
                keep[j] = False
    elements = np.array(elements, dtype=object)[keep].tolist()
    return System(wrapped[keep], box, elements=elements).with_columns(grain=grain[keep])


def twin_lamellae(cell: System, layers: int) -> System:
    """A crystal of twin lamellae: lamellae of layers atomic planes normal to
    z, each the mirror image of the next across the plane between them.

    Parameters
    ----------
    cell : System
        A periodic crystal whose third box vector is along z and whose atoms
        lie on equally spaced planes normal to z, such as fcc with z along
        [111].
    layers : int
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
    fractional = cell.box.to_fractional(cell.positions)
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
