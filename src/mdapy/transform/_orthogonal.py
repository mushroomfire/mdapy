# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""A crystal in a box whose lattice vectors lie along x, y and z."""

import numpy as np
import polars as pl

from mdapy.core.box import Box
from mdapy.core.system import System
from mdapy.core.topology import KINDS, Topology
from mdapy.errors import InvalidParameter
from mdapy.transform._periodic import _EXTENSIVE_INFO


def _along_axes(
    matrix: np.ndarray, most: int, tolerance: float
) -> tuple[np.ndarray | None, int]:
    """(3, 3) int64: for x, y and z the shortest integer combination of the
    rows of matrix, each from -most to most, that lies along that axis in its
    positive direction; None and the axis where there is none."""
    span = np.arange(-most, most + 1)
    combos = np.stack(np.meshgrid(span, span, span, indexing="ij"), -1).reshape(-1, 3)
    # the smallest combinations first, so that ties go to them
    combos = combos[np.argsort(np.abs(combos).sum(axis=1), kind="stable")][1:]
    vectors = combos @ matrix
    rows = []
    for axis in range(3):
        others = [d for d in range(3) if d != axis]
        along = (np.abs(vectors[:, others]) <= tolerance).all(axis=1)
        found = np.flatnonzero(along & (vectors[:, axis] > tolerance))
        if len(found) == 0:
            return None, axis
        rows.append(combos[found[np.argmin(vectors[found, axis])]])
    return np.array(rows, dtype=np.int64), -1


def _lattice_points(rows: np.ndarray) -> np.ndarray:
    """(C, 3) int64 the points of the lattice of the cell in the supercell
    whose vectors are rows times those of the cell, C = |det rows|."""
    corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    corners = corners @ rows
    low, high = corners.min(axis=0), corners.max(axis=0)
    grid = np.stack(
        np.meshgrid(*(np.arange(a, b + 1) for a, b in zip(low, high)), indexing="ij"),
        -1,
    ).reshape(-1, 3)
    fractional = grid @ np.linalg.inv(rows)
    inside = ((fractional >= -1e-9) & (fractional < 1 - 1e-9)).all(axis=1)
    return grid[inside]


def _copies_of_terms(
    topology: Topology,
    n: int,
    matrix: np.ndarray,
    unwrapped: np.ndarray,
    nearest: bool,
    rows: np.ndarray,
    points: np.ndarray,
) -> Topology:
    """The terms of the topology in each copy, each atom of a term in the copy
    where it lies next to the first atom of the term."""
    inverse = np.linalg.inv(matrix)
    inverse_rows = np.linalg.inv(rows)
    low = points.min(axis=0)
    size = points.max(axis=0) - low + 1
    lookup = np.full(int(np.prod(size)), -1, dtype=np.int64)
    keys = np.ravel_multi_index(tuple((points - low).T), tuple(size))
    lookup[keys] = np.arange(len(points))
    box = Box(matrix)
    tables = {}
    for kind, df in topology.tables().items():
        width = KINDS[kind]
        atoms = df["atoms"].to_numpy().reshape(len(df), width)
        first = unwrapped[atoms[:, :1]]
        delta = unwrapped[atoms] - first
        if nearest:  # no image flags: each atom at the image nearest the first
            delta = box.minimum_image(delta.reshape(-1, 3)).reshape(delta.shape)
        # the lattice translation that takes each atom next to the first
        shift = np.round((first + delta - unwrapped[atoms]) @ inverse).astype(np.int64)
        lattice = points[:, None, None, :] + shift[None]
        fractional = lattice @ inverse_rows
        fractional -= np.floor(fractional + 1e-9)
        reduced = np.round(fractional @ rows).astype(np.int64) - low
        copy = lookup[
            np.ravel_multi_index(tuple(np.moveaxis(reduced, -1, 0)), tuple(size))
        ]
        moved = copy * n + atoms[None]
        tables[kind] = pl.concat([df] * len(points)).with_columns(
            pl.Series(
                "atoms", moved.reshape(-1, width), dtype=pl.Array(pl.Int64, width)
            )
        )
    return Topology(**tables)


def orthogonal_cell(
    system: System,
    *,
    minimal: bool = False,
    max_search: int = 20,
    tolerance: float = 1e-6,
) -> System:
    """The same crystal in a box whose lattice vectors lie along x, y and z,
    as atomsk's orthogonal cell: along each axis, the shortest combination of
    the lattice vectors of the box that lies along it.

    Parameters
    ----------
    system : System
        A crystal in a box periodic in every direction.
    minimal : bool, optional
        Start from the primitive cell of the crystal, which gives the smallest
        orthogonal cell; otherwise from the box, as atomsk does, so that the
        cell of a box of several periods holds that many. Default False.
    max_search : int32, optional
        The largest multiple of each lattice vector in the combinations
        searched. Default 20.
    tolerance : float64, optional
        How far in Å a combination may lie from an axis. Default 1e-6.

    Returns
    -------
    System
        The copies of the atoms of the box, or of the primitive cell, at the
        points of its lattice in the orthogonal box, which has the origin of
        the box; the box itself when it is already orthogonal. Ids and
        molecules (those above 0) of each copy are offset by the largest of
        the original; with an ``image`` column, each atom is placed by its
        unwrapped position and its image flags are those of the new box; each
        topology term joins the atoms of the copies where they lie next to its
        first atom. ``energy`` and ``virial`` are left out of the info.

    Raises
    ------
    InvalidParameter
        If the box is not periodic in every direction, ``max_search`` is not
        a positive integer, ``tolerance`` is not positive, or no combination
        lies along an axis, as for a crystal turned at an angle with no
        lattice vector along it.
    """
    if not system.box.pbc.all():
        raise InvalidParameter(
            "an orthogonal cell needs a box periodic in every direction"
        )
    if (
        isinstance(max_search, bool)
        or not isinstance(max_search, (int, np.integer))
        or max_search < 1
    ):
        raise InvalidParameter(
            f"max_search must be a positive integer, not {max_search!r}"
        )
    if not tolerance > 0:
        raise InvalidParameter(f"tolerance must be positive, not {tolerance!r}")
    source = system
    if minimal:
        # here, as the lattice dynamics import the transforms
        from mdapy.lattice_dynamics._phonons import primitive_cell

        source = primitive_cell(system)
    box = source.box
    matrix = box.matrix
    rows, axis = _along_axes(matrix, int(max_search), float(tolerance))
    if rows is None:
        raise InvalidParameter(
            f"no combination of the lattice vectors of up to {max_search} periods "
            f"each lies along {'xyz'[axis]}: the crystal needs a lattice vector "
            "along each axis"
        )
    if np.array_equal(rows, np.eye(3, dtype=np.int64)) and box.is_orthogonal:
        return source
    lengths = np.diagonal(rows @ matrix)
    orthogonal = Box(np.diag(lengths), origin=box.origin)
    points = _lattice_points(rows)
    n, copies = len(source), len(points)
    every = np.tile(np.arange(n), copies)
    copy = np.repeat(np.arange(copies), n)
    table = source.table[every]
    unwrapped = source["positions"]
    imaged = "image" in table.columns
    if imaged:
        unwrapped = (
            unwrapped + source.table["image"].to_numpy().astype(np.int64) @ matrix
        )
    positions, flags = orthogonal.wrap(unwrapped[every] + (points @ matrix)[copy])
    columns = {"positions": positions}
    if imaged:
        columns["image"] = flags.astype(np.int32)
    if n:
        for name in ("id", "molecule"):
            if name in table.columns:
                values = table[name].to_numpy()
                shifted = values + copy * int(source.table[name].max())
                columns[name] = (
                    shifted if name == "id" else np.where(values > 0, shifted, values)
                )
    info = {k: v for k, v in source.info.items() if k not in _EXTENSIVE_INFO}
    topology = source.topology
    if topology is not None:
        topology = _copies_of_terms(
            topology, n, matrix, unwrapped, not imaged, rows, points
        )
    made = System(table=table, box=orthogonal, info=info, topology=topology)
    return made.with_columns(**columns)
