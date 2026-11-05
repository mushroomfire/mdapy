# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Atoms too close to another."""

import numpy as np
from numpy.typing import ArrayLike

from mdapy.core._atoms import _mask
from mdapy.core.system import System
from mdapy.errors import MissingProperty
from mdapy.neighbor._neighbor_list import NeighborList


def _fewest_to_delete(count: int, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """(count,) bool, what to delete so that no pair has both of its ends left, keeping
    first those in the fewest pairs: the fewest a simple rule can delete, as the very
    fewest is too hard to find. Each pair is listed both ways, sorted by first."""
    pairs = np.bincount(first, minlength=count)
    offsets = np.concatenate([[0], np.cumsum(pairs)])
    paired = np.flatnonzero(pairs)
    deleted = np.zeros(count, dtype=bool)
    for k in paired[np.argsort(pairs[paired], kind="stable")].tolist():
        if not deleted[k]:
            deleted[second[offsets[k] : offsets[k + 1]]] = True
    return deleted


def _molecules_of(system: System) -> np.ndarray:
    """(N,) int64, the molecule of each atom numbered from 0; an atom of molecule 0
    belongs to none, as in LAMMPS, and counts as a molecule of its own."""
    if "molecule" not in system.table.columns:
        raise MissingProperty("molecules need the molecule column")
    numbers = np.asarray(system["molecule"], dtype=np.int64)
    alone = -1 - np.arange(len(numbers))
    return np.unique(np.where(numbers == 0, alone, numbers), return_inverse=True)[1]


def _overlapping_molecules(
    molecule: np.ndarray, first: np.ndarray, second: np.ndarray, keep_one: bool
) -> np.ndarray:
    """(N,) bool, the atoms of the molecules selected, from the pairs of atoms."""
    count = int(molecule.max()) + 1 if len(molecule) else 1
    one, other = molecule[first], molecule[second]
    apart = one != other
    if not keep_one:
        selected = np.zeros(count, dtype=bool)
        selected[other[apart]] = True
        return selected[molecule]
    pairs = np.unique(one[apart] * count + other[apart])
    return _fewest_to_delete(count, pairs // count, pairs % count)[molecule]


def find_overlapping(
    system: System,
    distance: float,
    *,
    keep_one: bool = True,
    molecules: bool = False,
    atoms: ArrayLike | None = None,
) -> np.ndarray:
    """Atoms closer than a distance to another atom, to delete as few as possible.

    Parameters
    ----------
    system : System
        The atoms and the box.
    distance : float64
        Two atoms closer than this in Å overlap.
    keep_one : bool, optional
        Whether to select only what to delete: as few atoms as a simple rule can,
        those that overlap the fewest being kept first, the lower index before the
        higher, so that no pair that close is left. Otherwise every atom that
        overlaps another. Default True.
    molecules : bool, optional
        Whether to treat the molecules of the ``molecule`` column as wholes: atoms of
        one molecule do not overlap each other, and the atoms of a molecule are
        selected together. Default False.
    atoms : array_like, optional
        A boolean mask or indices of the atoms considered; the others
        neither overlap nor are selected. Default all of them.

    Returns
    -------
    numpy.ndarray
        (N,) bool, the atoms selected.

    Raises
    ------
    InvalidParameter
        If atoms is not a mask or indices of the system, and as
        ``NeighborList.build`` for the distance.
    MissingProperty
        If molecules is asked for and the system has no ``molecule`` column.
    """
    considered = (
        np.ones(len(system), dtype=bool) if atoms is None else _mask(atoms, len(system))
    )
    neighbors = NeighborList.build(system, distance)
    first = np.repeat(np.arange(len(system)), np.diff(neighbors.offsets))
    second = np.asarray(neighbors.j, dtype=np.int64)
    close = considered[first] & considered[second]
    first, second = first[close], second[close]
    if molecules:
        return _overlapping_molecules(_molecules_of(system), first, second, keep_one)
    selected = np.zeros(len(system), dtype=bool)
    if not keep_one:
        selected[second] = True
        return selected
    # too close to its own image, whatever else is deleted
    selected[first[first == second]] = True
    others = ~(selected[first] | selected[second])
    return selected | _fewest_to_delete(len(system), first[others], second[others])
