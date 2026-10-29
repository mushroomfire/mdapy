# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Atoms too close to another."""

import numpy as np
from numpy.typing import ArrayLike

from mdapy.core._atoms import _mask
from mdapy.core.system import System
from mdapy.neighbor._neighbor_list import NeighborList


def find_overlapping(
    system: System,
    distance: float,
    *,
    keep_one: bool = True,
    atoms: ArrayLike | None = None,
) -> np.ndarray:
    """Atoms closer than a distance to another atom.

    Parameters
    ----------
    system : System
        The atoms and the box.
    distance : float
        Two atoms closer than this in Å overlap.
    keep_one : bool, optional
        Whether to select only the atom of the higher index of each pair
        that overlaps, so that deleting the selection leaves no pair that
        close and keeps one atom of each group of atoms that all overlap,
        or both. Default True.
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
        :meth:`NeighborList.build` for the distance.
    """
    considered = (
        np.ones(len(system), dtype=bool) if atoms is None else _mask(atoms, len(system))
    )
    neighbors = NeighborList.build(system, distance)
    first = np.repeat(np.arange(len(system)), np.diff(neighbors.offsets))
    second = np.asarray(neighbors.j, dtype=np.int64)
    close = considered[first] & considered[second]
    if keep_one:
        close &= second > first
    selected = np.zeros(len(system), dtype=bool)
    selected[second[close]] = True
    return selected
