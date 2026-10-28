# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""A selection of atoms grown by their neighbors."""

import operator

import numpy as np
from numpy.typing import ArrayLike

from mdapy.core._atoms import _mask
from mdapy.core.system import System
from mdapy.errors import InvalidParameter, MissingProperty
from mdapy.neighbor._neighbor_list import NeighborList


def expand_selection(
    system: System,
    atoms: ArrayLike,
    *,
    cutoff: float | None = None,
    num_neighbors: int | None = None,
    bonds: bool = False,
    iterations: int = 1,
) -> np.ndarray:
    """Atoms and their neighbors: a selection grown by the atoms near it.

    Parameters
    ----------
    system : System
        The atoms and the box.
    atoms : array_like
        A boolean mask or indices of the atoms selected.
    cutoff : float, optional
        Add the atoms closer than this in Å to a selected atom.
    num_neighbors : int, optional
        Add the num_neighbors nearest neighbors of each selected atom.
    bonds : bool, optional
        Add the atoms bonded to a selected atom, by the bonds of the
        topology.
    iterations : int, optional
        How many times the selection grows, each time by the neighbors of
        all the atoms selected so far. Default 1.

    Returns
    -------
    numpy.ndarray
        (N,) bool, the atoms selected and those added.

    Raises
    ------
    InvalidParameter
        If not exactly one of cutoff, num_neighbors and bonds is given, if
        iterations is negative or not an integer, if atoms is not a mask or
        indices of the system, and as :meth:`NeighborList.build`.
    MissingProperty
        With bonds, if the system has no bonds.
    """
    given = (cutoff is not None) + (num_neighbors is not None) + bool(bonds)
    if given != 1:
        raise InvalidParameter("give exactly one of cutoff, num_neighbors and bonds")
    try:
        iterations = operator.index(iterations)
    except TypeError:
        raise InvalidParameter(
            f"iterations must be an integer, not {iterations!r}"
        ) from None
    if iterations < 0:
        raise InvalidParameter(f"iterations must not be negative, not {iterations}")
    selected = _mask(atoms, len(system)).copy()
    if bonds:
        topology = system.topology
        if topology is None or topology.bonds is None or not len(topology.bonds):
            raise MissingProperty("the system has no bonds")
        pairs = np.asarray(topology.bonds["atoms"].to_list(), dtype=np.int64).reshape(
            -1, 2
        )
        first = np.concatenate([pairs[:, 0], pairs[:, 1]])
        second = np.concatenate([pairs[:, 1], pairs[:, 0]])
    else:
        neighbors = NeighborList.build(system, cutoff, num_neighbors=num_neighbors)
        first = np.repeat(np.arange(len(system)), np.diff(neighbors.offsets))
        second = np.asarray(neighbors.j, dtype=np.int64)
    for _ in range(iterations):
        grown = selected.copy()
        grown[second[selected[first]]] = True
        if (grown == selected).all():
            break
        selected = grown
    return selected
