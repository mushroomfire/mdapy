# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Periodic copies and images of a system."""

from collections.abc import Sequence

import numpy as np
import polars as pl

from mdapy.core.box import Box
from mdapy.core.system import System
from mdapy.core.topology import KINDS, Topology
from mdapy.errors import InvalidParameter, MissingProperty

# Global values that describe the whole of the original system.
_EXTENSIVE_INFO = ("energy", "virial")


def _counts(repeat: int | Sequence[int], box: Box) -> np.ndarray:
    counts = np.asarray(repeat)
    if counts.ndim == 0:
        counts = np.repeat(counts, 3)
    if counts.shape != (3,) or counts.dtype.kind not in "iu":
        raise InvalidParameter(
            f"repeat must be an integer or 3 integers, not {repeat!r}"
        )
    counts = counts.astype(np.int64)
    if counts.min() < 1:
        raise InvalidParameter(f"repeat must be at least 1, not {repeat!r}")
    if np.any((counts > 1) & ~box.pbc):
        raise InvalidParameter(
            "a system is repeated only along periodic directions, "
            f"but repeat is {counts.tolist()} with pbc {box.pbc.tolist()}"
        )
    return counts


def _repeat_topology(topology: Topology, n: int, copies: int) -> Topology:
    tables = {}
    for kind, df in topology.tables().items():
        atoms = df["atoms"].to_numpy().reshape(len(df), KINDS[kind])
        shifted = atoms[None, :, :] + n * np.arange(copies)[:, None, None]
        tables[kind] = pl.concat([df] * copies).with_columns(
            pl.Series(
                "atoms",
                shifted.reshape(-1, KINDS[kind]),
                dtype=pl.Array(pl.Int64, KINDS[kind]),
            )
        )
    return Topology(**tables)


def replicate(system: System, repeat: int | Sequence[int]) -> System:
    """Copies of the system side by side along the lattice vectors.

    Parameters
    ----------
    system : System
        The system to copy.
    repeat : int or sequence of 3 int
        Number of copies along a, b and c, at least 1; more than 1 only along
        periodic directions.

    Returns
    -------
    System
        The copies, one after another with the copy along a changing fastest,
        in a box with the lattice vectors multiplied by ``repeat``. Ids and
        molecules (those above 0) of each copy are offset by the largest of
        the original; topology terms are copied within each copy. With an
        ``image`` column, each atom is placed by its unwrapped position and
        its image flags are those of the new box; without one, the copies
        are translated. ``energy`` and ``virial`` are left out of the info.

    Raises
    ------
    InvalidParameter
        If ``repeat`` is not 1 or 3 positive integers, or is above 1 along a
        non-periodic direction.
    """
    box = system.box
    counts = _counts(repeat, box)
    copies = int(np.prod(counts))
    n = len(system)
    new_box = Box(box.matrix * counts[:, None], origin=box.origin, pbc=box.pbc)
    # copy k is (i, j, l) with k = i + na (j + nb l), as LAMMPS numbers atoms
    l_, j_, i_ = np.meshgrid(*(np.arange(c) for c in counts[::-1]), indexing="ij")
    cells = np.stack([i_.ravel(), j_.ravel(), l_.ravel()], axis=1)
    rows = np.tile(np.arange(n), copies)
    copy = np.repeat(np.arange(copies), n)
    table = system.table[rows]
    positions = system["positions"][rows] + (cells @ box.matrix)[copy]
    columns = {}
    if "image" in table.columns:
        image = table["image"].to_numpy().astype(np.int64)
        positions, flags = new_box.wrap(positions + image @ box.matrix)
        columns["image"] = flags.astype(np.int32)
    columns["positions"] = positions
    if n:
        for name in ("id", "molecule"):
            if name in table.columns:
                values = table[name].to_numpy()
                largest = int(system.table[name].max())
                shifted = values + copy * largest
                columns[name] = (
                    shifted if name == "id" else np.where(values > 0, shifted, values)
                )
    info = {k: v for k, v in system.info.items() if k not in _EXTENSIVE_INFO}
    topology = system.topology
    if topology is not None:
        topology = _repeat_topology(topology, n, copies)
    copied = System(table=table, box=new_box, info=info, topology=topology)
    return copied.with_columns(**columns)


def wrap(system: System) -> System:
    """The atoms moved into the box along the periodic directions.

    Parameters
    ----------
    system : System
        The atoms.

    Returns
    -------
    System
        The same system with fractional coordinates in [0, 1) along periodic
        directions. An ``image`` column is updated so that the unwrapped
        positions do not change.
    """
    positions, moved = system.box.wrap(system["positions"])
    columns = {"positions": positions}
    if "image" in system.table.columns:
        image = system.table["image"].to_numpy().astype(np.int64) + moved
        columns["image"] = image.astype(np.int32)
    return system.with_columns(**columns)


def unwrap(system: System) -> System:
    """The atoms at their unwrapped positions.

    Parameters
    ----------
    system : System
        The atoms, with an ``image`` column.

    Returns
    -------
    System
        The same system with positions plus image flags times the lattice
        vectors, and the image flags set to zero.

    Raises
    ------
    MissingProperty
        If the system has no ``image`` column.
    """
    if "image" not in system.table.columns:
        raise MissingProperty("unwrapping needs the image column")
    image = system.table["image"].to_numpy().astype(np.int64)
    return system.with_columns(
        positions=system["positions"] + image @ system.box.matrix,
        image=np.zeros_like(image, dtype=np.int32),
    )
