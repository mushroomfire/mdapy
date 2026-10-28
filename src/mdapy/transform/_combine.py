# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The atoms of two systems in one."""

import numpy as np
import polars as pl

from mdapy.core.system import System
from mdapy.core.topology import KINDS, Topology
from mdapy.errors import InvalidParameter
from mdapy.transform._periodic import _EXTENSIVE_INFO


def _zeros(name: str, dtype: pl.DataType, count: int) -> pl.Series:
    """A column of count zeros of a type: the first name of an enumeration,
    such as the element X, an empty text, false; missing values of other
    types, such as lists."""
    if isinstance(dtype, pl.Enum):
        return pl.Series(name, [dtype.categories[0]] * count, dtype=dtype)
    if isinstance(dtype, pl.Array):
        return pl.Series(name, np.zeros((count, dtype.size))).cast(dtype)
    if dtype == pl.String:
        return pl.Series(name, [""] * count)
    if dtype == pl.Boolean:
        return pl.Series(name, np.zeros(count, dtype=bool))
    if dtype.is_numeric():
        return pl.Series(name, np.zeros(count)).cast(dtype)
    return pl.Series(name, [None] * count, dtype=dtype)


def _stack(first: pl.DataFrame, second: pl.DataFrame, what: str) -> pl.DataFrame:
    """The rows of second after those of first, with the columns of either,
    zeros where one has no column."""
    names = first.columns + [c for c in second.columns if c not in first.columns]
    for c in names:
        a, b = first.schema.get(c), second.schema.get(c)
        if a is not None and b is not None and a != b:
            if not (a.is_numeric() and b.is_numeric()):
                raise InvalidParameter(
                    f"{what} column {c!r} is {a} in one and {b} in the other"
                )
    parts = []
    for table, other in ((first, second), (second, first)):
        parts.append(
            table.with_columns(
                _zeros(c, other.schema[c], len(table))
                for c in names
                if c not in table.columns
            ).select(names)
        )
    return pl.concat(parts, how="vertical_relaxed")


def combine(system: System, other: System) -> System:
    """The atoms of other added after those of system, as OVITO's combine
    datasets.

    Parameters
    ----------
    system : System
        The atoms and the box that are kept.
    other : System
        The atoms added, at their positions; its box is not used.

    Returns
    -------
    System
        The atoms of both in the box of system, with the columns of either:
        zeros where one has no column (the element X for ``element``). Where
        system has an ``id`` column, the added atoms are numbered on from
        its largest id in their order; molecules of other above 0 are offset
        by the largest molecule of system. The topology terms of both, those
        of other moved to its atoms. The info of system, without ``energy``
        and ``virial``.

    Raises
    ------
    InvalidParameter
        If a column has types in the two that cannot be joined, such as
        text in one and numbers in the other.
    """
    n, m = len(system), len(other)
    first, second = system.table, other.table
    if "id" in first.columns:
        start = int(first["id"].max()) if n else 0
        second = second.with_columns(
            pl.Series("id", np.arange(start + 1, start + m + 1), dtype=pl.Int64)
        )
    if "molecule" in first.columns and "molecule" in second.columns and n:
        largest = int(first["molecule"].max())
        second = second.with_columns(
            pl.when(pl.col("molecule") > 0)
            .then(pl.col("molecule") + largest)
            .otherwise(pl.col("molecule"))
        )
    table = _stack(first, second, "the")
    topology = None
    if system.topology is not None or other.topology is not None:
        tables = {}
        for kind in KINDS:
            mine = system.topology.tables()[kind] if system.topology else None
            theirs = other.topology.tables()[kind] if other.topology else None
            if theirs is not None:
                atoms = theirs["atoms"].to_numpy() + n
                theirs = theirs.with_columns(
                    pl.Series("atoms", atoms, dtype=pl.Array(pl.Int64, KINDS[kind]))
                )
            parts = [t for t in (mine, theirs) if t is not None]
            tables[kind] = parts[0] if len(parts) == 1 else _stack(*parts, kind)
        topology = Topology(**tables)
    info = {k: v for k, v in system.info.items() if k not in _EXTENSIVE_INFO}
    return System(table=table, box=system.box, info=info, topology=topology)
