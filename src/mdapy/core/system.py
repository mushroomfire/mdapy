# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""The atomic system: one configuration of atoms in a box."""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import ArrayLike

from mdapy.core.box import Box
from mdapy.core.result import Result
from mdapy.core.topology import Topology
from mdapy.elements import ELEMENT_DTYPE
from mdapy.errors import InvalidParameter

__all__ = ["System"]

VECTOR = pl.Array(pl.Float64, 3)

# Standard per-atom columns and their types (see the glossary).
_STANDARD_COLUMNS = {
    "positions": VECTOR,
    "velocities": VECTOR,
    "forces": VECTOR,
    "element": ELEMENT_DTYPE,
    "type": pl.Int32,
    "id": pl.Int64,
    "molecule": pl.Int64,
    "mass": pl.Float64,
    "charge": pl.Float64,
    "image": pl.Array(pl.Int32, 3),
}


def _column(name: str, values: ArrayLike, n: int) -> pl.Series:
    """A table column from an array, a sequence or a single value; arrays
    are copied, so that changing them later does not change a system."""
    if isinstance(values, pl.Series):
        series = values.alias(name)
    else:
        array = np.array(values)
        if array.ndim == 0:
            array = np.repeat(array, n)
        series = pl.Series(name, array)
    if len(series) != n:
        raise InvalidParameter(f"{name} has {len(series)} values for {n} atoms")
    return series


def _normalize_table(table: pl.DataFrame) -> pl.DataFrame:
    """Check the standard columns, cast them to their types, make one chunk."""
    if "positions" not in table.columns:
        raise InvalidParameter(
            "the table needs a 'positions' column of 3 floats per atom; build it "
            "from separate columns with pl.concat_arr('x', 'y', 'z')"
        )
    casts = []
    for name, dtype in _STANDARD_COLUMNS.items():
        if name not in table.columns or table.schema[name] == dtype:
            continue
        inner = dtype.inner if isinstance(dtype, pl.Array) else dtype
        source = table.schema[name]
        source_inner = (
            source.inner if isinstance(source, (pl.Array, pl.List)) else source
        )
        if inner.is_integer() and source_inner.is_float():
            # polars truncates floats when casting to integers; only accept
            # floats that are whole numbers, e.g. types read as 1.0.
            flat = table.select(pl.col(name).explode())[name].drop_nulls()
            values = flat.to_numpy()
            if not np.array_equal(values, np.round(values)):
                raise InvalidParameter(f"column {name!r} must hold integers")
        casts.append(pl.col(name).cast(dtype, strict=True))
    if casts:
        try:
            table = table.with_columns(casts)
        except pl.exceptions.PolarsError as exc:
            raise InvalidParameter(f"invalid standard column: {exc}") from None
    for name in _STANDARD_COLUMNS:
        if name in table.columns and table[name].null_count():
            raise InvalidParameter(f"column {name!r} has missing values")
    if len(table):
        positions = table["positions"].to_numpy()
        if not np.all(np.isfinite(positions)):
            raise InvalidParameter("positions must be finite")
        if "type" in table.columns and table["type"].min() < 1:
            raise InvalidParameter("atom types must be at least 1")
    return table.rechunk()


def _freeze(value: Any) -> Any:
    """A read-only copy of an info value."""
    if isinstance(value, np.ndarray):
        value = value.copy()
        value.flags.writeable = False
        return value
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


class System:
    """One atomic configuration: a table of per-atom properties in a box.

    A system never changes. Arrays given to it are copied. Methods named
    ``with_*`` and :meth:`select` return a new system; columns that are not
    changed are shared, not copied.

    Parameters
    ----------
    positions : array_like, optional
        (N, 3) positions in Å. Give either this or ``table``.
    box : Box or array_like, optional
        Simulation box, or anything :class:`Box` accepts as lattice vectors.
        Default: a non-periodic box around the atoms (:meth:`Box.enclosing`).
    table : polars.DataFrame, optional
        Per-atom table with a ``positions`` column of 3 floats per atom. Give
        either this or ``positions``.
    elements : str or sequence of str, optional
        Chemical symbol of every atom, or one symbol for all atoms. Stored in
        the ``element`` column.
    types : int or sequence of int, optional
        Atom types (from 1) of every atom, or one type for all atoms. Stored
        in the ``type`` column.
    info : mapping, optional
        Global information such as ``energy`` (eV) or ``timestep``.
    topology : Topology, optional
        Bonds, angles, dihedrals and impropers, with atom indices into the
        table.

    Raises
    ------
    InvalidParameter
        If the inputs are inconsistent: neither or both of ``positions`` and
        ``table``, wrong shapes, non-finite positions, unknown chemical
        symbols, types below 1, missing values in standard columns, or
        topology indices beyond the number of atoms.
    """

    __slots__ = ("_table", "_box", "_info", "_topology", "_cache")

    def __init__(
        self,
        positions: ArrayLike | None = None,
        box: Box | ArrayLike | None = None,
        *,
        table: pl.DataFrame | None = None,
        elements: str | Sequence[str] | None = None,
        types: int | Sequence[int] | None = None,
        info: Mapping[str, Any] | None = None,
        topology: Topology | None = None,
    ) -> None:
        if (positions is None) == (table is None):
            raise InvalidParameter("give either positions or table")
        if table is None:
            r = np.array(positions, dtype=np.float64)  # a copy of its own
            if r.size == 0:
                r = r.reshape(0, 3)
            if r.ndim != 2 or r.shape[1] != 3:
                raise InvalidParameter(
                    f"positions must have shape (N, 3), got {r.shape}"
                )
            table = pl.DataFrame({"positions": pl.Series(r, dtype=VECTOR)})
        elif not isinstance(table, pl.DataFrame):
            raise InvalidParameter(
                f"table must be a polars DataFrame, got {type(table)}"
            )
        n = len(table)
        extra = []
        if elements is not None:
            if "element" in table.columns:
                raise InvalidParameter(
                    "elements given, but the table has an 'element' column"
                )
            extra.append(_column("element", elements, n).cast(pl.String))
        if types is not None:
            if "type" in table.columns:
                raise InvalidParameter("types given, but the table has a 'type' column")
            extra.append(_column("type", types, n))
        if extra:
            table = table.with_columns(extra)
        # a table of its own: changing the given table in place does not
        # change the system
        self._table = _normalize_table(table).clone()

        if box is None:
            box = Box.enclosing(self.positions)
        elif not isinstance(box, Box):
            box = Box(box)
        self._box = box

        self._info = MappingProxyType(
            {str(k): _freeze(v) for k, v in (info or {}).items()}
        )

        if topology is not None:
            if not isinstance(topology, Topology):
                raise InvalidParameter(
                    f"topology must be a Topology, got {type(topology)}"
                )
            if topology.max_atom_index() >= n:
                raise InvalidParameter(
                    f"topology refers to atom {topology.max_atom_index()}, "
                    f"but the system has {n} atoms"
                )
        self._topology = topology
        self._cache: dict = {}

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    @property
    def table(self) -> pl.DataFrame:
        """Per-atom properties, one row per atom. A copy that shares the
        columns: changing it in place does not change the system; use the
        ``with_*`` methods for a changed system."""
        return self._table.clone()

    @property
    def box(self) -> Box:
        """The simulation box."""
        return self._box

    @property
    def info(self) -> Mapping[str, Any]:
        """Global information, read-only."""
        return self._info

    @property
    def topology(self) -> Topology | None:
        """Bonds, angles, dihedrals and impropers, or None."""
        return self._topology

    @property
    def positions(self) -> np.ndarray:
        """(N, 3) positions in Å, a read-only view of the table (no copy)."""
        r = self._table["positions"].to_numpy(allow_copy=False)
        r.flags.writeable = False
        return r

    def __len__(self) -> int:
        """Number of atoms."""
        return len(self._table)

    # ------------------------------------------------------------------
    # New systems
    # ------------------------------------------------------------------

    def _replace(self, **changes: Any) -> "System":
        kwargs = {
            "table": self._table,
            "box": self._box,
            "info": self._info,
            "topology": self._topology,
        }
        kwargs.update(changes)
        return System(**kwargs)

    def with_table(self, table: pl.DataFrame) -> "System":
        """New system with a different per-atom table.

        Parameters
        ----------
        table : polars.DataFrame
            New table with the same number of atoms in the same order, since
            the topology refers to rows by position. Use :meth:`select` to
            remove or reorder atoms.

        Raises
        ------
        InvalidParameter
            If the number of rows differs or the table is invalid.
        """
        if not isinstance(table, pl.DataFrame) or len(table) != len(self):
            raise InvalidParameter(
                f"the new table must be a DataFrame with {len(self)} rows"
            )
        return self._replace(table=table)

    def with_columns(self, *columns: Any, **named: Any) -> "System":
        """New system with added or replaced columns.

        Parameters
        ----------
        *columns
            polars expressions or Series, as for ``DataFrame.with_columns``,
            or results of analyses, whose per-atom values are added.
        **named
            Columns by name: polars expressions or Series, NumPy arrays of
            shape (N,) or (N, ...), or single values for all atoms.

        Returns
        -------
        System
            New system; other columns are shared with this one.
        """
        exprs = []
        named = dict(named)
        for column in columns:
            if isinstance(column, Result):
                named.update(column.columns())
            else:
                exprs.append(column)
        for name, value in named.items():
            if isinstance(value, pl.Expr):
                exprs.append(value.alias(name))
            else:
                exprs.append(_column(name, value, len(self)))
        try:
            table = self._table.with_columns(exprs)
        except pl.exceptions.PolarsError as exc:
            raise InvalidParameter(f"cannot add columns: {exc}") from None
        return self._replace(table=table)

    def with_positions(self, positions: ArrayLike) -> "System":
        """New system with different positions.

        Parameters
        ----------
        positions : array_like
            (N, 3) positions in Å.
        """
        r = np.array(positions, dtype=np.float64)  # a copy of its own
        if r.shape != (len(self), 3):
            raise InvalidParameter(
                f"positions must have shape ({len(self)}, 3), got {r.shape}"
            )
        return self.with_columns(positions=pl.Series(r, dtype=VECTOR))

    def with_box(self, box: Box | ArrayLike, scale_positions: bool = False) -> "System":
        """New system in a different box.

        Parameters
        ----------
        box : Box or array_like
            The new box, or lattice vectors for :class:`Box`.
        scale_positions : bool, optional
            If True, keep the fractional coordinates of the atoms, so that
            they deform with the box. If False (default), keep the Cartesian
            positions.
        """
        if not isinstance(box, Box):
            box = Box(box)
        system = self._replace(box=box)
        if scale_positions:
            frac = self._box.to_fractional(self.positions)
            system = system.with_positions(box.to_cartesian(frac))
        return system

    def with_info(self, **values: Any) -> "System":
        """New system with updated global information.

        Parameters
        ----------
        **values
            Entries to add or replace. An entry set to None is removed.
        """
        info = dict(self._info)
        for key, value in values.items():
            if value is None:
                info.pop(key, None)
            else:
                info[key] = value
        return self._replace(info=info)

    def with_topology(self, topology: Topology | None) -> "System":
        """New system with a different topology, or none."""
        return self._replace(topology=topology)

    def select(self, atoms: ArrayLike) -> "System":
        """New system with a subset of the atoms, in the given order.

        Parameters
        ----------
        atoms : array_like
            Boolean mask of length N, or integer indices (negative indices
            count from the end; each atom at most once).

        Returns
        -------
        System
            The selected atoms. Topology terms whose atoms are all selected
            are kept and renumbered; the others are dropped.

        Raises
        ------
        InvalidParameter
            If the mask has the wrong length, or an index is out of range or
            repeated.
        """
        n = len(self)
        selection = np.asarray(atoms)
        if selection.dtype == bool:
            if selection.shape != (n,):
                raise InvalidParameter(
                    f"mask must have {n} entries, got shape {selection.shape}"
                )
            index = np.flatnonzero(selection)
        else:
            if selection.size == 0:
                selection = selection.astype(np.int64)
            if selection.ndim != 1 or selection.dtype.kind not in "iu":
                raise InvalidParameter(
                    "atoms must be a boolean mask or integer indices"
                )
            index = selection.astype(np.int64)
            if index.size and (index.min() < -n or index.max() >= n):
                raise InvalidParameter(f"atom index out of range for {n} atoms")
            index = np.where(index < 0, index + n, index)
            if len(np.unique(index)) != len(index):
                raise InvalidParameter("atom indices must not repeat")
        table = self._table[index]
        topology = self._topology
        if topology is not None:
            mapping = np.full(n, -1, dtype=np.int64)
            mapping[index] = np.arange(len(index))
            topology = topology.remap(mapping)
        return self._replace(table=table, topology=topology)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        parts = [f"{len(self)} atoms"]
        if "element" in self._table.columns and len(self):
            counts = self._table["element"].value_counts(sort=True)
            parts.append(" ".join(f"{e}{c}" for e, c in counts.iter_rows()))
        lengths = ", ".join(f"{x:.4g}" for x in self._box.lengths)
        pbc = "".join("T" if p else "F" for p in self._box.pbc)
        parts.append(f"box lengths ({lengths}) pbc {pbc}")
        if self._topology is not None:
            parts.append(repr(self._topology))
        columns = ", ".join(self._table.columns)
        return f"System({'; '.join(parts)}; columns: {columns})"
