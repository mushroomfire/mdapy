# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""The atomic system: one configuration of atoms in a box."""

from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import ArrayLike

from mdapy.core import _conditions
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


def _normalize_table(
    table: pl.DataFrame, columns: Iterable[str] | None = None
) -> pl.DataFrame:
    """Check the standard columns, cast them to their types, make one chunk;
    with columns, only the standard ones among them, the others being known
    to be right."""
    if "positions" not in table.columns:
        raise InvalidParameter(
            "the table needs a 'positions' column of 3 floats per atom; build it "
            "from separate columns with pl.concat_arr('x', 'y', 'z')"
        )
    checked = _STANDARD_COLUMNS if columns is None else set(columns)
    casts = []
    for name, dtype in _STANDARD_COLUMNS.items():
        if name not in checked or name not in table.columns:
            continue
        if table.schema[name] == dtype:
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
        if name in checked and name in table.columns and table[name].null_count():
            raise InvalidParameter(f"column {name!r} has missing values")
    if len(table):
        if "positions" in checked:
            positions = table["positions"].to_numpy()
            if not np.all(np.isfinite(positions)):
                raise InvalidParameter("positions must be finite")
        if "type" in checked and "type" in table.columns and table["type"].min() < 1:
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


def _set(system: "System", **values: Any) -> None:
    """Sets the private attributes of a system being made."""
    for name, value in values.items():
        object.__setattr__(system, f"_System__{name}", value)


def _changed(expr: Any) -> str | None:
    """The column an expression of with_columns makes, or None when that
    cannot be told."""
    if isinstance(expr, pl.Series):
        return expr.name
    try:
        if expr.meta.has_multiple_outputs():
            return None
        return expr.meta.output_name()
    except (AttributeError, pl.exceptions.PolarsError):
        return None


def _format(value: Any) -> str:
    """An info value for the summary of a system."""
    if isinstance(value, np.ndarray):
        return f" {value.shape}" if value.ndim else f" = {value.item():.6g}"
    if isinstance(value, bool) or value is None:
        return f" = {value}"
    if isinstance(value, (int, np.integer)):
        return f" = {value}"
    if isinstance(value, (float, np.floating)):
        return f" = {value:.6g}"
    if isinstance(value, str):
        return f" = {value!r}" if len(value) <= 24 else f" = {value[:21]!r}..."
    if isinstance(value, Mapping):
        return f" ({len(value)} entries)"
    if isinstance(value, tuple):
        return f" ({len(value)} values)"
    return f" ({type(value).__name__})"


def _wrapped(label: str, items: list[str], width: int = 88) -> list[str]:
    """Items after a label, separated by commas, in lines of at most width."""
    lines, line = [], f"  {label:<10}"
    indent = " " * len(line)
    for k, item in enumerate(items):
        text = item + ("," if k < len(items) - 1 else "")
        if len(line) + len(text) + 1 > width and line.strip() != label.strip():
            lines.append(line.rstrip())
            line = indent
        line += ("" if line.endswith(" ") else " ") + text
    lines.append(line.rstrip())
    return lines


class System:
    """One atomic configuration: a table of per-atom properties in a box.

    A system never changes. Arrays given to it are copied. Methods named
    ``with_*``, :meth:`filter`, :meth:`gather` and :meth:`sort` return a new
    system; columns that are not changed are shared, not copied.

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

    __slots__ = ("__table", "__box", "__info", "__topology")

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
        table = _normalize_table(table).clone()
        if box is None:
            box = Box.enclosing(table["positions"].to_numpy())
        _set(
            self,
            table=table,
            box=self.__checked_box(box),
            info=self.__frozen_info(info),
            topology=self.__checked_topology(topology, n),
        )

    @staticmethod
    def __checked_box(box: Box | ArrayLike) -> Box:
        return box if isinstance(box, Box) else Box(box)

    @staticmethod
    def __frozen_info(info: Mapping[str, Any] | None) -> Mapping[str, Any]:
        return MappingProxyType({str(k): _freeze(v) for k, v in (info or {}).items()})

    @staticmethod
    def __checked_topology(topology: Topology | None, n: int) -> Topology | None:
        if topology is None:
            return None
        if not isinstance(topology, Topology):
            raise InvalidParameter(f"topology must be a Topology, got {type(topology)}")
        if topology.max_atom_index() >= n:
            raise InvalidParameter(
                f"topology refers to atom {topology.max_atom_index()}, "
                f"but the system has {n} atoms"
            )
        return topology

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(
            "a System cannot be changed; the with_* methods, filter, gather and "
            "sort make a new one"
        )

    def __delattr__(self, name: str) -> None:
        raise AttributeError("a System cannot be changed")

    def __copy__(self) -> "System":
        # nothing in a system ever changes, so a copy would only cost memory
        return self

    def __deepcopy__(self, memo: dict) -> "System":
        return self

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    @property
    def table(self) -> pl.DataFrame:
        """Per-atom properties, one row per atom. A copy that shares the
        columns: changing it in place does not change the system; use the
        ``with_*`` methods for a changed system."""
        return self.__table.clone()

    @property
    def box(self) -> Box:
        """The simulation box."""
        return self.__box

    @property
    def info(self) -> Mapping[str, Any]:
        """Global information, read-only."""
        return self.__info

    @property
    def topology(self) -> Topology | None:
        """Bonds, angles, dihedrals and impropers, or None."""
        return self.__topology

    @property
    def positions(self) -> np.ndarray:
        """(N, 3) positions in Å, a read-only view of the table (no copy)."""
        r = self.__table["positions"].to_numpy(allow_copy=False)
        r.flags.writeable = False
        return r

    def __len__(self) -> int:
        """Number of atoms."""
        return len(self.__table)

    # ------------------------------------------------------------------
    # New systems
    # ------------------------------------------------------------------

    def __derived(
        self,
        *,
        table: pl.DataFrame | None = None,
        changed: Iterable[str] | None = (),
        box: Box | None = None,
        info: Mapping[str, Any] | None = None,
        topology: Any = ...,
    ) -> "System":
        """A system made from this one: the table, with only the columns
        changed checked (all for None), and the parts given replacing those
        of this system."""
        system = object.__new__(System)
        _set(
            system,
            table=self.__table if table is None else _normalize_table(table, changed),
            box=self.__box if box is None else box,
            info=self.__info if info is None else info,
            topology=self.__topology if topology is ... else topology,
        )
        return system

    def with_table(self, table: pl.DataFrame) -> "System":
        """New system with a different per-atom table.

        Parameters
        ----------
        table : polars.DataFrame
            New table with the same number of atoms in the same order, since
            the topology refers to rows by position. Use :meth:`sort` or
            :meth:`gather` to reorder atoms and :meth:`filter` to remove them.

        Raises
        ------
        InvalidParameter
            If the number of rows differs or the table is invalid.
        """
        if not isinstance(table, pl.DataFrame) or len(table) != len(self):
            raise InvalidParameter(
                f"the new table must be a DataFrame with {len(self)} rows"
            )
        return self.__derived(table=table.clone(), changed=None)

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
        made = [_changed(expr) for expr in exprs]
        try:
            table = self.__table.with_columns(exprs)
        except pl.exceptions.PolarsError as exc:
            raise InvalidParameter(f"cannot add columns: {exc}") from None
        return self.__derived(table=table, changed=None if None in made else made)

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
        box = self.__checked_box(box)
        system = self.__derived(box=box)
        if scale_positions:
            frac = self.__box.to_fractional(self.positions)
            system = system.with_positions(box.to_cartesian(frac))
        return system

    def with_info(self, **values: Any) -> "System":
        """New system with updated global information.

        Parameters
        ----------
        **values
            Entries to add or replace. An entry set to None is removed.
        """
        info = dict(self.__info)
        for key, value in values.items():
            if value is None:
                info.pop(key, None)
            else:
                info[key] = value
        return self.__derived(info=self.__frozen_info(info))

    def with_topology(self, topology: Topology | None) -> "System":
        """New system with a different topology, or none."""
        return self.__derived(topology=self.__checked_topology(topology, len(self)))

    # ------------------------------------------------------------------
    # Atoms chosen and ordered
    # ------------------------------------------------------------------

    def filter(self, *conditions: Any, **equal: Any) -> "System":
        """New system with the atoms that meet every condition, in their order.

        Parameters
        ----------
        *conditions
            Each a boolean mask of length N, a polars expression, or a
            condition written as a Python expression over the columns, such
            as ``"positions.x > 5 and element == 'Cu'"``, as described in
            Selecting and ordering atoms in the documentation.
        **equal
            Columns and the value each must equal, such as ``element="Cu"``.

        Returns
        -------
        System
            The atoms kept. Topology terms whose atoms are all kept are kept
            and renumbered; the others are dropped.

        Raises
        ------
        InvalidParameter
            For a mask of the wrong length, an unknown column or component, a
            condition that cannot be read or does not give true or false.
        """
        n, schema = len(self), self.__table.schema
        keep = np.ones(n, dtype=bool)
        exprs = []
        for condition in conditions:
            if isinstance(condition, str):
                exprs.append((condition, _conditions.condition(condition, schema)))
            elif isinstance(condition, pl.Expr):
                exprs.append((str(condition), condition))
            else:
                mask = np.asarray(condition)
                if mask.dtype != bool or mask.shape != (n,):
                    raise InvalidParameter(
                        f"a mask must hold {n} booleans, got {mask.dtype} of "
                        f"shape {mask.shape}"
                    )
                keep &= mask
        for name, value in equal.items():
            exprs.append(
                (
                    f"{name} == {value!r}",
                    _conditions.column(name, None, schema) == value,
                )
            )
        for text, expr in exprs:
            try:
                # with_columns spreads a constant over every atom
                found = self.__table.with_columns(expr.alias("_keep"))["_keep"]
            except pl.exceptions.PolarsError as exc:
                raise InvalidParameter(f"cannot evaluate {text!r}: {exc}") from None
            if found.dtype != pl.Boolean or len(found) != n:
                raise InvalidParameter(
                    f"{text!r} does not give true or false for each atom"
                )
            keep &= found.fill_null(False).to_numpy()
        return self.__rows(np.flatnonzero(keep))

    def gather(self, atoms: ArrayLike) -> "System":
        """New system with the atoms of the given indices, in that order.

        Parameters
        ----------
        atoms : array_like
            Integer indices; negative ones count from the end; each atom at
            most once.

        Returns
        -------
        System
            The atoms, in the given order. Topology terms whose atoms are all
            given are kept and renumbered; the others are dropped.

        Raises
        ------
        InvalidParameter
            For indices that are not integers, out of range or repeated.
        """
        n = len(self)
        index = np.asarray(atoms)
        if index.size == 0:
            index = index.astype(np.int64)
        if index.ndim != 1 or index.dtype.kind not in "iu":
            raise InvalidParameter(
                "atoms must be integer indices; filter takes a boolean mask"
            )
        index = index.astype(np.int64)
        if index.size and (index.min() < -n or index.max() >= n):
            raise InvalidParameter(f"atom index out of range for {n} atoms")
        index = np.where(index < 0, index + n, index)
        if len(np.unique(index)) != len(index):
            raise InvalidParameter("atom indices must not repeat")
        return self.__rows(index)

    def sort(
        self, *by: str | pl.Expr, descending: bool | Sequence[bool] = False
    ) -> "System":
        """New system with the atoms sorted.

        Parameters
        ----------
        *by
            Columns, components written ``column.component`` such as
            ``"positions.z"``, or polars expressions; atoms equal in the
            first are sorted by the second, and so on. Atoms equal in all
            keep their order. Elements sort by atomic number.
        descending : bool or sequence of bool, optional
            Sort from the largest, for all keys or for each. Default False.

        Returns
        -------
        System
            The atoms in sorted order, with the topology renumbered.

        Raises
        ------
        InvalidParameter
            Without a key, or for an unknown column or component.
        """
        if not by:
            raise InvalidParameter("give a column to sort by")
        schema = self.__table.schema
        keys = [
            (_conditions.key(k, schema) if isinstance(k, str) else k).alias(f"_key{i}")
            for i, k in enumerate(by)
        ]
        try:
            order = (
                self.__table.select(pl.int_range(pl.len()).alias("_row"), *keys)
                .sort(
                    [f"_key{i}" for i in range(len(keys))],
                    descending=descending,
                    maintain_order=True,
                )["_row"]
                .to_numpy()
            )
        except (pl.exceptions.PolarsError, ValueError) as exc:
            raise InvalidParameter(f"cannot sort: {exc}") from None
        return self.__rows(order)

    def __rows(self, index: np.ndarray) -> "System":
        """New system with the rows of index, of a table already checked."""
        topology = self.__topology
        if topology is not None:
            mapping = np.full(len(self), -1, dtype=np.int64)
            mapping[index] = np.arange(len(index))
            topology = topology.remap(mapping)
        return self.__derived(table=self.__table[index], topology=topology)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n = len(self)
        head = f"System: {n} atom{'' if n == 1 else 's'}"
        if "element" in self.__table.columns and n:
            counts = (
                self.__table.group_by("element")
                .len()
                .sort(["len", "element"], descending=[True, False])
            )
            formula = " ".join(
                f"{e}{c if c > 1 else ''}" for e, c in counts.iter_rows()
            )
            head += f" ({formula})"
        lines = [head]
        box = self.__box
        if box.is_orthogonal:
            shape = " × ".join(f"{x:.4g}" for x in box.lengths) + " Å"
            if np.any(box.origin != 0):
                shape += " at (" + ", ".join(f"{x:.4g}" for x in box.origin) + ")"
            shape += ", orthogonal"
        else:
            a, b, c = box.lengths
            alpha, beta, gamma = box.angles
            shape = (
                f"a {a:.4g}, b {b:.4g}, c {c:.4g} Å; "
                f"α {alpha:.4g}°, β {beta:.4g}°, γ {gamma:.4g}°"
            )
        periodic = [axis for axis, p in zip("xyz", box.pbc) if p]
        shape += "; " if not box.is_orthogonal else ", "
        shape += "periodic " + " ".join(periodic) if periodic else "not periodic"
        lines.append(f"  {'box:':<10}{shape}")
        columns = [
            f"{name} ({dtype.size})" if isinstance(dtype, pl.Array) else name
            for name, dtype in self.__table.schema.items()
        ]
        lines += _wrapped("columns:", columns)
        if self.__info:
            lines += _wrapped(
                "info:", [f"{k}{_format(v)}" for k, v in self.__info.items()]
            )
        if self.__topology is not None:
            terms = []
            for kind, table in self.__topology.tables().items():
                if len(table):
                    name = kind if len(table) > 1 else kind[:-1]
                    terms.append(f"{len(table)} {name}")
            lines += _wrapped("topology:", terms or ["none"])
        return "\n".join(lines)
