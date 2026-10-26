# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""LAMMPS text dump files of the atom and custom styles, as LAMMPS
stable_30Sep2026 writes them."""

import math
import re
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, BinaryIO

import numpy as np
import polars as pl

from mdapy import units as _units
from mdapy._core import io as _native
from mdapy.core.box import Box
from mdapy.core.system import System
from mdapy.elements import ELEMENT_DTYPE, SYMBOLS
from mdapy.errors import (
    FileFormatError,
    InvalidParameter,
    UnsupportedFormat,
)
from mdapy.io._files import Source
from mdapy.io._lammps_data import _atom_types
from mdapy.io._registry import Format, FramePosition, register
from mdapy.io._text import TextReader, categorize, strings, write_rows

_UNITS = ("metal", "real")
# Coordinates, in order of preference: wrapped, scaled, unwrapped, both.
_POSITIONS = (
    ("x", "y", "z"),
    ("xs", "ys", "zs"),
    ("xu", "yu", "zu"),
    ("xsu", "ysu", "zsu"),
)
_SCALED = {("xs", "ys", "zs"), ("xsu", "ysu", "zsu")}
_VECTORS = {
    "image": ("ix", "iy", "iz"),
    "velocities": ("vx", "vy", "vz"),
    "forces": ("fx", "fy", "fz"),
}
# Dump column -> mdapy column, for single values.
_SCALARS = {
    "id": "id",
    "type": "type",
    "mol": "molecule",
    "q": "charge",
    "mass": "mass",
    "element": "element",
    "typelabel": "type_label",
}
_INTEGERS = frozenset({"id", "type", "mol", "ix", "iy", "iz", "proc", "procp1"})
_STRINGS = frozenset({"element", "typelabel"})
_INDEXED = re.compile(r"(.+)\[([0-9]+)\]\Z")


def _kind(name: str) -> str:
    """Kernel kind of a dump column."""
    if name in _STRINGS:
        return "s"
    if name in _INTEGERS or name.startswith(("i_", "i2_")):
        return "i"
    return "f"


# ------------------------------------------------------------------ reading


class _Frame:
    """The header of one frame."""

    def __init__(self):
        self.units: str | None = None
        self.time_line: int | None = None  # line of the TIME value
        self.time_offset = 0
        self.timestep: int | None = None
        self.count: int | None = None
        self.box: Box | None = None
        self.columns: list[str] = []
        self.atoms_line = 0  # line of the first atom


def _check_text(source: Source) -> None:
    head = bytes(source.buffer[:256])
    if b"\0" in head:
        raise UnsupportedFormat(
            f"{source.name} is a binary dump; mdapy reads text dumps (convert it with "
            "the binary2txt tool of LAMMPS)"
        )


def _numbers(reader: TextReader, count: int) -> list[float]:
    words = reader.readline().split()
    try:
        values = [float(w) for w in words]
    except ValueError:
        values = []
    if len(values) != count:
        raise reader.error(f"expected {count} numbers, found {' '.join(words)!r}")
    return values


def _integer(reader: TextReader) -> int:
    text = reader.readline().strip()
    if not re.fullmatch(r"-?[0-9]+", text):
        raise reader.error(f"expected an integer, found {text[:40]!r}")
    return int(text)


def _box(reader: TextReader, words: list[str]) -> Box:
    """Box of an ITEM: BOX BOUNDS line (given as words after BOX BOUNDS)."""
    if words[:2] == ["abc", "origin"]:
        shape, flags = "general", words[2:]
    elif words[:3] == ["xy", "xz", "yz"]:
        shape, flags = "triclinic", words[3:]
    else:
        shape, flags = "orthogonal", words
    if flags and (len(flags) != 3 or any(len(f) != 2 for f in flags)):
        raise reader.error(f"invalid boundary flags {' '.join(flags)!r}")
    pbc = [f[0] == "p" for f in flags] if flags else True
    rows = [
        _numbers(reader, {"general": 4, "triclinic": 3}.get(shape, 2)) for _ in "xyz"
    ]
    try:
        if shape == "general":
            return Box([r[:3] for r in rows], origin=[r[3] for r in rows], pbc=pbc)
        lo = [r[0] for r in rows]
        hi = [r[1] for r in rows]
        if shape == "orthogonal":
            return Box.from_lammps(lo, hi, pbc=pbc)
        xy, xz, yz = rows[0][2], rows[1][2], rows[2][2]
        # LAMMPS writes the bounding box of a triclinic box.
        lo[0] -= min(0.0, xy, xz, xy + xz)
        hi[0] -= max(0.0, xy, xz, xy + xz)
        lo[1] -= min(0.0, yz)
        hi[1] -= max(0.0, yz)
        return Box.from_lammps(lo, hi, (xy, xz, yz), pbc=pbc)
    except InvalidParameter as e:
        raise reader.error(str(e)) from None


def _header(reader: TextReader) -> _Frame:
    frame = _Frame()
    while True:
        text = reader.readline().strip()
        if not text.startswith("ITEM:"):
            raise reader.error(f"expected an ITEM line, found {text[:40]!r}")
        item = text[len("ITEM:") :].split()
        if item == ["UNITS"]:
            frame.units = reader.readline().strip()
        elif item == ["TIME"]:
            frame.time_offset, frame.time_line = reader.offset, reader.line
            reader.readline()
        elif item == ["TIMESTEP"]:
            frame.timestep = _integer(reader)
        elif item == ["NUMBER", "OF", "ATOMS"]:
            frame.count = _integer(reader)
            if frame.count < 0:
                raise reader.error("the number of atoms is negative")
        elif item[:2] == ["BOX", "BOUNDS"]:
            frame.box = _box(reader, item[2:])
        elif item[:1] == ["ATOMS"]:
            if frame.count is None or frame.box is None:
                raise reader.error(
                    "ITEM: ATOMS comes before the number of atoms or the box"
                )
            frame.columns = item[1:]
            if len(set(frame.columns)) != len(frame.columns):
                raise reader.error("a column name is repeated")
            frame.atoms_line = reader.line + 1
            return frame
        elif item == ["NUMBER", "OF", "ENTRIES"]:
            raise UnsupportedFormat(
                "dump local files are not supported; mdapy reads dumps of the atom "
                "and custom styles"
            )
        else:
            raise reader.error(f"unknown item {' '.join(item)!r}")


def _skip_blank(reader: TextReader) -> None:
    while not reader.at_end():
        offset, line = reader.offset, reader.line
        if reader.readline().strip():
            reader.offset, reader.line = offset, line
            return


def _scan(source: Source) -> Iterator[FramePosition]:
    _check_text(source)
    reader = source.reader()
    while True:
        _skip_blank(reader)
        if reader.at_end():
            return
        position = (reader.offset, reader.line)
        frame = _header(reader)
        try:
            reader.skip(frame.count)
        except FileFormatError as e:
            raise reader.error(
                f"the frame has {frame.count} atoms, but the file ends after "
                f"{e.line - frame.atoms_line}",
                e.line,
            ) from None
        yield position


def _first_units(source: Source) -> str | None:
    """Units of the first frame: LAMMPS writes ITEM: UNITS only there."""
    reader = source.reader()
    _skip_blank(reader)
    if not reader.at_end() and reader.readline().strip() == "ITEM: UNITS":
        return reader.readline().strip()
    return None


def _plan_columns(columns: list[str]) -> dict[str, Any]:
    """Which dump columns make which mdapy columns."""
    present = set(columns)
    vectors: dict[str, tuple[str, ...]] = {}
    for names in _POSITIONS:
        if present.issuperset(names):
            vectors["positions"] = names
            break
    for target, names in _VECTORS.items():
        if present.issuperset(names):
            vectors[target] = names
    used = {name for names in vectors.values() for name in names}
    # name[1] ... name[k] become one array column.
    groups: dict[str, list[str]] = {}
    for name in columns:
        match = _INDEXED.match(name)
        if match and name not in used:
            groups.setdefault(match.group(1), []).append(name)
    arrays = {}
    for base, names in groups.items():
        expected = [f"{base}[{i}]" for i in range(1, len(names) + 1)]
        kinds = {_kind(n) for n in names}
        if names == expected and len(kinds) == 1 and base not in present:
            arrays[base] = tuple(names)
            used.update(names)
    return {"vectors": vectors, "arrays": arrays, "used": used}


def _read(
    source: Source,
    position: FramePosition,
    *,
    units: str | None = None,
    elements: Sequence[str] | None = None,
) -> System:
    if units is not None and units not in _UNITS:
        raise InvalidParameter(f"units must be 'metal' or 'real', not {units!r}")
    _check_text(source)
    reader = source.reader(*position)
    _skip_blank(reader)
    frame = _header(reader)
    file_units = units or frame.units or _first_units(source) or "metal"
    if file_units not in _UNITS:
        raise UnsupportedFormat(
            f"the dump is in {file_units} units; mdapy reads metal and real units "
            "(pass units='metal' to read the numbers unchanged)"
        )
    real = file_units == "real"
    plan = _plan_columns(frame.columns)
    velocity_names = set(plan["vectors"].get("velocities", ()))
    spec = [
        (_kind(name), 1, 3 if real and name in velocity_names else 0)
        for name in frame.columns
    ]
    after_header = (reader.offset, reader.line)
    arrays = dict(zip(frame.columns, reader.table(frame.count, spec)))

    info: dict[str, Any] = {}
    if frame.timestep is not None:
        info["timestep"] = frame.timestep
    if frame.time_line is not None:
        reader.offset, reader.line = frame.time_offset, frame.time_line
        # TIME is in fs in real units; the shift converts it to ps exactly.
        (value,) = reader.table(1, [("f", 1, -3 if real else 0)])
        info["time"] = float(value[0])
        reader.offset, reader.line = after_header
    table = _table(reader, frame, plan, arrays, real, elements)
    try:
        return System(table=table, box=frame.box, info=info)
    except InvalidParameter as e:
        raise reader.error(str(e), frame.atoms_line - 1) from None


def _table(
    reader: TextReader,
    frame: _Frame,
    plan: dict[str, Any],
    arrays: dict[str, Any],
    real: bool,
    elements: Sequence[str] | None,
) -> pl.DataFrame:
    vectors, groups = plan["vectors"], plan["arrays"]
    first_of = {names[0]: target for target, names in vectors.items()}
    first_of.update({names[0]: base for base, names in groups.items()})
    series: list[pl.Series] = []
    for name in frame.columns:
        if name in first_of:
            target = first_of[name]
            names = vectors.get(target) or groups[target]
            values = np.column_stack([arrays[n] for n in names])
            if target == "positions" and names in _SCALED:
                values = frame.box.to_cartesian(values)
            elif target == "forces" and real:
                values = values * _units.KCAL_PER_MOL_TO_EV
            series.append(pl.Series(target, values))
        elif name in plan["used"]:
            continue
        elif name == "element":
            codes, categories = arrays[name]
            for code, symbol in enumerate(categories):
                if symbol not in SYMBOLS:
                    row = int(np.flatnonzero(codes == code)[0])
                    raise reader.error(
                        f"unknown element {symbol!r}", frame.atoms_line + row
                    )
            series.append(strings(codes, categories, "element", ELEMENT_DTYPE))
        elif _kind(name) == "s":
            series.append(strings(*arrays[name], _SCALARS[name]))
        else:
            series.append(pl.Series(_SCALARS.get(name, name), arrays[name]))
    table = pl.DataFrame(series)
    if elements is not None and "element" not in table.columns:
        if "type" not in table.columns:
            raise InvalidParameter("elements= needs a type column in the dump")
        elements = list(elements)
        types = table["type"].to_numpy()
        if len(types) and types.max() > len(elements):
            raise InvalidParameter(
                f"elements= names {len(elements)} elements, but the dump has type "
                f"{types.max()}"
            )
        try:
            # index 0 is never used: types start at 1
            symbols = pl.Series(["X"] + elements, dtype=pl.String).cast(ELEMENT_DTYPE)
        except pl.exceptions.InvalidOperationError:
            raise InvalidParameter(
                f"elements= has an unknown symbol: {elements}"
            ) from None
        table = table.with_columns(symbols.gather(types).alias("element"))
    return table


# ------------------------------------------------------------------ writing


def _bounds_lines(box: Box, triclinic: str, vectors: list, image):
    """BOX BOUNDS lines and positions, other vectors and image flags in the
    frame of the written box."""
    m, origin = box.matrix, box.origin
    flags = " ".join("pp" if p else "ff" for p in box.pbc)
    restricted = m[0, 1] == m[0, 2] == m[1, 2] == 0 and np.all(np.diag(m) > 0)
    if triclinic == "restricted" and not restricted:
        lammps = box.to_lammps()
        vectors = [None if v is None else v @ lammps.rotation for v in vectors]
        if np.linalg.det(m) < 0:
            pbc = box.pbc[[0, 2, 1]]
            flags = " ".join("pp" if p else "ff" for p in pbc)
            if image is not None:
                image = image[:, [0, 2, 1]]
        lo, hi, tilt = lammps.lo, lammps.hi, lammps.tilt
    elif triclinic != "general" and restricted:
        lo, hi = origin, origin + np.diag(m)
        tilt = np.array([m[1, 0], m[2, 0], m[2, 1]])
    else:
        if np.linalg.det(m) < 0:  # keep the edge vectors right-handed
            m = m[[0, 2, 1]]
            flags = " ".join("pp" if p else "ff" for p in box.pbc[[0, 2, 1]])
            if image is not None:
                image = image[:, [0, 2, 1]]
        lines = [f"ITEM: BOX BOUNDS abc origin {flags}"]
        lines += [
            " ".join(repr(float(x)) for x in (*m[i], origin[i])) for i in range(3)
        ]
        return lines, vectors, image
    if not np.any(tilt):
        lines = [f"ITEM: BOX BOUNDS {flags}"] + [
            f"{float(lo[i])!r} {float(hi[i])!r}" for i in range(3)
        ]
        return lines, vectors, image
    xy, xz, yz = (float(t) for t in tilt)
    bounds_lo = [
        lo[0] + min(0.0, xy, xz, xy + xz),
        lo[1] + min(0.0, yz),
        lo[2],
    ]
    bounds_hi = [
        hi[0] + max(0.0, xy, xz, xy + xz),
        hi[1] + max(0.0, yz),
        hi[2],
    ]
    lines = [f"ITEM: BOX BOUNDS xy xz yz {flags}"] + [
        f"{float(bounds_lo[i])!r} {float(bounds_hi[i])!r} {t!r}"
        for i, t in enumerate((xy, xz, yz))
    ]
    return lines, vectors, image


def _other_columns(table: pl.DataFrame, names: Sequence[str]) -> list[tuple]:
    """(dump names, kind, values) of the columns without a dump keyword."""
    result = []
    for name in names:
        series = table[name]
        if any(c.isspace() for c in name) or not name:
            raise InvalidParameter(f"column name {name!r} cannot be written to a dump")
        if series.null_count():
            raise InvalidParameter(f"column {name!r} has missing values")
        inner, shape = series.dtype, ()
        if isinstance(inner, pl.Array):
            shape = inner.shape
            while isinstance(inner, pl.Array):
                inner = inner.inner
        width = math.prod(shape) if shape else 1
        if inner.is_float():
            kind = "f"
        elif inner.is_integer() or inner == pl.Boolean:
            kind = "i"
        else:
            raise UnsupportedFormat(
                f"column {name!r} of type {series.dtype} cannot be written to a dump"
            )
        values = series.to_numpy().reshape(len(series), width)
        if kind == "i":
            values = values.astype(np.int64)
        dump_names = (
            [name] if not shape else [f"{name}[{i}]" for i in range(1, width + 1)]
        )
        result.append((dump_names, kind, values))
    return result


_KEYWORDS = {
    "molecule": ("mol",),
    "charge": ("q",),
    "mass": ("mass",),
    "image": ("ix", "iy", "iz"),
    "velocities": ("vx", "vy", "vz"),
    "forces": ("fx", "fy", "fz"),
}
_FIXED = ("id", "type", "element", "type_label", "positions")


def _write(
    file: BinaryIO,
    systems: Iterable[System],
    *,
    units: str = "metal",
    elements: Sequence[str] | None = None,
    triclinic: str = "auto",
    columns: Sequence[str] | None = None,
) -> None:
    if units not in _UNITS:
        raise InvalidParameter(f"units must be 'metal' or 'real', not {units!r}")
    if triclinic not in ("auto", "general", "restricted"):
        raise InvalidParameter(
            f"triclinic must be 'auto', 'general' or 'restricted', not {triclinic!r}"
        )
    real = units == "real"
    first = True
    for system in systems:
        table = system.table
        names = list(table.columns if columns is None else columns)
        for name in names:
            if name not in table.columns:
                raise InvalidParameter(f"the system has no column named {name!r}")
        n = len(table)
        ids = (
            table["id"].to_numpy().astype(np.int64)
            if "id" in table.columns
            else np.arange(1, n + 1, dtype=np.int64)
        )
        types, _ = _atom_types(table, elements)
        vectors = {
            name: table[name].to_numpy() if name in table.columns else None
            for name in ("velocities", "forces")
        }
        image = table["image"].to_numpy() if "image" in table.columns else None
        box_lines, moved, image = _bounds_lines(
            system.box,
            triclinic,
            [system["positions"], vectors["velocities"], vectors["forces"]],
            image,
        )
        positions, vectors["velocities"], vectors["forces"] = moved

        header_names = ["id", "type"]
        data: list[tuple] = [("i", ids), ("i", types)]
        if "element" in table.columns:
            header_names.append("element")
            data.append(("s", categorize(table["element"])))
        if "type_label" in table.columns:
            labels = categorize(table["type_label"].cast(pl.String))
            if any(not c or any(ch.isspace() for ch in c) for c in labels[1]):
                raise InvalidParameter("type labels must be words without blanks")
            header_names.append("typelabel")
            data.append(("s", labels))
        header_names += ["x", "y", "z"]
        data.append(("f", positions))
        for name, keywords in _KEYWORDS.items():
            if name not in names or name not in table.columns:
                continue
            header_names += list(keywords)
            if name == "velocities":
                # Å/ps to Å/fs in real units, exactly.
                data.append(("f", vectors["velocities"], -3 if real else 0))
            elif name == "forces":
                forces = vectors["forces"]
                data.append(
                    ("f", forces / _units.KCAL_PER_MOL_TO_EV if real else forces)
                )
            elif name == "image":
                data.append(("i", image))
            elif name == "molecule":
                data.append(("i", table[name].to_numpy()))
            else:
                data.append(("f", table[name].to_numpy()))
        others = [n for n in names if n not in _FIXED and n not in _KEYWORDS]
        for dump_names, kind, values in _other_columns(table, others):
            clash = set(dump_names) & set(header_names)
            if clash:
                raise InvalidParameter(f"columns are both written as {sorted(clash)}")
            header_names += dump_names
            data.append((kind, values))

        lines = []
        if first:
            lines += ["ITEM: UNITS", units]
            first = False
        if "time" in system.info:
            time = np.array([float(system.info["time"])])
            text = _native.format_table([("f", time, [], 3 if real else 0)])
            lines += ["ITEM: TIME", text.decode().strip()]
        lines += ["ITEM: TIMESTEP", str(int(system.info.get("timestep", 0)))]
        lines += ["ITEM: NUMBER OF ATOMS", str(n)]
        lines += box_lines
        lines.append("ITEM: ATOMS " + " ".join(header_names))
        file.write(("\n".join(lines) + "\n").encode())
        write_rows(file, data)


register(
    Format(
        name="lammps-dump",
        description="LAMMPS text dump (atom and custom styles)",
        extensions=(".dump", ".lammpstrj"),
        patterns=("dump.*",),
        read=_read,
        scan=_scan,
        write=_write,
    )
)
