# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Extended XYZ files, with the conventions of ASE and GPUMD."""

import json
import math
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any, BinaryIO

import numpy as np
import polars as pl

from mdapy import units
from mdapy.core.box import Box
from mdapy.core.system import System
from mdapy.elements import ATOMIC_MASSES, ELEMENT_DTYPE, SYMBOLS
from mdapy.errors import (
    FileFormatError,
    InvalidParameter,
    MissingProperty,
    UnsupportedFormat,
)
from mdapy.io._files import Source
from mdapy.io._registry import Format, FramePosition, register
from mdapy.io._text import TextReader, categorize, strings, write_rows

# ASE's unit of time, Å·sqrt(amu/eV), in ps; ASE momenta are in amu·Å per
# this unit.
_ASE_TIME_IN_PS = math.sqrt(units.AMU_A2_PER_PS2_TO_EV)

# Comment-line keys that set the box or the columns; matched in any case.
_BOX_KEYS = ("lattice", "properties", "pbc", "origin")
# Comment-line keys with units, also matched in any case.
_VALUE_KEYS = ("energy", "virial", "stress")

_DEFAULT_PROPERTIES = "species:S:1:pos:R:3"
_KINDS = {"R": "f", "I": "i", "L": "b", "S": "s"}

# Property in the file -> (column, type, width). velocities come from vel in
# Å/fs (GPUMD) or from momenta in ASE units.
_READ_NAMES = {
    "species": ("element", "S", 1),
    "Z": ("element", "I", 1),
    "pos": ("positions", "R", 3),
    "vel": ("velocities", "R", 3),
    "momenta": ("velocities", "R", 3),
    "force": ("forces", "R", 3),
    "forces": ("forces", "R", 3),
    "mass": ("mass", "R", 1),
    "masses": ("mass", "R", 1),
    "charge": ("charge", "R", 1),
    "initial_charges": ("charge", "R", 1),
    "energies": ("atomic_energies", "R", 1),
    "energy_atom": ("atomic_energies", "R", 1),
    "atomic_virials": ("atomic_virials", "R", 9),
}
# Column -> property in the file, where the names differ.
_WRITE_NAMES = {
    "element": "species",
    "positions": "pos",
    "velocities": "vel",
    "atomic_energies": "energies",
}

_SYMBOLS = frozenset(SYMBOLS)
_COUNT = re.compile(r"\s*([0-9]+)\s*\Z")
_INT = re.compile(r"[+-]?[0-9]+\Z")
_FLOAT = re.compile(
    r"[+-]?(?:(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|nan|inf|infinity)\Z",
    re.IGNORECASE,
)
_BOOLS = {
    "T": True,
    "F": False,
    "True": True,
    "False": False,
    "true": True,
    "false": False,
    "TRUE": True,
    "FALSE": False,
}
_DELIMITERS = {'"': '"', "'": "'", "{": "}", "[": "]"}
_NEEDS_QUOTES = frozenset("\"'{}[]=\\")
_BAD_PROPERTY_CHARS = frozenset("\"'{}[]=\\:")


# ------------------------------------------------------------------ reading


def _split_items(line: str) -> list[tuple[str, str | None]]:
    """Splits a comment line into (key, value) pairs, as ASE does.

    Values may be quoted with "", '', {} or []; a backslash escapes the next
    character. A key without a value gets None.
    """
    items = []
    item: list[list[str] | None] = [None]  # parts separated by '='
    closing = None
    escaped = False
    for char in line:
        if escaped:
            if item[-1] is None:
                item[-1] = []
            item[-1].append(char)
            escaped = False
        elif closing is not None:
            if char == closing:
                closing = None
            elif char == "\\":
                escaped = True
            else:
                item[-1].append(char)
        elif char == "\\":
            escaped = True
        elif char in _DELIMITERS:
            if item[-1] is None:
                item[-1] = []
            closing = _DELIMITERS[char]
        elif char.isspace():
            if item[-1] is not None:
                items.append(item)
                item = [None]
        elif char == "=":
            if item == [None] and items:  # "key = value"
                item = items.pop()
            item.append(None)
        else:
            if item[-1] is None:
                item[-1] = []
            item[-1].append(char)
    if item != [None]:
        items.append(item)
    pairs = []
    for parts in items:
        if parts[0] is None:
            raise ValueError("'=' without a key")
        key = "".join(parts[0])
        value = (
            None if len(parts) == 1 else "=".join("".join(p or []) for p in parts[1:])
        )
        pairs.append((key, value))
    return pairs


def _tokens(raw: str) -> list[str]:
    return re.findall(r"[^\s,]+", raw)


def _value(raw: str | None) -> Any:
    """Value of a comment-line entry: number, boolean, array, JSON or text."""
    if raw is None:
        return True
    if raw.startswith("_JSON "):
        value = json.loads(raw[len("_JSON ") :])
        if isinstance(value, list):
            array = np.array(value)
            if array.dtype.kind in "biuf":
                return array
        return value
    tokens = _tokens(raw)
    if not tokens:
        return raw
    if all(_INT.match(t) for t in tokens):
        values = np.array([int(t) for t in tokens], dtype=np.int64)
    elif all(_FLOAT.match(t) for t in tokens):
        values = np.array([float(t) for t in tokens])
    elif all(t in _BOOLS for t in tokens):
        values = np.array([_BOOLS[t] for t in tokens])
    else:
        return raw
    return values[0].item() if len(values) == 1 else values


def _numbers(key: str, raw: str | None, sizes: tuple[int, ...]) -> np.ndarray:
    tokens = _tokens(raw or "")
    if len(tokens) not in sizes or not all(_FLOAT.match(t) for t in tokens):
        expected = " or ".join(str(s) for s in sizes)
        raise ValueError(f"{key} must have {expected} numbers")
    return np.array([float(t) for t in tokens])


def _tensor(key: str, raw: str | None) -> np.ndarray:
    """A 3x3 tensor from 9 values in Fortran order or 6 in Voigt order."""
    values = _numbers(key, raw, (9, 6))
    if len(values) == 6:
        xx, yy, zz, yz, xz, xy = values
        return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    return values.reshape(3, 3, order="F")


def _special(key: str, raw: str | None) -> Any:
    if key == "properties":
        if raw is None:
            raise ValueError("Properties needs a value")
        return raw
    if key == "lattice":
        return _numbers("Lattice", raw, (9,)).reshape(3, 3)
    if key == "origin":
        return _numbers("Origin", raw, (3,))
    if key == "pbc":
        tokens = _tokens(raw) if raw is not None else ["T"]
        if len(tokens) not in (1, 3) or not all(t in _BOOLS for t in tokens):
            raise ValueError("pbc must have 1 or 3 booleans (T or F)")
        return np.array([_BOOLS[t] for t in tokens] * (3 // len(tokens)))
    if key == "energy":
        return float(_numbers("energy", raw, (1,))[0])
    if key == "virial":
        return _tensor("virial", raw)
    return _tensor("stress", raw) * units.EV_PER_A3_TO_GPA


def _parse_comment(reader: TextReader, line: str) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    try:
        for key, raw in _split_items(line):
            name = key.lower() if key.lower() in _BOX_KEYS + _VALUE_KEYS else key
            if name in fields:
                raise ValueError(f"the key {key!r} appears twice")
            if name in _BOX_KEYS + _VALUE_KEYS:
                fields[name] = _special(name, raw)
            else:
                fields[name] = _value(raw)
    except ValueError as e:
        raise reader.error(str(e)) from None
    return fields


def _parse_properties(reader: TextReader, text: str) -> list[tuple[str, str, int]]:
    fields = text.split(":")
    if len(fields) % 3 != 0:
        raise reader.error(f"Properties must be name:type:width triples, not {text!r}")
    properties = []
    for name, ptype, width in zip(fields[::3], fields[1::3], fields[2::3]):
        if ptype not in _KINDS or not width.isdigit() or int(width) < 1 or not name:
            raise reader.error(f"invalid property {name}:{ptype}:{width}")
        properties.append((name, ptype, int(width)))
    return properties


def _atom_count(reader: TextReader, text: str) -> int:
    match = _COUNT.match(text)
    if match is None:
        raise reader.error(f"expected the number of atoms, found {text.strip()[:40]!r}")
    return int(match.group(1))


def _elements(
    reader: TextReader, codes: np.ndarray, categories: list[str], first_line: int
) -> pl.Series:
    symbols = [c.capitalize() for c in categories]
    for code, symbol in enumerate(symbols):
        if symbol not in _SYMBOLS:
            row = int(np.flatnonzero(codes == code)[0])
            raise reader.error(
                f"unknown element {categories[code]!r}", first_line + row
            )
    return strings(codes, symbols, "element", ELEMENT_DTYPE)


def _column(
    reader: TextReader,
    target: str,
    ptype: str,
    width: int,
    values: Any,
    first_line: int,
) -> pl.Series:
    """The column `target` from the values of one property."""
    if target == "element" and ptype == "S":
        return _elements(reader, *values, first_line)
    if target == "element":  # atomic numbers
        bad = np.flatnonzero((values < 0) | (values >= len(SYMBOLS)))
        if bad.size:
            raise reader.error(
                f"unknown atomic number {values[bad[0]]}", first_line + int(bad[0])
            )
        return strings(
            values.astype(np.uint32), list(SYMBOLS), "element", ELEMENT_DTYPE
        )
    if ptype == "S":
        codes, categories = values
        series = strings(codes.ravel(), categories, target)
        return series.reshape((-1, width)) if width > 1 else series
    if target == "atomic_virials":
        values = values.reshape(-1, 3, 3)
    return pl.Series(target, values)


def _read(source: Source, position: FramePosition) -> System:
    reader = source.reader(*position)
    count = _atom_count(reader, reader.readline())
    comment_line = reader.line + 1
    fields = _parse_comment(reader, reader.readline())
    properties = _parse_properties(
        reader, fields.pop("properties", _DEFAULT_PROPERTIES)
    )
    # The column of each property: the first property that gives a known
    # column takes it; later ones keep their own names.
    targets: list[str] = []
    for name, ptype, width in properties:
        target = name
        known = _READ_NAMES.get(name)
        if known is not None and known[0] not in targets:
            if (ptype, width) != known[1:]:
                raise reader.error(
                    f"property {name!r} must be {name}:{known[1]}:{known[2]}",
                    comment_line,
                )
            target = known[0]
        if target in targets:
            raise reader.error(f"property {target!r} appears twice", comment_line)
        targets.append(target)
    if "positions" not in targets:
        raise reader.error("Properties has no pos property", comment_line)
    arrays = reader.table(
        count,
        [
            # vel is in Å/fs; the shift converts it to Å/ps exactly.
            (_KINDS[t], w, 3 if (name, target) == ("vel", "velocities") else 0)
            for (name, t, w), target in zip(properties, targets)
        ],
    )

    columns: dict[str, pl.Series | None] = {}
    momenta = None
    for (name, ptype, width), target, values in zip(properties, targets, arrays):
        if name == "momenta" and target == "velocities":
            columns[target] = None  # filled in below, once masses are known
            momenta = values
        else:
            columns[target] = _column(
                reader, target, ptype, width, values, comment_line + 1
            )
    if momenta is not None:
        if "mass" in columns:
            masses = columns["mass"].to_numpy()
        elif "element" in columns:
            masses = ATOMIC_MASSES[columns["element"].to_physical().to_numpy()]
        else:
            raise reader.error(
                "momenta need masses, but there are neither masses nor elements",
                comment_line,
            )
        if np.any(masses <= 0):
            raise reader.error(
                "momenta need masses, but an atom has none", comment_line
            )
        columns["velocities"] = pl.Series(
            "velocities", momenta / masses[:, None] / _ASE_TIME_IN_PS
        )

    lattice = fields.pop("lattice", None)
    pbc = fields.pop("pbc", None)
    origin = fields.pop("origin", None)
    if lattice is not None and not lattice.any() and (pbc is None or not pbc.any()):
        lattice = None  # some programs write a zero Lattice for molecules
        pbc = None
    try:
        if lattice is None:
            if (pbc is not None and pbc.any()) or origin is not None:
                raise InvalidParameter("pbc and Origin need a Lattice")
            box = None
        else:
            box = Box(
                lattice,
                origin=(0.0, 0.0, 0.0) if origin is None else origin,
                pbc=True if pbc is None else pbc,
            )
        return System(table=pl.DataFrame(list(columns.values())), box=box, info=fields)
    except InvalidParameter as e:
        raise reader.error(str(e), comment_line) from None


def _scan(source: Source) -> Iterator[FramePosition]:
    reader = source.reader()
    while not reader.at_end():
        position = (reader.offset, reader.line)
        text = reader.readline()
        if not text.strip():
            # Blank lines may end the file, but frames cannot follow them.
            while not reader.at_end():
                if reader.readline().strip():
                    raise reader.error("a frame follows a blank line")
            return
        count = _atom_count(reader, text)
        try:
            reader.skip(count + 1)
        except FileFormatError as e:
            atoms = max(e.line - reader.line - 2, 0)
            raise reader.error(
                f"the frame has {count} atoms, but the file ends after {atoms}",
                e.line,
            ) from None
        yield position


# ------------------------------------------------------------------ writing


def _quote(text: str) -> str:
    if text and not any(c.isspace() or c in _NEEDS_QUOTES for c in text):
        return text
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _number(value: float) -> str:
    return repr(float(value))


def _plain(value: Any) -> Any:
    """value with NumPy and read-only containers replaced for JSON."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _text(key: str, value: Any) -> str:
    """Text of an info value; _value reads it back."""
    if key in ("virial", "stress"):
        tensor = np.asarray(value, dtype=float)
        if tensor.shape != (3, 3):
            raise InvalidParameter(
                f"info[{key!r}] must be a 3x3 array to be written to extxyz"
            )
        if key == "stress":
            tensor = tensor * units.GPA_TO_EV_PER_A3
        return " ".join(_number(v) for v in tensor.ravel(order="F"))
    if key == "energy":
        if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(
            value, (bool, np.bool_)
        ):
            raise InvalidParameter("info['energy'] must be a number")
        return _number(value)
    if isinstance(value, (bool, np.bool_)):
        return "T" if value else "F"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return _number(value)
    if isinstance(value, str):
        try:
            parsed = _value(value)
        except ValueError:  # text that starts with "_JSON " but is not JSON
            parsed = None
        if isinstance(parsed, str) and parsed == value:
            return value
    elif isinstance(value, (np.ndarray, list, tuple)):
        array = np.asarray(value)
        if array.ndim == 1 and array.size > 1 and array.dtype.kind in "biuf":
            return " ".join(_text("", v) for v in array.tolist())
    try:
        return "_JSON " + json.dumps(_plain(value))
    except TypeError:
        raise InvalidParameter(
            f"info[{key!r}] of type {type(value).__name__} cannot be written to extxyz"
        ) from None


def _comment(system: System, properties: str) -> str:
    box = system.box
    parts = [
        "Lattice=" + _quote(" ".join(_number(v) for v in box.matrix.ravel())),
        f"Properties={properties}",
        "pbc=" + _quote(" ".join("T" if p else "F" for p in box.pbc)),
    ]
    if np.any(box.origin != 0):
        parts.append("Origin=" + _quote(" ".join(_number(v) for v in box.origin)))
    for key, value in system.info.items():
        if key.lower() in _BOX_KEYS:
            raise InvalidParameter(f"the info key {key!r} is reserved in extxyz")
        if key.lower() in _VALUE_KEYS and key != key.lower():
            raise InvalidParameter(
                f"the info key {key!r} would be read back as {key.lower()!r}"
            )
        if not key:
            raise InvalidParameter("info keys cannot be empty in extxyz")
        parts.append(f"{_quote(key)}={_quote(_text(key, value))}")
    return " ".join(parts)


def _property(name: str, series: pl.Series) -> tuple[str, int, str, Any]:
    """(type, width, kernel kind, values) of a column."""
    if series.null_count():
        raise InvalidParameter(f"column {name!r} has missing values")
    inner, shape = series.dtype, ()
    if isinstance(inner, pl.Array):
        shape = inner.shape
        while isinstance(inner, pl.Array):
            inner = inner.inner
    width = math.prod(shape) if shape else 1
    rows = len(series)
    if inner.is_float() or inner.is_integer() or inner == pl.Boolean:
        values = series.to_numpy().reshape(rows, width)
        if inner.is_float():
            return "R", width, "f", values
        if inner.is_integer():
            return "I", width, "i", values
        return "L", width, "b", values
    if inner in (pl.String, pl.Categorical) or isinstance(inner, pl.Enum):
        flat = series.reshape((-1,)) if shape else series
        codes, categories = categorize(flat)
        if any(not c or any(ch.isspace() for ch in c) for c in categories):
            raise InvalidParameter(
                f"column {name!r} has empty values or values with blanks"
            )
        return "S", width, "s", (codes.reshape(rows, width), categories)
    raise UnsupportedFormat(
        f"column {name!r} of type {series.dtype} cannot be written to extxyz"
    )


def _write(
    file: BinaryIO, systems: Iterable[System], *, columns: Sequence[str] | None = None
) -> None:
    for system in systems:
        table = system.table
        if "element" not in table.columns:
            raise MissingProperty(
                "extxyz needs element symbols, but the system has no element column"
            )
        names = list(table.columns if columns is None else columns)
        for name in names:
            if name not in table.columns:
                raise InvalidParameter(f"the system has no column named {name!r}")
        names = ["element", "positions"] + [
            n for n in names if n not in ("element", "positions")
        ]
        properties, data, seen = [], [], set()
        for name in names:
            ptype, width, kind, values = _property(name, table[name])
            # vel is in Å/fs; the shift converts from Å/ps exactly.
            shift = -3 if name == "velocities" and kind == "f" else 0
            written = _WRITE_NAMES.get(name, name)
            if any(c.isspace() or c in _BAD_PROPERTY_CHARS for c in written):
                raise InvalidParameter(
                    f"column name {name!r} cannot be an extxyz property name"
                )
            if written in seen:
                raise InvalidParameter(
                    f"columns {name!r} and another one are both written as {written!r}"
                )
            seen.add(written)
            properties.append(f"{written}:{ptype}:{width}")
            data.append((kind, values, shift))
        header = _comment(system, ":".join(properties))
        file.write(f"{len(system)}\n{header}\n".encode())
        write_rows(file, data)


register(
    Format(
        name="extxyz",
        description="Extended XYZ, as written by ASE and GPUMD",
        extensions=(".xyz", ".extxyz"),
        read=_read,
        scan=_scan,
        write=_write,
    )
)
