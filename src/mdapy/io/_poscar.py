# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""VASP POSCAR and CONTCAR files."""

from collections.abc import Iterable, Sequence
from typing import BinaryIO

import numpy as np
import polars as pl

from mdapy.core.box import Box
from mdapy.core.system import System
from mdapy.elements import ELEMENT_DTYPE, SYMBOLS
from mdapy.errors import FileFormatError, InvalidParameter, MissingProperty
from mdapy.io._files import Source
from mdapy.io._registry import Format, FramePosition, register
from mdapy.io._text import TextReader, write_rows

_SYMBOLS = frozenset(SYMBOLS)


def _floats(reader: TextReader, words: list[str], what: str) -> list[float]:
    try:
        return [float(w) for w in words]
    except ValueError:
        raise reader.error(f"expected {what}, found {' '.join(words)!r}") from None


def _species(reader: TextReader, words: list[str]) -> list[str]:
    """Element symbols of the species line; POTCAR labels such as Na_pv or
    Na_pv/6a2f546d name the element before '_'."""
    symbols = [w.split("/")[0].split("_")[0] for w in words]
    for word, symbol in zip(words, symbols):
        if symbol not in _SYMBOLS:
            raise reader.error(f"unknown element {word!r}")
    return symbols


def _coordinates(reader: TextReader, count: int, flags: bool) -> list:
    """Coordinates and selective dynamics flags; words after them, such as
    site labels or comments, are ignored."""
    offset, line = reader.offset, reader.line
    words = len(reader.readline().split("!")[0].split()) if count else 0
    reader.offset, reader.line = offset, line
    needed = 6 if flags else 3
    if count and words < needed:
        raise reader.error(
            f"expected {needed} values per coordinate line, found {words}", line + 1
        )
    columns = [("f", 3)] + ([("b", 3)] if flags else [])
    columns += [("x", 1)] * (words - needed)
    try:
        return reader.table(count, columns, "!")
    except FileFormatError as e:
        if "values, found" not in str(e):
            raise
    # Lines with different numbers of extra words: read them one by one.
    positions = np.empty((count, 3))
    movable = np.empty((count, 3), dtype=bool)
    for i in range(count):
        tokens = reader.readline().split("!")[0].split()
        if len(tokens) < needed:
            raise reader.error(f"expected {needed} values, found {len(tokens)}")
        positions[i] = _floats(reader, tokens[:3], "coordinates")
        if flags:
            if any(t not in ("T", "F") for t in tokens[3:6]):
                raise reader.error("selective dynamics flags must be T or F")
            movable[i] = [t == "T" for t in tokens[3:6]]
    return [positions, movable] if flags else [positions]


def _read(
    source: Source, position: FramePosition, *, elements: Sequence[str] | None = None
) -> System:
    reader = source.reader(*position)
    comment = reader.readline().strip()
    scale = _floats(reader, reader.readline().split(), "the scaling factor")
    if (
        len(scale) not in (1, 3)
        or (len(scale) == 3 and min(scale) <= 0)
        or scale[0] == 0
    ):
        raise reader.error(
            "the scaling factor must be one number or three positive numbers"
        )
    rows = []
    for _ in range(3):
        words = reader.readline().split()[:3]
        if len(words) != 3:
            raise reader.error("a lattice vector needs three numbers")
        rows.append(_floats(reader, words, "a lattice vector"))
    matrix = np.array(rows)
    if scale[0] < 0:  # the volume of the cell
        determinant = abs(np.linalg.det(matrix))
        if determinant == 0:
            raise reader.error("the lattice vectors are linearly dependent")
        factors = np.full(3, np.cbrt(-scale[0] / determinant))
    else:
        factors = np.array(scale * (3 // len(scale)))
    matrix = matrix * factors

    words = reader.readline().split()
    if words and not words[0].isdigit():
        symbols = _species(reader, words)
        words = reader.readline().split()
    elif elements is not None:
        symbols = list(elements)
        for symbol in symbols:
            if symbol not in _SYMBOLS:
                raise InvalidParameter(f"elements= has an unknown symbol {symbol!r}")
    else:
        raise reader.error(
            "the file has no line of element symbols (VASP 4); pass elements="
        )
    words = words[
        : next((i for i, w in enumerate(words) if w.startswith("!")), len(words))
    ]
    if not words or not all(w.isdigit() for w in words) or len(words) != len(symbols):
        raise reader.error(
            f"expected {len(symbols)} atom counts, one per element, found {' '.join(words)!r}"
        )
    counts = [int(w) for w in words]

    mode = reader.readline().strip()
    flags = mode[:1] in ("S", "s")
    if flags:
        mode = reader.readline().strip()
    if not mode:
        raise reader.error("expected Direct or Cartesian")
    cartesian = mode[0] in "CcKk"
    arrays = _coordinates(reader, sum(counts), flags)
    positions = arrays[0] * factors if cartesian else arrays[0] @ matrix
    columns = {
        "element": pl.Series(np.repeat(symbols, counts), dtype=pl.String).cast(
            ELEMENT_DTYPE
        ),
        "positions": positions,
    }
    if flags:
        columns["movable"] = arrays[1]
    velocities = _velocities(reader, sum(counts), matrix)
    if velocities is not None:
        columns["velocities"] = velocities
    table = pl.DataFrame([pl.Series(k, v) for k, v in columns.items()])
    try:
        return System(table=table, box=Box(matrix), info={"comment": comment})
    except InvalidParameter as e:
        raise reader.error(str(e), position[1] + 3) from None


def _velocities(
    reader: TextReader, count: int, matrix: np.ndarray
) -> np.ndarray | None:
    """The velocity block, if present, in Å/ps.

    The line after the coordinates gives the coordinates of the velocities:
    Direct if it starts with D, Cartesian otherwise (VASP leaves it blank,
    ASE writes Cartesian). A line starting with L begins the lattice
    velocities and vectors, which are skipped.
    """
    if reader.at_end() or count == 0:
        return None
    mode = reader.readline().strip()
    if mode[:1] in ("L", "l"):
        reader.skip(7)
        if reader.at_end():
            return None
        mode = reader.readline().strip()
    if reader.at_end():
        return None
    offset, line = reader.offset, reader.line
    if len(reader.readline().split()) < 3:
        return None  # no velocities, or the predictor-corrector block
    reader.offset, reader.line = offset, line
    # Velocities are in Å/fs; the shift gives Å/ps exactly.
    (values,) = reader.table(count, [("f", 3, 3)])
    return values @ matrix if mode[:1] in ("D", "d") else values


def _write(file: BinaryIO, systems: Iterable[System], *, direct: bool = True) -> None:
    for system in systems:
        table = system.table
        if "element" not in table.columns:
            raise MissingProperty(
                "POSCAR needs element symbols, but the system has no element column"
            )
        symbols = table["element"].cast(pl.String)
        species = symbols.unique(maintain_order=True).to_list()
        # VASP needs the atoms of each element together: group them in the
        # order of first appearance, keeping their order within an element.
        rank = {s: i for i, s in enumerate(species)}
        order = np.argsort(symbols.replace_strict(rank).to_numpy(), kind="stable")
        counts = np.bincount(
            symbols.replace_strict(rank).to_numpy(), minlength=len(species)
        )
        box = system.box
        # POSCAR has no box origin: coordinates are relative to it.
        positions = system["positions"][order]
        coordinates = box.to_fractional(positions) if direct else positions - box.origin
        comment = system.info.get("comment", "")
        if not isinstance(comment, str) or "\n" in comment:
            raise InvalidParameter("info['comment'] must be one line of text")
        lines = [comment or "written by mdapy", "1.0"]
        lines += [" ".join(repr(float(x)) for x in row) for row in box.matrix]
        lines += [" ".join(species), " ".join(str(int(c)) for c in counts)]
        columns: list[tuple] = [("f", coordinates)]
        if "movable" in table.columns:
            movable = table["movable"].to_numpy()
            if movable.dtype != np.bool_ or movable.shape != (len(table), 3):
                raise InvalidParameter(
                    "the movable column must hold 3 booleans per atom"
                )
            lines.append("Selective dynamics")
            columns.append(("b", movable[order]))
        lines.append("Direct" if direct else "Cartesian")
        file.write(("\n".join(lines) + "\n").encode())
        write_rows(file, columns)
        if "velocities" in table.columns:
            file.write(b"\n")
            # Å/ps to Å/fs, exactly
            write_rows(file, [("f", table["velocities"].to_numpy()[order], -3)])


register(
    Format(
        name="poscar",
        description="VASP POSCAR and CONTCAR",
        extensions=(".vasp", ".poscar"),
        patterns=("POSCAR*", "CONTCAR*"),
        read=_read,
        write=_write,
    )
)
