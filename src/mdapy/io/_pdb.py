# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Protein Data Bank files: the atoms of each model, their bonds."""

from collections.abc import Iterator

import numpy as np
import polars as pl

from mdapy.core.box import Box
from mdapy.core.system import System
from mdapy.core.topology import Topology
from mdapy.elements import ELEMENT_DTYPE, SYMBOLS
from mdapy.errors import FileFormatError
from mdapy.io._cif import _cell
from mdapy.io._files import Source
from mdapy.io._registry import Format, FramePosition, register

_SYMBOLS = frozenset(SYMBOLS)


def _lines(source: Source, offset: int, line: int) -> Iterator[tuple[int, int, str]]:
    """The offset, number and text of each line from an offset on."""
    buffer = source.buffer
    while offset < len(buffer):
        end = buffer.find(b"\n", offset)
        stop = len(buffer) if end < 0 else end
        line += 1
        yield offset, line, bytes(buffer[offset:stop]).decode("utf-8", "replace")
        offset = stop + 1


def _element(text: str, name: str) -> str | None:
    """The element of the element columns, or else of the atom name, whose
    first two columns hold the symbol right-justified."""
    symbol = text.strip().rstrip("+-0123456789").capitalize()
    if symbol in _SYMBOLS:
        return symbol
    if name[:1] in " 0123456789":
        return name[1:2].upper() if name[1:2].upper() in _SYMBOLS else None
    pair = name[:2].strip().capitalize()
    if pair in _SYMBOLS:
        return pair
    return pair[:1] if pair[:1] in _SYMBOLS else None


def _number(text: str, what: str, name: str, line: int) -> float:
    try:
        return float(text)
    except ValueError:
        raise FileFormatError(f"{what} is not a number: {text!r}", name, line) from None


def _crystal(text: str, name: str, line: int) -> np.ndarray | None:
    """The lattice vectors of a CRYST1 record; none for a cell of 1 Å, as
    some programs write for molecules."""
    parameters = [
        _number(text[a:b], "a cell parameter", name, line)
        for a, b in ((6, 15), (15, 24), (24, 33), (33, 40), (40, 47), (47, 54))
    ]
    return None if parameters[:3] == [1.0, 1.0, 1.0] else _cell(*parameters)


def _header(source: Source) -> np.ndarray | None:
    """The cell of the CRYST1 record before the first atom or model."""
    if "pdb cell" not in source.cache:
        source.cache["pdb cell"] = None
        for _, line, text in _lines(source, 0, 0):
            if text.startswith(("ATOM  ", "HETATM", "MODEL ")):
                break
            if text.startswith("CRYST1"):
                source.cache["pdb cell"] = _crystal(text, source.name, line)
    return source.cache["pdb cell"]


def _bonds(source: Source) -> list[tuple[int, int, int]]:
    """The bonds of the CONECT records of the file, by serial number, each
    once, with the line of its record."""
    if "pdb bonds" not in source.cache:
        found: dict[tuple[int, int], int] = {}
        for _, line, text in _lines(source, 0, 0):
            if not text.startswith("CONECT"):
                continue
            fields = [text[k : k + 5].strip() for k in range(6, 31, 5)]
            try:
                first, *others = [int(f) for f in fields if f]
            except ValueError:
                raise FileFormatError(
                    f"a CONECT record holds a word that is not a serial: {text!r}",
                    source.name,
                    line,
                ) from None
            for other in others:
                found.setdefault((min(first, other), max(first, other)), line)
        source.cache["pdb bonds"] = [(a, b, line) for (a, b), line in found.items()]
    return source.cache["pdb bonds"]


def _read(source: Source, position: FramePosition) -> System:
    name = source.name
    ids, names, residues, numbers, elements, positions = [], [], [], [], [], []
    matrix = _header(source)
    for _, line, text in _lines(source, *position):
        record = text[:6]
        if record.startswith(("ENDMDL", "END")) and record.strip() in ("ENDMDL", "END"):
            break
        if record == "MODEL " and ids:
            break  # the next model of a file without ENDMDL
        if record == "CRYST1":
            matrix = _crystal(text, name, line)
        elif record in ("ATOM  ", "HETATM"):
            if len(text) < 54:
                raise FileFormatError(
                    "an atom record ends before its coordinates", name, line
                )
            atom = text[12:16]
            element = _element(text[76:78], atom)
            if element is None:
                raise FileFormatError(
                    f"no element in columns 77-78 or the atom name {atom!r}", name, line
                )
            serial = text[6:11].strip()
            ids.append(int(serial) if serial.isdigit() else len(ids) + 1)
            names.append(atom.strip())
            residues.append(text[17:20].strip())
            number = text[22:26].strip()
            numbers.append(int(number) if number.lstrip("-").isdigit() else 0)
            elements.append(element)
            positions.append(
                [
                    _number(text[a:b], "a coordinate", name, line)
                    for a, b in ((30, 38), (38, 46), (46, 54))
                ]
            )
    table = pl.DataFrame(
        [
            pl.Series("id", ids, dtype=pl.Int64),
            pl.Series("element", elements, dtype=ELEMENT_DTYPE),
            pl.Series(
                "positions",
                np.array(positions).reshape(-1, 3),
                dtype=pl.Array(pl.Float64, 3),
            ),
            pl.Series("atom_name", names, dtype=pl.String),
            pl.Series("residue", residues, dtype=pl.String),
            pl.Series("residue_number", numbers, dtype=pl.Int64),
        ]
    )
    row = {serial: k for k, serial in enumerate(ids)}
    pairs = []
    for a, b, line in _bonds(source):
        if a not in row or b not in row:
            raise FileFormatError(
                f"CONECT bonds atom {b if a in row else a}, which the model lacks",
                name,
                line,
            )
        pairs.append((row[a], row[b]))
    topology = Topology(bonds=pairs) if pairs else None
    box = None if matrix is None else Box(matrix)
    return System(table=table, box=box, topology=topology)


def _scan(source: Source) -> Iterator[FramePosition]:
    """The start of each model, or of the file when it has none."""
    found = False
    for offset, line, text in _lines(source, 0, 0):
        if text.startswith("MODEL "):
            found = True
            yield offset, line - 1
    if not found:
        yield 0, 0


register(
    Format(
        name="pdb",
        description="Protein Data Bank (read only)",
        extensions=(".pdb", ".ent"),
        read=_read,
        scan=_scan,
    )
)
