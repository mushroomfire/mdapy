# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""VASP OUTCAR files: the atoms, forces, energy and virial of each ionic
step."""

import re
from collections.abc import Iterator

import numpy as np
import polars as pl

from mdapy.core.box import Box
from mdapy.core.system import System
from mdapy.elements import ELEMENT_DTYPE, SYMBOLS
from mdapy.errors import FileFormatError
from mdapy.io._files import Source
from mdapy.io._registry import Format, FramePosition, register

_SYMBOLS = frozenset(SYMBOLS)
# the line that ends each ionic step, before its energies
_END = b"FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)"
_FLOAT = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def _ends(source: Source) -> list[tuple[int, int]]:
    """The offset and line number of the line after each ionic step."""
    if "outcar ends" not in source.cache:
        buffer, ends, at, line = source.buffer, [], 0, 0
        while True:
            k = buffer.find(_END, at)
            if k < 0:
                break
            start = buffer.rfind(b"\n", 0, k) + 1
            # not the lines of the machine-learned force field, "ML FREE ..."
            if not bytes(buffer[start:k]).strip():
                end = start
                for _ in range(5):  # the energies follow in four lines
                    found = buffer.find(b"\n", end)
                    end = len(buffer) if found < 0 else found + 1
                ends.append(end)
            at = k + len(_END)
        numbered, previous, line = [], 0, 0
        for end in ends:
            line += bytes(buffer[previous:end]).count(b"\n")
            numbered.append((end, line))
            previous = end
        source.cache["outcar ends"] = numbered
    return source.cache["outcar ends"]


def _species(source: Source) -> list[str]:
    """The element of each atom, from the POTCAR lines, each written twice,
    and the numbers of ions per type of the header."""
    if "outcar species" in source.cache:
        return source.cache["outcar species"]
    ends = _ends(source)
    stop = ends[0][0] if ends else len(source.buffer)
    symbols, counts = [], None
    for line, text in enumerate(
        bytes(source.buffer[:stop]).decode("utf-8", "replace").splitlines(), 1
    ):
        if "POTCAR:" in text:
            parts = text.split()
            word = (
                parts[parts.index("POTCAR:") + 2]
                if "1/r potential" not in text
                else parts[1]
            )
            symbol = "".join(c for c in word.split("_")[0] if c.isalpha())
            if symbol not in _SYMBOLS:
                raise FileFormatError(
                    f"unknown element in {text.strip()!r}", source.name, line
                )
            symbols.append(symbol)
        elif "ions per type" in text and counts is None:
            try:
                counts = [int(w) for w in text.split("=")[1].split()]
            except (IndexError, ValueError):
                raise FileFormatError(
                    "cannot read the ions per type", source.name, line
                ) from None
    if counts is None:
        raise FileFormatError(
            "no line of ions per type before the first step", source.name, 1
        )
    # each POTCAR is listed twice, all of them, then all again
    half = len(symbols) // 2
    if len(symbols) == 2 * len(counts) and symbols[:half] == symbols[half:]:
        symbols = symbols[:half]
    if len(symbols) != len(counts):
        raise FileFormatError(
            f"{len(symbols)} POTCAR lines for {len(counts)} numbers of ions per type",
            source.name,
            1,
        )
    elements = [s for s, c in zip(symbols, counts) for _ in range(c)]
    source.cache["outcar species"] = elements
    return elements


def _numbers(text: str, count: int, name: str, line: int) -> list[float]:
    """The first numbers of a line, also where VASP writes them without
    spaces between them, as -10293.69148-10013.02274."""
    values = [float(x) for x in _FLOAT.findall(text)][:count]
    if len(values) != count:
        raise FileFormatError(
            f"expected {count} numbers, found {text.strip()!r}", name, line
        )
    return values


def _read(source: Source, position: FramePosition) -> System:
    name = source.name
    elements = _species(source)
    n = len(elements)
    end = next((e for e in _ends(source) if e[0] > position[0]), None)
    if end is None:
        raise FileFormatError("the ionic step is not complete", name, position[1] + 1)
    lines = (
        bytes(source.buffer[position[0] : end[0]])
        .decode("utf-8", "replace")
        .splitlines()
    )
    first = position[1]
    matrix = virial = positions = forces = energy = None
    converged = False
    k = 0
    while k < len(lines):
        text = lines[k]
        line = first + k + 1
        if "direct lattice vectors" in text:
            matrix = [_numbers(lines[k + i], 3, name, line + i) for i in (1, 2, 3)]
            k += 4
            continue
        if "FORCE on cell =-STRESS" in text:
            k += 1
            while k < len(lines) and not lines[k].split()[:1] == ["Total"]:
                k += 1
            if k < len(lines):
                xx, yy, zz, xy, yz, zx = _numbers(lines[k], 6, name, first + k + 1)
                virial = [[xx, xy, zx], [xy, yy, yz], [zx, yz, zz]]
        elif text.split()[:1] == ["POSITION"] and "TOTAL-FORCE" in text:
            rows = [
                _numbers(lines[k + 2 + i], 6, name, line + 2 + i)
                for i in range(n)
                if k + 2 + i < len(lines)
            ]
            if len(rows) != n:
                raise FileFormatError(f"the step has fewer than {n} atoms", name, line)
            rows = np.array(rows)
            positions, forces = rows[:, :3], rows[:, 3:]
            k += 2 + n
            continue
        elif "aborting loop because EDIFF is reached" in text:
            converged = True
        elif "free  energy   TOTEN" in text:
            energy = _numbers(text.split("=", 1)[1], 1, name, line)[0]
        k += 1
    if matrix is None:
        raise FileFormatError(
            "no direct lattice vectors before the step", name, first + 1
        )
    if positions is None:
        raise FileFormatError("the step has no positions and forces", name, first + 1)
    if energy is None:
        raise FileFormatError("the step has no free energy TOTEN", name, end[1])
    table = pl.DataFrame(
        [
            pl.Series("element", elements, dtype=ELEMENT_DTYPE),
            pl.Series("positions", positions, dtype=pl.Array(pl.Float64, 3)),
            pl.Series("forces", forces, dtype=pl.Array(pl.Float64, 3)),
        ]
    )
    info = {"energy": energy, "converged": converged}
    if virial is not None:
        info["virial"] = np.array(virial)
    return System(table=table, box=Box(np.array(matrix)), info=info)


def _scan(source: Source) -> Iterator[FramePosition]:
    """The start of each ionic step: the line after the step before."""
    start = (0, 0)
    for end in _ends(source):
        yield start
        start = end


register(
    Format(
        name="outcar",
        description="VASP OUTCAR (read only)",
        extensions=(".outcar",),
        patterns=("OUTCAR*",),
        read=_read,
        scan=_scan,
    )
)
