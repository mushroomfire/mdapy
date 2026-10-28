# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Crystallographic Information Files: the crystal of each data block."""

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
# a number with an optional standard uncertainty, such as 5.4307(2)
_NUMBER = re.compile(r"([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)(?:\(\d+\))?")
# a term of a symmetry operation, such as -x, +1/2, 2*y or 0.5
_TERM = re.compile(r"([+-]?)(\d+(?:\.\d*)?(?:/\d+(?:\.\d*)?)?|\.\d+)?\*?([xyz]?)")
_OPERATIONS = ("_space_group_symop_operation_xyz", "_symmetry_equiv_pos_as_xyz")
_GROUPS = (
    "_space_group_it_number",
    "_symmetry_int_tables_number",
    "_space_group_name_h-m_alt",
    "_symmetry_space_group_name_h-m",
)
#: Distance in Å below which images of a site made by symmetry are one atom.
TOLERANCE = 0.01


def _lines(source: Source, position: FramePosition) -> Iterator[tuple[int, str]]:
    """The numbered lines of the data block at a position."""
    buffer = source.buffer
    offset, line = position
    while offset < len(buffer):
        end = buffer.find(b"\n", offset)
        stop = len(buffer) if end < 0 else end
        text = bytes(buffer[offset:stop]).decode("utf-8", "replace").rstrip("\r")
        line += 1
        if line > position[1] + 1 and text[:5].lower() == "data_":
            return
        yield line, text
        offset = stop + 1


def _tokens(lines, name: str) -> Iterator[tuple[str, bool, int]]:
    """The words of the lines, whether each was quoted, and its line."""
    field: list[str] | None = None
    start = 0
    for line, text in lines:
        if field is not None:  # inside a text field between lines of ;
            if text.startswith(";"):
                yield "\n".join(field), True, start
                field, text = None, text[1:]
            else:
                field.append(text)
                continue
        elif text.startswith(";"):
            field, start = [text[1:]], line
            continue
        i = 0
        while i < len(text):
            if text[i].isspace():
                i += 1
            elif text[i] == "#":
                break
            elif text[i] in "'\"":
                # a quote ends at the same quote followed by a space
                end = text.find(text[i], i + 1)
                while end >= 0 and end + 1 < len(text) and not text[end + 1].isspace():
                    end = text.find(text[i], end + 1)
                if end < 0:
                    raise FileFormatError("a quote is not closed", name, line)
                yield text[i + 1 : end], True, line
                i = end + 1
            else:
                end = i
                while end < len(text) and not text[end].isspace():
                    end += 1
                yield text[i:end], False, line
                i = end
    if field is not None:
        raise FileFormatError("a text field is not closed with ;", name, start)


def _reserved(word: str) -> bool:
    low = word.lower()
    return low.startswith(("_", "data_", "loop_", "save_", "global_", "stop_"))


def _tag(word: str) -> str:
    """A tag in lowercase, the dotted names of CIF 2 as those of CIF 1."""
    return word.lower().replace(".", "_", 1)


def _parse(source: Source, position: FramePosition):
    """The name of the data block, the values of each tag and the line of
    its first value."""
    tokens = list(_tokens(_lines(source, position), source.name))
    values: dict[str, list[str]] = {}
    lines: dict[str, int] = {}
    block, k = None, 0
    while k < len(tokens):
        word, quoted, line = tokens[k]
        low = word.lower()
        if not quoted and low.startswith("data_"):
            block = word[5:]
            k += 1
        elif not quoted and low == "loop_":
            tags = []
            k += 1
            while k < len(tokens) and not tokens[k][1] and tokens[k][0].startswith("_"):
                tags.append(_tag(tokens[k][0]))
                k += 1
            first = tokens[k][2] if k < len(tokens) else line
            row = []
            while k < len(tokens) and (tokens[k][1] or not _reserved(tokens[k][0])):
                row.append(tokens[k][0])
                k += 1
            if not tags or len(row) % len(tags):
                raise FileFormatError(
                    f"a loop of {len(tags)} tags has {len(row)} values",
                    source.name,
                    line,
                )
            for i, tag in enumerate(tags):
                values[tag], lines[tag] = row[i :: len(tags)], first
        elif not quoted and word.startswith("_"):
            if k + 1 == len(tokens) or (
                not tokens[k + 1][1] and _reserved(tokens[k + 1][0])
            ):
                raise FileFormatError(f"{word} has no value", source.name, line)
            values[_tag(word)], lines[_tag(word)] = [tokens[k + 1][0]], tokens[k + 1][2]
            k += 2
        elif not quoted and low.startswith(("save_", "global_", "stop_")):
            k += 1  # frames of dictionaries: their tags are read as the others
        else:
            raise FileFormatError(f"unexpected {word!r}", source.name, line)
    if block is None:
        raise FileFormatError("no data block (data_)", source.name, position[1] + 1)
    return block, values, lines


def _numbers(values, lines, tag: str, name: str) -> np.ndarray:
    out = []
    for word in values[tag]:
        match = _NUMBER.fullmatch(word)
        if match is None:
            raise FileFormatError(f"{tag} is not a number: {word!r}", name, lines[tag])
        out.append(float(match[1]))
    return np.array(out)


def _cell(a, b, c, alpha, beta, gamma) -> np.ndarray:
    """Lattice vectors of cell parameters: a along x, b in the xy plane."""
    cosines = []
    for angle in (alpha, beta, gamma):
        # right angles exactly, so that orthogonal cells stay so
        cosines.append(0.0 if abs(angle - 90.0) < 1e-10 else np.cos(np.radians(angle)))
    ca, cb, cg = cosines
    sg = np.sin(np.radians(gamma))
    cy = (ca - cb * cg) / sg
    return np.array(
        [
            [a, 0.0, 0.0],
            [b * cg, b * sg, 0.0],
            [c * cb, c * cy, c * np.sqrt(1.0 - cb * cb - cy * cy)],
        ]
    )


def _operation(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Rotation and translation of a symmetry operation such as -x+1/2,y,z."""
    rows = text.replace(" ", "").lower().split(",")
    if len(rows) != 3:
        raise ValueError(f"{text!r} is not three expressions in x, y and z")
    rotation, translation = np.zeros((3, 3)), np.zeros(3)
    for i, row in enumerate(rows):
        at = 0
        while at < len(row) or at == 0:
            match = _TERM.match(row, at)
            sign, number, axis = match.groups()
            if (not number and not axis) or (at and not sign):
                raise ValueError(f"cannot read {text!r}")
            value = 1.0
            if number:
                top, _, bottom = number.partition("/")
                value = float(top) / (float(bottom) if bottom else 1.0)
            value = -value if sign == "-" else value
            if axis:
                rotation[i, "xyz".index(axis)] += value
            else:
                translation[i] += value
            at = match.end()
    return rotation, translation


def _element(word: str) -> str | None:
    """The element of a type symbol or label, such as Fe2+, OW1 or Ca2."""
    letters = re.match(r"[A-Za-z]{1,2}", word)
    if letters is None:
        return None
    pair = letters[0].capitalize()
    if pair in _SYMBOLS:
        return pair
    return pair[0] if pair[0] in _SYMBOLS else None


def _read(source: Source, position: FramePosition) -> System:
    name = source.name
    block, values, lines = _parse(source, position)
    missing = [
        tag
        for tag in ("_cell_length_a", "_cell_length_b", "_cell_length_c")
        if tag not in values
    ]
    if missing:
        raise FileFormatError(
            f"data_{block} has no {missing[0]}", name, position[1] + 1
        )
    lengths = [_numbers(values, lines, f"_cell_length_{k}", name)[0] for k in "abc"]
    angles = [
        _numbers(values, lines, f"_cell_angle_{k}", name)[0]
        if f"_cell_angle_{k}" in values
        else 90.0
        for k in ("alpha", "beta", "gamma")
    ]
    matrix = _cell(*lengths, *angles)
    if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        raise FileFormatError(
            f"data_{block} has no cell of volume", name, position[1] + 1
        )

    if "_atom_site_fract_x" in values:
        fractions = np.stack(
            [_numbers(values, lines, f"_atom_site_fract_{k}", name) for k in "xyz"], 1
        )
    elif "_atom_site_cartn_x" in values:
        cartesian = np.stack(
            [_numbers(values, lines, f"_atom_site_cartn_{k}", name) for k in "xyz"], 1
        )
        fractions = cartesian @ np.linalg.inv(matrix)
    else:
        raise FileFormatError(
            f"data_{block} has no atom sites (_atom_site_fract_x)",
            name,
            position[1] + 1,
        )
    kind = "_atom_site_type_symbol"
    kind = kind if kind in values else "_atom_site_label"
    if kind not in values:
        raise FileFormatError(f"data_{block} names no elements", name, position[1] + 1)
    elements = []
    for word in values[kind]:
        element = _element(word)
        if element is None:
            raise FileFormatError(f"unknown element {word!r}", name, lines[kind])
        elements.append(element)
    if len(elements) != len(fractions):
        raise FileFormatError(
            f"{kind} and the coordinates of the sites differ in number",
            name,
            lines[kind],
        )

    tag = next((t for t in _OPERATIONS if t in values), None)
    if tag is None:
        group = next((t for t in _GROUPS if t in values), None)
        if group is not None and values[group][0].replace(" ", "") not in ("1", "P1"):
            raise FileFormatError(
                f"data_{block} gives the space group {values[group][0]} but not "
                "its symmetry operations",
                name,
                lines[group],
            )
        operations = [(np.eye(3), np.zeros(3))]
    else:
        try:
            operations = [_operation(text) for text in values[tag]]
        except ValueError as e:
            raise FileFormatError(str(e), name, lines[tag]) from None

    # every image of each site, those closer than TOLERANCE to one before
    # left out
    sites, kinds = [], []
    for fraction, element in zip(fractions, elements):
        images = np.array([r @ fraction + t for r, t in operations])
        images -= np.floor(images)
        kept: list[np.ndarray] = []
        for image in images:
            if kept:
                offset = np.array(kept) - image
                offset -= np.round(offset)
                if (np.linalg.norm(offset @ matrix, axis=1) < TOLERANCE).any():
                    continue
            kept.append(image)
        sites.extend(kept)
        kinds.extend([element] * len(kept))
    sites = np.array(sites).reshape(-1, 3)
    table = pl.DataFrame(
        [
            pl.Series("element", kinds, dtype=ELEMENT_DTYPE),
            pl.Series("positions", sites @ matrix, dtype=pl.Array(pl.Float64, 3)),
        ]
    )
    return System(table=table, box=Box(matrix), info={"name": block})


def _scan(source: Source) -> Iterator[FramePosition]:
    """The start of each data block."""
    buffer, offset, line, found = source.buffer, 0, 0, False
    while offset < len(buffer):
        if bytes(buffer[offset : offset + 5]).lower() == b"data_":
            found = True
            yield offset, line
        end = buffer.find(b"\n", offset)
        if end < 0:
            break
        offset, line = end + 1, line + 1
    if not found:
        yield 0, 0  # reading it tells that it has no block


register(
    Format(
        name="cif",
        description="Crystallographic Information File (read only)",
        extensions=(".cif",),
        read=_read,
        scan=_scan,
    )
)
