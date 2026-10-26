# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""LAMMPS data files, as read_data and write_data of LAMMPS stable_30Sep2026
read and write them, for the atom styles of molecular dynamics."""

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, BinaryIO

import numpy as np
import polars as pl

from mdapy.core.box import Box
from mdapy.core.system import System
from mdapy.core.topology import Topology
from mdapy.elements import ATOMIC_MASSES, ELEMENT_DTYPE, SYMBOLS, element_from_mass
from mdapy.errors import (
    FileFormatError,
    InvalidParameter,
    MissingProperty,
    UnsupportedFormat,
)
from mdapy.io._files import Source
from mdapy.io._registry import Format, FramePosition, register
from mdapy.io._text import TextReader, write_rows

# Columns of the Atoms lines of each atom style, after the atom ID.
_STYLES = {
    "atomic": ("type", "positions"),
    "charge": ("type", "charge", "positions"),
    "bond": ("molecule", "type", "positions"),
    "angle": ("molecule", "type", "positions"),
    "molecular": ("molecule", "type", "positions"),
    "full": ("molecule", "type", "charge", "positions"),
}
_ALL_TERMS = ("bonds", "angles", "dihedrals", "impropers")
# Kinds of topology terms each atom style can hold.
_ALLOWED_TERMS = {
    "atomic": (),
    "charge": (),
    "bond": ("bonds",),
    "angle": ("bonds", "angles"),
    "molecular": _ALL_TERMS,
    "full": _ALL_TERMS,
}
_UNITS = ("metal", "real")

# Topology: kind -> (singular name used in headers and sections, atoms per term)
_TERMS = {"bonds": ("bond", 2), "angles": ("angle", 3), "dihedrals": ("dihedral", 4)}
_TERMS["impropers"] = ("improper", 4)
_TYPE_NAMES = ("atom", "bond", "angle", "dihedral", "improper")

# Coefficient sections and the type count that gives their number of lines;
# "pair" means atom types * (atom types + 1) / 2 lines.
_COEFFS = {
    "Pair Coeffs": "atom",
    "PairIJ Coeffs": "pair",
    "Bond Coeffs": "bond",
    "Angle Coeffs": "angle",
    "Dihedral Coeffs": "dihedral",
    "Improper Coeffs": "improper",
    "BondBond Coeffs": "angle",
    "BondAngle Coeffs": "angle",
    "UreyBradley Coeffs": "angle",
    "MiddleBondTorsion Coeffs": "dihedral",
    "EndBondTorsion Coeffs": "dihedral",
    "AngleTorsion Coeffs": "dihedral",
    "AngleAngleTorsion Coeffs": "dihedral",
    "BondBond13 Coeffs": "dihedral",
    "AngleAngle Coeffs": "improper",
}
_LABELS = {f"{name.capitalize()} Type Labels": name for name in _TYPE_NAMES}
_TOPOLOGY_SECTIONS = {kind.capitalize(): kind for kind in _ALL_TERMS}
# Sections and header counts of the atom styles of finite-size particles.
_PARTICLE_SECTIONS = ("Ellipsoids", "Lines", "Triangles", "Bodies")
_PARTICLE_COUNTS = ("ellipsoids", "lines", "triangles", "bodies")

_DIGITS = re.compile(r"[0-9]+\Z")
_SYMBOL_SET = frozenset(SYMBOLS)


def _unsupported_style(style: str) -> UnsupportedFormat:
    return UnsupportedFormat(
        f"atom style {style!r} is not supported; mdapy reads and writes the atom "
        f"styles {', '.join(_STYLES)}"
    )


# ------------------------------------------------------------------ reading


class _Header:
    """Values of the header of a data file."""

    def __init__(self, title: str):
        timestep = re.search(r"timestep = ([0-9]+)", title)
        units = re.search(r"units = (\w+)", title)
        self.timestep = None if timestep is None else int(timestep.group(1))
        self.units = None if units is None else units.group(1)
        self.counts = {"atoms": 0, **{kind: 0 for kind in _ALL_TERMS}}
        self.types = {name: 0 for name in _TYPE_NAMES}
        self.extra: dict[str, int] = {}
        self.lo = [-0.5, -0.5, -0.5]  # LAMMPS defaults
        self.hi = [0.5, 0.5, 0.5]
        self.tilt = [0.0, 0.0, 0.0]
        self.general: dict[str, list[float]] = {}
        self.restricted = False

    def parse(self, reader: TextReader, words: list[str]) -> bool:
        """Takes one header line; False if it is not a header line."""
        n = len(words)
        if n == 2 and words[1] in self.counts:
            self.counts[words[1]] = _count(reader, words[0])
        elif n == 2 and words[1] in _PARTICLE_COUNTS:
            raise UnsupportedFormat(
                f"data files with {words[1]} need atom styles of finite-size "
                f"particles; mdapy reads the atom styles {', '.join(_STYLES)}"
            )
        elif n == 3 and words[1] in self.types and words[2] == "types":
            self.types[words[1]] = _count(reader, words[0])
        elif (
            n == 5
            and words[1] == "extra"
            and words[2] in (*_TYPE_NAMES[1:], "special")
            and words[3:] == ["per", "atom"]
        ):
            self.extra[words[2]] = _count(reader, words[0])
        elif n == 4 and words[2:] in (["xlo", "xhi"], ["ylo", "yhi"], ["zlo", "zhi"]):
            axis = "xyz".index(words[2][0])
            self.lo[axis], self.hi[axis] = _floats(reader, words[:2])
            self.restricted = True
        elif n == 6 and words[3:] == ["xy", "xz", "yz"]:
            self.tilt = _floats(reader, words[:3])
            self.restricted = True
        elif n == 4 and words[3] in ("avec", "bvec", "cvec"):
            self.general[words[3]] = _floats(reader, words[:3])
        elif n == 5 and words[3:] == ["abc", "origin"]:
            self.general["origin"] = _floats(reader, words[:3])
        else:
            return False
        return True

    def box(self, reader: TextReader) -> Box:
        try:
            if not self.general:
                return Box.from_lammps(self.lo, self.hi, self.tilt)
            if self.restricted:
                raise InvalidParameter(
                    "the header gives the box both by bounds (xlo xhi ...) and by "
                    "edge vectors (avec ...)"
                )
            missing = [k for k in ("avec", "bvec", "cvec") if k not in self.general]
            if missing:
                raise InvalidParameter(f"the header has no {' or '.join(missing)}")
            matrix = [self.general[k] for k in ("avec", "bvec", "cvec")]
            return Box(matrix, origin=self.general.get("origin", (0.0, 0.0, 0.0)))
        except InvalidParameter as e:
            raise FileFormatError(str(e), reader.name) from None


def _count(reader: TextReader, word: str) -> int:
    if not _DIGITS.match(word):
        raise reader.error(f"expected a count, found {word!r}")
    return int(word)


def _floats(reader: TextReader, words: list[str]) -> list[float]:
    try:
        return [float(w) for w in words]
    except ValueError:
        raise reader.error(f"expected numbers, found {' '.join(words)!r}") from None


def _split(text: str) -> tuple[list[str], str]:
    """Words before a '#', and the text after it."""
    content, _, comment = text.partition("#")
    return content.split(), comment.strip()


def _next_keyword(reader: TextReader) -> tuple[str, str, int] | None:
    """The next section keyword, its style comment and its line, skipping
    blank and comment lines; None at the end of the file."""
    while not reader.at_end():
        text = reader.readline()
        content, _, comment = text.partition("#")
        if content.strip():
            return content.strip(), comment.strip(), reader.line
    return None


def _ids_to_rows(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sorted atom IDs and the row of each, for _rows_of."""
    order = np.argsort(ids, kind="stable")
    return ids[order], order


def _rows_of(index: tuple[np.ndarray, np.ndarray], query: np.ndarray) -> np.ndarray:
    """Rows of the atoms with the given IDs; -1 where no atom has the ID."""
    sorted_ids, order = index
    if len(sorted_ids) == 0:
        return np.full(query.shape, -1, dtype=np.int64)
    position = np.searchsorted(sorted_ids, query)
    clipped = np.minimum(position, len(sorted_ids) - 1)
    found = (position < len(sorted_ids)) & (sorted_ids[clipped] == query)
    return np.where(found, order[clipped], -1)


def _check_unique(reader: TextReader, values: np.ndarray, first_line: int) -> None:
    """Raises at the first line whose atom ID appeared before."""
    _, first = np.unique(values, return_index=True)
    if len(first) != len(values):
        repeated = np.ones(len(values), dtype=bool)
        repeated[first] = False
        row = int(np.flatnonzero(repeated)[0])
        raise reader.error("atom IDs are repeated", first_line + row)


def _type_numbers(
    reader: TextReader,
    values: tuple[np.ndarray, list[str]],
    labels: Sequence[str] | None,
    count: int,
    name: str,
    first_line: int,
) -> np.ndarray:
    """Numeric types from type numbers or type labels."""
    codes, categories = values
    numbers = np.empty(len(categories), dtype=np.int32)
    lookup = {} if labels is None else {label: i + 1 for i, label in enumerate(labels)}
    for code, text in enumerate(categories):
        if _DIGITS.match(text):
            number = int(text)
            problem = (
                None
                if 1 <= number <= count
                else (f"{name} type {number} is not between 1 and {count}")
            )
        else:
            number = lookup.get(text, 0)
            problem = None if number else f"unknown {name} type label {text!r}"
        if problem:
            row = int(np.flatnonzero(codes == code)[0])
            raise reader.error(problem, first_line + row)
        numbers[code] = number
    return numbers[codes]


class _Contents:
    """The sections of a data file, before they are combined."""

    def __init__(self):
        self.atoms: list[Any] | None = None  # parsed columns
        self.atoms_line = 0  # line of the first Atoms line
        self.style = ""
        self.velocities: list[Any] | None = None
        self.velocities_line = 0
        self.masses: list[tuple[str, float, str]] | None = None  # (type, mass, comment)
        self.masses_line = 0
        self.labels: dict[str, tuple[str, ...]] = {}
        self.terms: dict[str, tuple[list[Any], int]] = {}  # kind -> (columns, line)
        self.coeffs: dict[str, dict[str, Any]] = {}


def _read_atoms(
    reader: TextReader,
    header: _Header,
    contents: _Contents,
    hint: str,
    style: str | None,
):
    count = header.counts["atoms"]
    hinted = hint.split()[0] if hint else None
    if style is None and hinted is not None:
        if hinted not in _STYLES:
            raise _unsupported_style(hinted)
        style = hinted
    words = 0
    if count:
        offset, line = reader.offset, reader.line
        words = len(_split(reader.readline())[0])
        reader.offset, reader.line = offset, line
    if style is None:
        style = _infer_style(reader, header, words)
    expected = 1 + len(_STYLES[style]) + 2  # positions take 3 words
    if count and words not in (expected, expected + 3):
        raise reader.error(
            f"atom style {style} has {expected} values per Atoms line, or "
            f"{expected + 3} with image flags; found {words}",
            reader.line + 1,
        )
    columns = [("i", 1)]
    for name in _STYLES[style]:
        columns.append(
            {"molecule": ("i", 1), "type": ("s", 1), "charge": ("f", 1)}.get(
                name, ("f", 3)
            )
        )
    if words == expected + 3:
        columns.append(("i", 3))
    contents.style = style
    contents.atoms_line = reader.line + 1
    contents.atoms = reader.table(count, columns, "#")


def _infer_style(reader: TextReader, header: _Header, words: int) -> str:
    """Atom style of an Atoms section that does not name one."""
    has_terms = any(header.counts[k] or header.types[_TERMS[k][0]] for k in _ALL_TERMS)
    candidates = {5: "atomic", 8: "atomic", 7: "full", 10: "full"}
    if words in candidates:
        return candidates[words]
    if words in (6, 9) and has_terms:
        return "molecular"
    if words in (6, 9):
        raise reader.error(
            "the Atoms section does not name its atom style, and its lines fit "
            "both charge and molecular; pass atom_style='charge' or "
            "atom_style='molecular'",
            reader.line + 1,
        )
    raise reader.error(
        f"the Atoms lines have {words} values, which fits none of the atom styles "
        f"{', '.join(_STYLES)}",
        reader.line + 1,
    )


def _read_section(
    reader: TextReader,
    keyword: str,
    hint: str,
    header: _Header,
    contents: _Contents,
    style: str | None,
    units: str,
    line: int,
) -> None:
    """Reads the lines of one section; line is the line of its keyword."""
    types = header.types
    if keyword == "Atoms":
        _read_atoms(reader, header, contents, hint, style)
    elif keyword == "Velocities":
        contents.velocities_line = reader.line + 1
        # In real units velocities are in Å/fs; the shift gives Å/ps exactly.
        shift = 3 if units == "real" else 0
        contents.velocities = reader.table(
            header.counts["atoms"], [("i", 1), ("f", 3, shift)], "#"
        )
    elif keyword == "Masses":
        contents.masses_line = reader.line + 1
        contents.masses = []
        for _ in range(types["atom"]):
            words, comment = _split(reader.readline())
            if len(words) != 2:
                raise reader.error("expected a type and a mass")
            contents.masses.append((words[0], _floats(reader, words[1:])[0], comment))
    elif keyword in _LABELS:
        name = _LABELS[keyword]
        labels = [""] * types[name]
        for _ in range(types[name]):
            words, _ = _split(reader.readline())
            if len(words) != 2 or not _DIGITS.match(words[0]):
                raise reader.error("expected a type and a label")
            number, label = int(words[0]), words[1]
            if not 1 <= number <= types[name]:
                raise reader.error(
                    f"{name} type {number} is not between 1 and {types[name]}"
                )
            if labels[number - 1] or label in labels or _label_problem(label):
                raise reader.error(
                    f"{name} type {number} or label {label!r} is repeated or invalid"
                )
            labels[number - 1] = label
        contents.labels[name] = tuple(labels)
    elif keyword in _TOPOLOGY_SECTIONS:
        kind = _TOPOLOGY_SECTIONS[keyword]
        columns = [("x", 1), ("s", 1), ("i", _TERMS[kind][1])]
        contents.terms[kind] = (
            reader.table(header.counts[kind], columns, "#"),
            reader.line + 1 - header.counts[kind],
        )
    elif keyword in _COEFFS:
        need = _COEFFS[keyword]
        n = types["atom"] * (types["atom"] + 1) // 2 if need == "pair" else types[need]
        lines = []
        for _ in range(n):
            words, _ = _split(reader.readline())
            if not words:
                raise reader.error("expected coefficients")
            lines.append(" ".join(words))
        contents.coeffs[keyword] = {"style": hint or None, "lines": tuple(lines)}
    elif keyword in _PARTICLE_SECTIONS:
        raise UnsupportedFormat(
            f"the {keyword} section needs an atom style of finite-size particles; "
            f"mdapy reads the atom styles {', '.join(_STYLES)}"
        )
    else:
        raise reader.error(f"unknown section {keyword!r}", line)


def _label_problem(label: str) -> bool:
    """Whether LAMMPS rejects a type label."""
    return (
        not label
        or label[0].isdigit()
        or label[0] in "*#"
        or any(c.isspace() for c in label)
        or not label.isascii()
    )


def _guess_elements(
    used: np.ndarray, masses: list[tuple[str, float, str]] | None, labels: tuple | None
) -> dict[int, str] | None:
    """Element of each used type, if one source names them all."""

    def complete(found: dict[int, str]) -> dict[int, str] | None:
        return found if all(int(t) in found for t in used) else None

    if masses is not None:
        comments = {
            i + 1: words[0]
            for i, (_, _, comment) in enumerate(masses)
            if (words := comment.split()) and words[0] in _SYMBOL_SET
        }
        if found := complete(comments):
            return found
    if labels is not None:
        found = {i + 1: label for i, label in enumerate(labels) if label in _SYMBOL_SET}
        if found := complete(found):
            return found
    if masses is not None:
        found = {}
        for i, (_, mass, _) in enumerate(masses):
            try:
                found[i + 1] = element_from_mass(mass)
            except InvalidParameter:
                pass
        return complete(found)
    return None


def _read(
    source: Source,
    position: FramePosition,
    *,
    atom_style: str | None = None,
    units: str | None = None,
    elements: Sequence[str] | None = None,
) -> System:
    if atom_style is not None and atom_style not in _STYLES:
        raise InvalidParameter(
            f"atom_style must be one of {', '.join(_STYLES)}, not {atom_style!r}"
        )
    if units is not None and units not in _UNITS:
        raise InvalidParameter(f"units must be 'metal' or 'real', not {units!r}")
    reader = source.reader(*position)
    header = _Header(reader.readline())
    keyword = None
    while not reader.at_end():
        text = reader.readline()
        words, comment = _split(text)
        if words and not header.parse(reader, words):
            keyword = (text.partition("#")[0].strip(), comment, reader.line)
            break
    units = units or header.units or "metal"
    if units not in _UNITS:
        raise UnsupportedFormat(
            f"the file is in {units} units; mdapy reads metal and real units "
            "(pass units='metal' to read the numbers unchanged)"
        )
    box = header.box(reader)

    contents = _Contents()
    seen = set()
    while keyword is not None:
        name, hint, line = keyword
        if name in seen:
            raise reader.error(f"the section {name!r} appears twice", line)
        seen.add(name)
        if not reader.at_end():
            reader.readline()  # LAMMPS skips the line after a keyword
        _read_section(reader, name, hint, header, contents, atom_style, units, line)
        keyword = _next_keyword(reader)
    return _combine(reader, header, contents, box, units, elements)


def _combine(
    reader: TextReader,
    header: _Header,
    contents: _Contents,
    box: Box,
    units: str,
    elements: Sequence[str] | None,
) -> System:
    natoms = header.counts["atoms"]
    if contents.atoms is None:
        if natoms:
            raise FileFormatError(
                f"the header has {natoms} atoms but there is no Atoms section",
                reader.name,
            )
        empty_type = (np.empty(0, np.uint32), [])
        contents.atoms = [np.empty(0, np.int64), empty_type, np.empty((0, 3))]
        contents.style = "atomic"
    style = contents.style
    ids = contents.atoms[0]
    sorted_ids = _ids_to_rows(ids)
    _check_unique(reader, ids, contents.atoms_line)

    columns: dict[str, Any] = {"id": ids}
    for name, values in zip(_STYLES[style], contents.atoms[1:]):
        if name == "type":
            values = _type_numbers(
                reader,
                values,
                contents.labels.get("atom"),
                header.types["atom"],
                "atom",
                contents.atoms_line,
            )
        columns[name] = values
    if len(contents.atoms) > len(_STYLES[style]) + 1:
        columns["image"] = contents.atoms[-1]

    if contents.velocities is not None:
        vid, velocities = contents.velocities
        rows = _rows_of(sorted_ids, vid)
        bad = np.flatnonzero(rows < 0)
        if bad.size:
            raise reader.error(
                f"no atom has ID {vid[bad[0]]}", contents.velocities_line + int(bad[0])
            )
        _check_unique(reader, rows, contents.velocities_line)
        ordered = np.empty_like(velocities)
        ordered[rows] = velocities
        columns["velocities"] = ordered

    types = columns["type"]
    lammps: dict[str, Any] = {"atom_style": style, "units": units}
    for name in _TYPE_NAMES:
        if header.types[name]:
            lammps[f"{name}_types"] = header.types[name]
    if header.extra:
        lammps["extra"] = dict(header.extra)
    if contents.labels:
        lammps["type_labels"] = dict(contents.labels)
    if contents.coeffs:
        lammps["coeffs"] = contents.coeffs

    masses = None
    if contents.masses is not None:
        masses = np.full(header.types["atom"], np.nan)
        for i, (word, mass, _) in enumerate(contents.masses):
            number = _type_numbers(
                reader,
                (np.zeros(1, np.uint32), [word]),
                contents.labels.get("atom"),
                header.types["atom"],
                "atom",
                contents.masses_line + i,
            )[0]
            masses[number - 1] = mass
        columns["mass"] = masses[types - 1]
        lammps["masses"] = tuple(masses.tolist())
    if "atom" in contents.labels:
        labels = np.array(contents.labels["atom"], dtype=object)
        columns["type_label"] = pl.Series(labels[types - 1].tolist(), dtype=pl.String)

    if elements is not None:
        elements = list(elements)
        if len(elements) < header.types["atom"]:
            raise InvalidParameter(
                f"elements= names {len(elements)} elements for "
                f"{header.types['atom']} atom types"
            )
        named = dict(enumerate(elements, start=1))
    else:
        named = _guess_elements(
            np.unique(types), contents.masses, contents.labels.get("atom")
        )
    if named:
        symbols = [named.get(t, "X") for t in range(header.types["atom"] + 1)]
        try:
            element_type = pl.Series(symbols, dtype=pl.String).cast(ELEMENT_DTYPE)
        except pl.exceptions.InvalidOperationError:
            raise InvalidParameter(
                f"elements= has an unknown symbol: {elements}"
            ) from None
        columns["element"] = element_type.gather(types)

    topology = _topology(reader, header, contents, style, sorted_ids)
    table = pl.DataFrame(
        [
            pl.Series(name, values)
            if not isinstance(values, pl.Series)
            else values.alias(name)
            for name, values in columns.items()
        ]
    )
    info: dict[str, Any] = {}
    if header.timestep is not None:
        info["timestep"] = header.timestep
    info["lammps"] = lammps
    try:
        return System(table=table, box=box, info=info, topology=topology)
    except InvalidParameter as e:
        raise FileFormatError(str(e), reader.name) from None


def _topology(
    reader: TextReader,
    header: _Header,
    contents: _Contents,
    style: str,
    sorted_ids: tuple[np.ndarray, np.ndarray],
) -> Topology | None:
    tables = {}
    for kind in _ALL_TERMS:
        name, width = _TERMS[kind]
        if (header.counts[kind] or header.types[name]) and kind not in _ALLOWED_TERMS[
            style
        ]:
            raise FileFormatError(f"atom style {style} cannot have {kind}", reader.name)
        if kind not in contents.terms:
            if header.counts[kind]:
                raise FileFormatError(
                    f"the header has {header.counts[kind]} {kind} but there is no "
                    f"{kind.capitalize()} section",
                    reader.name,
                )
            continue
        (_, types, atom_ids), line = contents.terms[kind]
        types = _type_numbers(
            reader, types, contents.labels.get(name), header.types[name], name, line
        )
        rows = _rows_of(sorted_ids, atom_ids)
        bad = np.flatnonzero((rows < 0).any(axis=1))
        if bad.size:
            missing = atom_ids[bad[0]][rows[bad[0]] < 0][0]
            raise reader.error(f"no atom has ID {missing}", line + int(bad[0]))
        table = {"atoms": rows, "type": types}
        if name in contents.labels:
            labels = np.array(contents.labels[name], dtype=object)
            table["type_label"] = pl.Series(labels[types - 1].tolist(), dtype=pl.String)
        tables[kind] = pl.DataFrame(table)
    if not any(len(t) for t in tables.values()):
        return None
    try:
        return Topology(**tables)
    except InvalidParameter as e:
        raise FileFormatError(str(e), reader.name) from None


# ------------------------------------------------------------------ writing


def _number(value: float) -> str:
    return repr(float(value))


def _per_type(values: np.ndarray, types: np.ndarray, name: str) -> dict[int, Any]:
    """The value of each type, checking that all atoms of a type agree."""
    pairs = pl.DataFrame({"type": types, "value": values}).unique()
    repeated = pairs.filter(pl.col("type").is_duplicated())
    if len(repeated):
        t = repeated["type"].min()
        raise InvalidParameter(f"atoms of type {t} have different {name} values")
    return {int(t): value for t, value in pairs.iter_rows()}


def _box_lines(
    box: Box, triclinic: str, vectors: list[np.ndarray], image: np.ndarray | None
) -> tuple[list[str], list[np.ndarray], np.ndarray | None]:
    """Header lines of the box, and positions, other vectors and image flags
    in the frame of the written box."""
    m, origin = box.matrix, box.origin
    restricted_form = m[0, 1] == m[0, 2] == m[1, 2] == 0 and np.all(np.diag(m) > 0)
    if triclinic == "restricted" and not restricted_form:
        lammps = box.to_lammps()
        vectors = [v @ lammps.rotation for v in vectors]
        if np.linalg.det(m) < 0 and image is not None:
            image = image[:, [0, 2, 1]]
        lo, hi = lammps.lo, lammps.hi
        xy, xz, yz = lammps.tilt
        return _restricted_lines(lo, hi, (xy, xz, yz)), vectors, image
    if triclinic != "general" and restricted_form:
        hi = origin + np.diag(m)
        tilt = (m[1, 0], m[2, 0], m[2, 1])
        return _restricted_lines(origin, hi, tilt), vectors, image
    if np.linalg.det(m) < 0:  # LAMMPS needs right-handed edge vectors
        m = m[[0, 2, 1]]
        if image is not None:
            image = image[:, [0, 2, 1]]
    lines = [
        " ".join(_number(x) for x in m[i]) + f" {n}vec" for i, n in enumerate("abc")
    ]
    lines.append(" ".join(_number(x) for x in origin) + " abc origin")
    return lines, vectors, image


def _restricted_lines(lo: Sequence[float], hi: Sequence[float], tilt) -> list[str]:
    lines = [
        f"{_number(lo[i])} {_number(hi[i])} {a}lo {a}hi" for i, a in enumerate("xyz")
    ]
    if any(t != 0 for t in tilt):
        lines.append(" ".join(_number(t) for t in tilt) + " xy xz yz")
    return lines


def _default_style(table: pl.DataFrame, topology: Topology | None) -> str:
    molecular = (
        topology is not None and len(topology) > 0 or "molecule" in table.columns
    )
    if molecular:
        return "full" if "charge" in table.columns else "molecular"
    return "charge" if "charge" in table.columns else "atomic"


def _atom_types(
    table: pl.DataFrame, elements: Sequence[str] | None
) -> tuple[np.ndarray, dict[int, str]]:
    """Type of each atom and the element of each type that has one."""
    if "type" in table.columns:
        if elements is not None:
            raise InvalidParameter(
                "elements= assigns types from elements, but the system has a type column"
            )
        types = table["type"].to_numpy().astype(np.int64)
        named = {}
        if "element" in table.columns:
            # the elements of each type, from the pairs that occur
            pairs = table.select("type", "element").unique()
            found = pairs.group_by("type").agg(pl.col("element").first(), pl.len())
            for t, symbol, count in found.iter_rows():
                if count == 1 and symbol != "X":
                    named[int(t)] = str(symbol)
        return types, named
    if "element" not in table.columns:
        if elements is not None:
            raise InvalidParameter("elements= needs an element column")
        return np.ones(len(table), dtype=np.int64), {}
    symbols = table["element"].cast(pl.String)
    order = (
        list(elements) if elements is not None else symbols.unique(maintain_order=True)
    )
    order = [str(s) for s in order]
    missing = set(symbols.unique().to_list()) - set(order)
    if missing:
        raise InvalidParameter(f"elements= lacks the elements {sorted(missing)}")
    lookup = {symbol: i + 1 for i, symbol in enumerate(order)}
    types = symbols.replace_strict(lookup, return_dtype=pl.Int64).to_numpy()
    return types, {i + 1: s for i, s in enumerate(order) if s != "X"}


def _lammps_info(system: System) -> Mapping[str, Any]:
    value = system.info.get("lammps", {})
    if not isinstance(value, Mapping):
        raise InvalidParameter("info['lammps'] must be a mapping")
    return value


def _labels(
    name: str, count: int, stored: Sequence[str], found: Mapping[int, Any]
) -> list[str] | None:
    """Type labels of every type, from info and from the columns."""
    labels = dict(enumerate(stored, start=1))
    labels.update(found)
    if not labels:
        return None
    missing = [t for t in range(1, count + 1) if t not in labels]
    values = [str(v) for v in labels.values()]
    bad = [v for v in values if _label_problem(v)]
    if missing or bad or len(set(values)) != len(values):
        details = []
        if missing:
            details.append(f"types {missing} have none")
        if bad:
            details.append(f"invalid labels {bad}")
        if len(set(values)) != len(values):
            details.append("labels are repeated")
        raise InvalidParameter(f"{name} type labels: {'; '.join(details)}")
    return [str(labels[t]) for t in range(1, count + 1)]


def _plan(system: System, atom_style, units, elements, triclinic) -> dict[str, Any]:
    """Everything that goes into the file, checked."""
    table, topology = system.table, system.topology
    lammps = _lammps_info(system)
    style = atom_style or lammps.get("atom_style") or _default_style(table, topology)
    if style not in _STYLES:
        raise _unsupported_style(style)
    if "charge" in _STYLES[style] and "charge" not in table.columns:
        raise MissingProperty(f"atom style {style} needs a charge column")

    n = len(table)
    if "id" in table.columns:
        ids = table["id"].to_numpy().astype(np.int64)
        if np.any(ids < 1) or table["id"].n_unique() != n:
            raise InvalidParameter("atom IDs must be positive and unique")
    else:
        ids = np.arange(1, n + 1, dtype=np.int64)
    types, named = _atom_types(table, elements)
    counts = {name: int(lammps.get(f"{name}_types", 0)) for name in _TYPE_NAMES}
    counts["atom"] = max(
        counts["atom"], int(types.max()) if n else 0, max(named, default=0)
    )

    # Masses, by priority: mass column, masses kept in info, standard atomic
    # weights of the elements.
    masses = {t: float(ATOMIC_MASSES[SYMBOLS.index(s)]) for t, s in named.items()}
    for t, mass in enumerate(lammps.get("masses", ()), start=1):
        if t <= counts["atom"] and not np.isnan(mass):
            masses[t] = float(mass)
    if "mass" in table.columns:
        masses.update(_per_type(table["mass"].to_numpy(), types, "mass"))
    if masses and len(masses) != counts["atom"]:
        missing = [t for t in range(1, counts["atom"] + 1) if t not in masses]
        raise InvalidParameter(f"no mass is known for atom types {missing}")

    stored_labels = lammps.get("type_labels", {})
    labels: dict[str, list[str]] = {}
    found = {}
    if "type_label" in table.columns:
        found = _per_type(table["type_label"].to_numpy(), types, "type_label")
    atom_labels = _labels("atom", counts["atom"], stored_labels.get("atom", ()), found)
    if atom_labels is not None:
        labels["atom"] = atom_labels
    terms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for kind in _ALL_TERMS:
        name = _TERMS[kind][0]
        frame = getattr(topology, kind) if topology is not None else None
        found = {}
        if frame is not None and len(frame):
            if kind not in _ALLOWED_TERMS[style]:
                raise InvalidParameter(f"atom style {style} cannot hold {kind}")
            numbers = frame["type"].to_numpy().astype(np.int64)
            counts[name] = max(counts[name], int(numbers.max()))
            terms[kind] = (ids[frame["atoms"].to_numpy()], numbers)
            if "type_label" in frame.columns:
                found = _per_type(frame["type_label"].to_numpy(), numbers, "type_label")
        names = _labels(name, counts[name], stored_labels.get(name, ()), found)
        if names is not None:
            labels[name] = names
    for kind in _ALL_TERMS:
        if counts[_TERMS[kind][0]] and kind not in _ALLOWED_TERMS[style]:
            raise InvalidParameter(f"atom style {style} cannot hold {kind}")

    coeffs = lammps.get("coeffs", {})
    if coeffs and lammps.get("units", units) != units:
        raise InvalidParameter(
            f"the force-field coefficients in info['lammps'] are in "
            f"{lammps.get('units')} units and cannot be written in {units} units"
        )
    for section, value in coeffs.items():
        need = _COEFFS.get(section)
        if need is None:
            raise InvalidParameter(f"unknown coefficient section {section!r}")
        expected = (
            counts["atom"] * (counts["atom"] + 1) // 2
            if need == "pair"
            else counts[need]
        )
        if len(value["lines"]) != expected:
            raise InvalidParameter(
                f"{section} in info['lammps'] has {len(value['lines'])} lines, but "
                f"{expected} are needed"
            )

    velocities = (
        table["velocities"].to_numpy() if "velocities" in table.columns else None
    )
    image = table["image"].to_numpy() if "image" in table.columns else None
    vectors = [system["positions"]] + ([velocities] if velocities is not None else [])
    box_lines, vectors, image = _box_lines(system.box, triclinic, vectors, image)
    return {
        "style": style,
        "ids": ids,
        "types": types,
        "counts": counts,
        "masses": masses,
        "named": named,
        "labels": labels,
        "terms": terms,
        "coeffs": coeffs,
        "extra": lammps.get("extra", {}),
        "box_lines": box_lines,
        "positions": vectors[0],
        "velocities": vectors[1] if velocities is not None else None,
        "image": image,
    }


def _header_text(system: System, plan: dict[str, Any], units: str) -> str:
    """The file up to the line before the first Atoms line."""
    from mdapy import __version__

    counts = plan["counts"]
    lines = [
        f"LAMMPS data file via mdapy {__version__}, "
        f"timestep = {int(system.info.get('timestep', 0))}, units = {units}",
        "",
        f"{len(system)} atoms",
        f"{counts['atom']} atom types",
    ]
    for kind in _ALL_TERMS:
        name = _TERMS[kind][0]
        number = len(plan["terms"][kind][1]) if kind in plan["terms"] else 0
        if number or counts[name]:
            lines += [f"{number} {kind}", f"{counts[name]} {name} types"]
    for name, value in plan["extra"].items():
        lines.append(f"{int(value)} extra {name} per atom")
    lines += [""] + plan["box_lines"]
    for name, labels in plan["labels"].items():
        lines += ["", f"{name.capitalize()} Type Labels", ""]
        lines += [f"{t} {label}" for t, label in enumerate(labels, start=1)]
    if plan["masses"]:
        lines += ["", "Masses", ""]
        for t in range(1, counts["atom"] + 1):
            symbol = plan["named"].get(t)
            comment = f" # {symbol}" if symbol else ""
            lines.append(f"{t} {_number(plan['masses'][t])}{comment}")
    for section, value in plan["coeffs"].items():
        style = f" # {value['style']}" if value.get("style") else ""
        lines += ["", f"{section}{style}", ""] + list(value["lines"])
    lines += ["", f"Atoms # {plan['style']}", "", ""]
    return "\n".join(lines)


def _write(
    file: BinaryIO,
    systems: Iterable[System],
    *,
    atom_style: str | None = None,
    units: str = "metal",
    elements: Sequence[str] | None = None,
    triclinic: str = "auto",
) -> None:
    if atom_style is not None and atom_style not in _STYLES:
        raise _unsupported_style(atom_style)
    if units not in _UNITS:
        raise InvalidParameter(f"units must be 'metal' or 'real', not {units!r}")
    if triclinic not in ("auto", "general", "restricted"):
        raise InvalidParameter(
            f"triclinic must be 'auto', 'general' or 'restricted', not {triclinic!r}"
        )
    for system in systems:
        plan = _plan(system, atom_style, units, elements, triclinic)
        table = system.table
        file.write(_header_text(system, plan, units).encode())
        columns: list[tuple] = [("i", plan["ids"])]
        for name in _STYLES[plan["style"]]:
            if name == "molecule":
                # molecule-ID 0 means "not part of a molecule" in LAMMPS
                values = (
                    table["molecule"].to_numpy()
                    if "molecule" in table.columns
                    else np.zeros(len(table), dtype=np.int64)
                )
                columns.append(("i", values))
            elif name == "type":
                columns.append(("i", plan["types"]))
            elif name == "charge":
                columns.append(("f", table["charge"].to_numpy()))
            else:
                columns.append(("f", plan["positions"]))
        if plan["image"] is not None:
            columns.append(("i", plan["image"]))
        write_rows(file, columns)
        if plan["velocities"] is not None:
            file.write(b"\nVelocities\n\n")
            shift = -3 if units == "real" else 0  # Å/ps to Å/fs, exactly
            write_rows(file, [("i", plan["ids"]), ("f", plan["velocities"], shift)])
        for kind, (atom_ids, numbers) in plan["terms"].items():
            file.write(f"\n{kind.capitalize()}\n\n".encode())
            write_rows(
                file,
                [
                    ("i", np.arange(1, len(numbers) + 1)),
                    ("i", numbers),
                    ("i", atom_ids),
                ],
            )


register(
    Format(
        name="lammps-data",
        description="LAMMPS data file (atom styles atomic, charge, bond, angle, "
        "molecular, full)",
        extensions=(".data", ".lmp"),
        patterns=("data.*",),
        read=_read,
        write=_write,
    )
)
