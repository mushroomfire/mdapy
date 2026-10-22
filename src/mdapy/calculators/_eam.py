# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""The embedded atom method, with potentials in the files of LAMMPS."""

import os
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from mdapy._core import calculators as _native
from mdapy.calculators import _zhou
from mdapy.calculators._base import Calculator, CalculatorResult
from mdapy.core.system import System
from mdapy.core.threads import _native_token
from mdapy.elements import SYMBOLS
from mdapy.errors import FileFormatError, InvalidParameter, MissingProperty
from mdapy.neighbor import NeighborList
from mdapy.neighbor._neighbor_list import _given_or_built


class _Numbers:
    """Reads a potential file as LAMMPS does: values may continue on further
    lines, values left on the last line of a block are ignored, and text
    after '#' is a comment."""

    def __init__(self, path: str | os.PathLike):
        self.name = os.fspath(path)
        with open(path, encoding="utf-8") as f:
            self._lines = f.read().splitlines()
        self.line = 0

    def next_line(self) -> list[str]:
        while self.line < len(self._lines):
            self.line += 1
            words = self._lines[self.line - 1].split("#")[0].split()
            if words:
                return words
        raise FileFormatError("unexpected end of file", self.name, self.line + 1)

    def floats(self, count: int) -> np.ndarray:
        values: list[str] = []
        while len(values) < count:
            values.extend(self.next_line()[: count - len(values)])
        try:
            return np.array([float(v) for v in values])
        except ValueError:
            raise FileFormatError("expected numbers", self.name, self.line) from None


def _read_setfl(path: str | os.PathLike) -> dict[str, Any]:
    reader = _Numbers(path)
    comments = tuple(reader._lines[:3]) + ("",) * (3 - len(reader._lines[:3]))
    reader.line = 3  # three comment lines
    words = reader.next_line()
    try:
        count = int(words[0])
    except ValueError:
        raise FileFormatError(
            "expected the number of elements", reader.name, reader.line
        ) from None
    elements = words[1:]
    if len(elements) != count or count < 1:
        raise FileFormatError(
            f"expected {count} element names, found {len(elements)}",
            reader.name,
            reader.line,
        )
    words = reader.next_line()
    try:
        nrho, drho, nr, dr, cutoff = (
            int(words[0]),
            float(words[1]),
            int(words[2]),
            float(words[3]),
            float(words[4]),
        )
    except (ValueError, IndexError):
        raise FileFormatError(
            "expected Nrho drho Nr dr cutoff", reader.name, reader.line
        ) from None
    if nrho < 5 or nr < 5 or drho <= 0 or dr <= 0 or cutoff <= 0:
        raise FileFormatError(
            "Nrho and Nr must be at least 5, and drho, dr and cutoff positive",
            reader.name,
            reader.line,
        )
    numbers, masses, constants, lattices = [], [], [], []
    embedding, density = [], []
    for _ in elements:
        words = reader.next_line()
        try:
            numbers.append(int(float(words[0])))
            masses.append(float(words[1]))
        except (ValueError, IndexError):
            raise FileFormatError(
                "expected the atomic number and the mass", reader.name, reader.line
            ) from None
        try:
            constants.append(float(words[2]))
        except (ValueError, IndexError):
            constants.append(0.0)
        lattices.append(words[3] if len(words) > 3 else "")
        embedding.append(reader.floats(nrho))
        density.append(reader.floats(nr))
    pair = [reader.floats(nr) for i in range(count) for _ in range(i + 1)]
    return {
        "elements": tuple(elements),
        "numbers": tuple(numbers),
        "masses": np.array(masses),
        "lattice_constants": np.array(constants),
        "lattices": tuple(lattices),
        "embedding": np.array(embedding),
        "density": np.array(density),
        "pair": np.array(pair),
        "drho": drho,
        "dr": dr,
        "cutoff": cutoff,
        "comments": comments,
    }


def _lines(values: np.ndarray, per_line: int = 5) -> list[str]:
    text = [f"{v:.16e}" for v in values]
    return [" ".join(text[k : k + per_line]) for k in range(0, len(text), per_line)]


class EAM(Calculator):
    """Embedded atom method potential, in the LAMMPS ``eam/alloy`` (setfl) form.

    Parameters
    ----------
    path : str or os.PathLike
        The potential file.

    Attributes
    ----------
    path : str or None
        The file the potential was read from, None for a built one.
    elements : tuple of str or None
        The elements of the potential, in the order of the file; None for an
        average-atom potential, which handles atoms of any element.
    masses : numpy.ndarray
        Their masses in amu; the mass of the average atom for an
        average-atom potential.
    cutoff : float
        Cutoff radius in Å.
    """

    def __init__(self, path: str | os.PathLike):
        self._set(_read_setfl(path), os.fspath(path))

    def _set(self, tables: dict[str, Any], path: str | None, average: bool = False):
        self.path = path
        self._tables = tables
        self._average = average
        names = tables["elements"]
        self.cutoff = tables["cutoff"]
        self._potential = _native.make_eam(
            tables["embedding"],
            tables["density"],
            tables["pair"],
            tables["drho"],
            tables["dr"],
            tables["cutoff"],
        )
        self._species = np.full(len(SYMBOLS), -1, dtype=np.int32)
        if average:
            self.elements = None
            self.masses = tables["masses"][-1:].copy()
            self._species[:] = len(names) - 1
        else:
            known = [k for k, name in enumerate(names) if name in SYMBOLS]
            self.elements = tuple(names[k] for k in known)
            self.masses = tables["masses"][known]
            for k in reversed(known):  # the first of repeated names
                self._species[SYMBOLS.index(names[k])] = k

    @classmethod
    def _built(cls, tables: dict[str, Any], average: bool = False) -> "EAM":
        potential = cls.__new__(cls)
        potential._set(tables, None, average)
        return potential

    @classmethod
    def zhou(
        cls, elements: Sequence[str], *, nr: int = 2000, nrho: int = 2000
    ) -> "EAM":
        """The potential of Zhou, Johnson and Wadley for elements, tabulated as
        the LAMMPS tool tools/eam_database does.

        Parameters
        ----------
        elements : sequence of str
            Any of Cu, Ag, Au, Ni, Pd, Pt, Al, Pb, Fe, Mo, Ta, W, Mg, Co, Ti,
            Zr and Cr.
        nr, nrho : int, optional
            Number of points of the tables in distance and in density.
            Default 2000.

        Returns
        -------
        EAM
            The potential, written to a file with :meth:`write`.
        """
        return cls._built(_zhou.tables(list(elements), nr, nrho))

    def average(self, concentrations: Mapping[str, float] | Sequence[float]) -> "EAM":
        """The average-atom potential of an alloy of the elements.

        Parameters
        ----------
        concentrations : mapping or sequence of float
            The fraction of each element, by element or in the order of
            ``elements``; elements not given have none. They sum to 1.

        Returns
        -------
        EAM
            A potential in which every atom is the average atom, whatever its
            element. :meth:`write` writes it with the elements of this
            potential and the average atom, named ``A``, last.
        """
        if self._average:
            raise InvalidParameter("the potential is already an average-atom potential")
        assert self.elements is not None
        if isinstance(concentrations, Mapping):
            unknown = set(concentrations) - set(self.elements)
            if unknown:
                raise InvalidParameter(
                    f"the potential has no {sorted(unknown)}; it has {list(self.elements)}"
                )
            fractions = [float(concentrations.get(e, 0.0)) for e in self.elements]
        else:
            fractions = [float(c) for c in concentrations]
            if len(fractions) != len(self.elements):
                raise InvalidParameter(
                    f"expected {len(self.elements)} concentrations, one per element "
                    f"of {list(self.elements)}"
                )
        c = np.array(fractions)
        if (c < 0).any() or not np.isclose(c.sum(), 1.0, rtol=0, atol=1e-10):
            raise InvalidParameter(
                f"concentrations must be non-negative and sum to 1, not {fractions}"
            )
        t = self._tables
        names = t["elements"]
        n = len(names)
        weights = np.zeros(n)
        for e, fraction in zip(self.elements, c):
            weights[names.index(e)] = fraction
        # pair tables: the lower triangle, row by row
        index = {
            (i, j): k
            for k, (i, j) in enumerate((i, j) for i in range(n) for j in range(i + 1))
        }
        pair = [
            t["pair"][index[max(i, j), min(i, j)]] for i in range(n) for j in range(n)
        ]
        pair = np.array(pair).reshape(n, n, -1)
        mixed = np.tensordot(weights, pair, axes=1)  # average atom with each element
        tables = dict(t)
        tables["elements"] = names + ("A",)
        tables["numbers"] = t["numbers"] + (0,)
        tables["masses"] = np.append(t["masses"], weights @ t["masses"])
        tables["lattice_constants"] = np.append(t["lattice_constants"], 0.0)
        tables["lattices"] = t["lattices"] + ("",)
        tables["embedding"] = np.vstack([t["embedding"], weights @ t["embedding"]])
        tables["density"] = np.vstack([t["density"], weights @ t["density"]])
        tables["pair"] = np.vstack([t["pair"], mixed, weights @ mixed])
        return EAM._built(tables, average=True)

    def write(self, path: str | os.PathLike) -> None:
        """Writes the potential as a LAMMPS ``eam/alloy`` (setfl) file.

        Parameters
        ----------
        path : str or os.PathLike
            The file to write; it reads back to the same tables.
        """
        t = self._tables
        names = t["elements"]
        lines = [line.replace("\n", " ") for line in t["comments"]]
        lines.append(f"{len(names)} {' '.join(names)}")
        lines.append(
            f"{t['embedding'].shape[1]} {t['drho']:.16e} {t['density'].shape[1]} "
            f"{t['dr']:.16e} {t['cutoff']:.16e}"
        )
        for k in range(len(names)):
            lines.append(
                f"{t['numbers'][k]} {t['masses'][k]:.16e} "
                f"{t['lattice_constants'][k]:.16e} {t['lattices'][k] or 'none'}"
            )
            lines += _lines(t["embedding"][k]) + _lines(t["density"][k])
        for row in t["pair"]:
            lines += _lines(row)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def compute(
        self,
        system: System,
        properties: Iterable[str] = ("energy", "forces"),
        *,
        neighbors: NeighborList | None = None,
        progress: Any = None,
        cancel: Any = None,
    ) -> CalculatorResult:
        self._check_properties(properties)
        if self._average:
            species = np.full(len(system), len(self._tables["elements"]) - 1, np.int32)
        else:
            if "element" not in system.table.columns:
                raise MissingProperty("EAM needs the element column")
            numbers = system.table["element"].to_physical().to_numpy()
            species = self._species[numbers]
            if (species < 0).any():
                missing = sorted({SYMBOLS[z] for z in numbers[species < 0]})
                raise InvalidParameter(
                    f"the potential has no parameters for {missing}; it has {list(self.elements or ())}"
                )
        neighbors = _given_or_built(system, self.cutoff, neighbors)
        energies, forces, virials = _native.compute_eam(
            self._potential,
            species,
            np.ascontiguousarray(system.positions),
            np.ascontiguousarray(system.box.matrix),
            neighbors.offsets,
            neighbors.j,
            neighbors.shifts,
            progress,
            _native_token(cancel),
        )
        return CalculatorResult(
            energy=float(energies.sum()),
            atomic_energies=energies,
            forces=forces,
            virial=virials.sum(axis=0),
            atomic_virials=virials,
        )

    def __repr__(self) -> str:
        source = f"{self.path!r}, " if self.path is not None else ""
        if self._average:
            return f"EAM({source}average atom, cutoff={self.cutoff})"
        return f"EAM({source}elements={self.elements}, cutoff={self.cutoff})"
