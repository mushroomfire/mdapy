# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""The embedded atom method, with potentials in the files of LAMMPS."""

import os
from collections.abc import Iterable
from typing import Any

import numpy as np

from mdapy._core import calculators as _native
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
    for element in elements:
        if element not in SYMBOLS:
            raise FileFormatError(
                f"unknown element {element!r}", reader.name, reader.line
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
    masses, embedding, density = [], [], []
    for _ in elements:
        words = reader.next_line()
        try:
            masses.append(float(words[1]))
        except (ValueError, IndexError):
            raise FileFormatError(
                "expected the atomic number and the mass", reader.name, reader.line
            ) from None
        embedding.append(reader.floats(nrho))
        density.append(reader.floats(nr))
    pair = [reader.floats(nr) for i in range(count) for _ in range(i + 1)]
    return {
        "elements": tuple(elements),
        "masses": np.array(masses),
        "embedding": np.array(embedding),
        "density": np.array(density),
        "pair": np.array(pair),
        "drho": drho,
        "dr": dr,
        "cutoff": cutoff,
    }


class EAM(Calculator):
    """Embedded atom method potential of a LAMMPS ``eam/alloy`` (setfl) file.

    Parameters
    ----------
    path : str or os.PathLike
        The potential file.

    Attributes
    ----------
    elements : tuple of str
        The elements of the potential, in the order of the file.
    masses : numpy.ndarray
        Their masses in amu.
    cutoff : float
        Cutoff radius in Å.
    """

    def __init__(self, path: str | os.PathLike):
        data = _read_setfl(path)
        self.path = os.fspath(path)
        self.elements = data["elements"]
        self.masses = data["masses"]
        self.cutoff = data["cutoff"]
        self._potential = _native.make_eam(
            data["embedding"],
            data["density"],
            data["pair"],
            data["drho"],
            data["dr"],
            data["cutoff"],
        )
        self._species = np.full(len(SYMBOLS), -1, dtype=np.int32)
        for index, element in enumerate(self.elements):
            self._species[SYMBOLS.index(element)] = index

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
        if "element" not in system.table.columns:
            raise MissingProperty("EAM needs the element column")
        numbers = system.table["element"].to_physical().to_numpy()
        species = self._species[numbers]
        if (species < 0).any():
            missing = sorted({SYMBOLS[z] for z in numbers[species < 0]})
            raise InvalidParameter(
                f"the potential has no parameters for {missing}; it has {list(self.elements)}"
            )
        neighbors = _given_or_built(system, self.cutoff, neighbors)
        energies, forces, virials = _native.compute_eam(
            self._potential,
            species,
            neighbors.offsets,
            neighbors.j,
            np.ascontiguousarray(neighbors.vectors),
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
        return f"EAM({self.path!r}, elements={self.elements}, cutoff={self.cutoff})"
