# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The angular-dependent potential (ADP) of LAMMPS files."""

import os
from collections.abc import Iterable
from typing import Any

import numpy as np

from mdapy import _core
from mdapy.calculators._base import Calculator, CalculatorResult, _element_indices
from mdapy.calculators._eam import _Numbers
from mdapy.core.system import System
from mdapy.core.threads import _native_token
from mdapy.elements import SYMBOLS
from mdapy.errors import FileFormatError, InvalidParameter
from mdapy.neighbor import NeighborList
from mdapy.neighbor._neighbor_list import _given_or_built

_native = _core.calculators


def _read_adp(path: str | os.PathLike) -> dict[str, Any]:
    """The tables of an ADP file: an eam/alloy (setfl) file followed by the
    dipole function u(r) and the quadrupole function w(r) of each pair."""
    try:
        reader = _Numbers(path)
    except OSError as error:
        raise InvalidParameter(f"cannot read {os.fspath(path)!r}: {error}") from None
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
    tables = {}
    for name in ("pair", "dipole", "quadrupole"):
        tables[name] = np.array(
            [reader.floats(nr) for i in range(count) for _ in range(i + 1)]
        )
    return {
        "elements": tuple(elements),
        "masses": np.array(masses),
        "embedding": np.array(embedding),
        "density": np.array(density),
        "drho": drho,
        "dr": dr,
        "cutoff": cutoff,
        **tables,
    }


class ADP(Calculator):
    """Angular-dependent potential of a LAMMPS file, computed as LAMMPS
    ``pair_style adp`` computes it: the embedded atom method with the dipole
    and the quadrupole of the neighbors of each atom.

    Parameters
    ----------
    path : str or os.PathLike
        The potential file, as an ``eam/alloy`` file followed by the dipole
        function u(r) and the quadrupole function w(r) of each pair of
        elements.

    Attributes
    ----------
    path : str
        The potential file.
    elements : tuple of str
        The elements of the potential, in the order of the file.
    masses : numpy.ndarray
        (E,) float64, their masses in amu.
    cutoff : float64
        Cutoff radius in Å.
    """

    def __init__(self, path: str | os.PathLike):
        tables = _read_adp(path)
        for name in tables["elements"]:
            if name not in SYMBOLS:
                raise InvalidParameter(
                    f"{os.fspath(path)!r}: {name!r} is not an element"
                )
        self.path = os.fspath(path)
        self.elements = tables["elements"]
        self.masses = tables["masses"]
        self.cutoff = float(tables["cutoff"])
        self.__potential = _native.make_adp(
            tables["embedding"],
            tables["density"],
            tables["pair"],
            tables["dipole"],
            tables["quadrupole"],
            tables["drho"],
            tables["dr"],
            tables["cutoff"],
        )

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
        species = _element_indices(system, self.elements, "ADP")
        neighbors = _given_or_built(system, self.cutoff, neighbors)
        with neighbors._reading():
            energies, forces, virials = _native.compute_adp(
                self.__potential,
                species,
                np.ascontiguousarray(system["positions"]),
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
            volume=system.box.volume,
        )

    def __repr__(self) -> str:
        return f"ADP({self.path!r}, elements={self.elements}, cutoff={self.cutoff})"
