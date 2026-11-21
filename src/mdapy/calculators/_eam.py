# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The embedded atom method, with potentials in the files of LAMMPS."""

import os
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from mdapy._core import calculators as _native
from mdapy.calculators import _gpu, _zhou
from mdapy.calculators._base import (
    HEAT_FLUX_TERMS,
    WITHOUT_VIRIAL,
    Calculator,
    CalculatorResult,
    _Part,
    _reversed_part,
)
from mdapy.core.system import System
from mdapy.core.threads import _native_token, get_num_threads
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


def _is_fs(lines: list[str], count: int, nrho: int, nr: int) -> bool:
    """Whether the setfl-like file holds a density per element and element of
    the neighbour (eam/fs) rather than one per element (eam/alloy): the two
    differ by count (count - 1) nr values, far more than the two to four
    words of the line of each element."""
    if count == 1:
        return False
    words = sum(len(line.split("#")[0].split()) for line in lines[5:])
    pairs = count * (count + 1) // 2 * nr
    return words >= count * (nrho + count * nr) + pairs + 2 * count


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
    fs = _is_fs(reader._lines, count, nrho, nr)
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
        # eam/fs: the density this element gives each element, in their order
        density += [reader.floats(nr) for _ in range(count if fs else 1)]
    pair = [reader.floats(nr) for i in range(count) for _ in range(i + 1)]
    density = np.array(density)
    if fs:
        given = density.reshape(count, count, nr)
        if np.array_equal(given, np.broadcast_to(given[:, :1], given.shape)):
            density, fs = given[:, 0], False  # the same for every receiver
    return {
        "elements": tuple(elements),
        "numbers": tuple(numbers),
        "masses": np.array(masses),
        "lattice_constants": np.array(constants),
        "lattices": tuple(lattices),
        "embedding": np.array(embedding),
        "density": density,
        "pair": np.array(pair),
        "drho": drho,
        "dr": dr,
        "cutoff": cutoff,
        "comments": comments,
        "fs": fs,
    }


def _is_funcfl(path: str | os.PathLike) -> bool:
    """Whether the file is a funcfl file of LAMMPS pair_style eam, of one
    element: its fourth line holds values, where that of a setfl file names
    the elements."""
    with open(path, encoding="utf-8") as f:
        lines = [f.readline() for _ in range(4)]
    words = lines[3].split("#")[0].split()
    return not (len(words) > 1 and words[1][:1].isalpha())


def _read_funcfl(path: str | os.PathLike) -> dict[str, Any]:
    """A funcfl file: a comment, the atomic number, mass, lattice constant
    and lattice, the grid, then F(rho), the effective charge Z(r) and
    rho(r)."""
    reader = _Numbers(path)
    reader.line = 1  # one comment line
    words = reader.next_line()
    try:
        number, mass = int(float(words[0])), float(words[1])
    except (ValueError, IndexError):
        raise FileFormatError(
            "expected the atomic number and the mass", reader.name, reader.line
        ) from None
    if not 0 < number < len(SYMBOLS):
        raise FileFormatError(
            f"{number} is not the atomic number of an element", reader.name, reader.line
        )
    try:
        constant = float(words[2])
    except (ValueError, IndexError):
        constant = 0.0
    lattice = words[3] if len(words) > 3 else ""
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
    return {
        "name": os.path.basename(os.fspath(path)),
        "number": number,
        "mass": mass,
        "constant": constant,
        "lattice": lattice,
        "drho": drho,
        "dr": dr,
        "cutoff": cutoff,
        "embedding": reader.floats(nrho),
        "charge": reader.floats(nr),
        "density": reader.floats(nr),
    }


#: Hartree times bohr in eV Å, as LAMMPS converts the effective charges.
_HARTREE_BOHR = 27.2 * 0.529


def _onto(values: np.ndarray, delta: float, count: int, step: float) -> np.ndarray:
    """A function tabulated every delta, at count points every step: by the
    four-point Lagrange interpolation of PairEAM::file2array_funcfl."""
    sixth = 1.0 / 6.0
    table = np.concatenate([[0.0], values])  # from 1, as in LAMMPS
    p = (np.arange(count) * step) / delta + 1.0
    k = np.maximum(np.minimum(p.astype(np.int64), len(values) - 2), 2)
    p = np.minimum(p - k, 2.0)
    cof1 = -sixth * p * (p - 1.0) * (p - 2.0)
    cof2 = 0.5 * (p * p - 1.0) * (p - 2.0)
    cof3 = -0.5 * p * (p + 1.0) * (p - 2.0)
    cof4 = sixth * p * (p * p - 1.0)
    return (
        cof1 * table[k - 1]
        + cof2 * table[k]
        + cof3 * table[k + 1]
        + cof4 * table[k + 2]
    )


def _funcfl_tables(files: list[dict[str, Any]]) -> dict[str, Any]:
    """The tables LAMMPS pair_style eam makes of funcfl files, one element
    each: on the grid of the largest spacings and ranges, the pair of two
    elements from their effective charges, phi = 27.2 0.529 Z_a Z_b / r."""
    elements = tuple(SYMBOLS[f["number"]] for f in files)
    if len(set(elements)) != len(elements):
        raise InvalidParameter(
            f"the elements of the funcfl files {list(elements)} repeat"
        )
    dr = max(f["dr"] for f in files)
    drho = max(f["drho"] for f in files)
    r_max = max((len(f["density"]) - 1) * f["dr"] for f in files)
    rho_max = max((len(f["embedding"]) - 1) * f["drho"] for f in files)
    nr, nrho = int(np.floor(r_max / dr + 0.5)), int(np.floor(rho_max / drho + 0.5))
    charges = [_onto(f["charge"], f["dr"], nr, dr) for f in files]
    pair = [
        _HARTREE_BOHR * charges[i] * charges[j]
        for i in range(len(files))
        for j in range(i + 1)
    ]
    return {
        "elements": elements,
        "numbers": tuple(f["number"] for f in files),
        "masses": np.array([f["mass"] for f in files]),
        "lattice_constants": np.array([f["constant"] for f in files]),
        "lattices": tuple(f["lattice"] for f in files),
        "embedding": np.array(
            [_onto(f["embedding"], f["drho"], nrho, drho) for f in files]
        ),
        "density": np.array([_onto(f["density"], f["dr"], nr, dr) for f in files]),
        "pair": np.array(pair),
        "drho": drho,
        "dr": dr,
        "cutoff": max(f["cutoff"] for f in files),
        # the embedding energy goes on linearly beyond, one step past its table
        "rho_max": rho_max,
        "comments": (
            "converted by MDAPY as LAMMPS pair_style eam does, from the funcfl files",
            " ".join(f["name"] for f in files),
            "",
        ),
        "fs": False,
    }


def _lines(values: np.ndarray, per_line: int = 5) -> list[str]:
    text = [f"{v:.16e}" for v in values]
    return [" ".join(text[k : k + per_line]) for k in range(0, len(text), per_line)]


class EAM(Calculator):
    """Embedded atom method potential, in the LAMMPS ``eam/alloy`` (setfl),
    ``eam/fs`` (Finnis-Sinclair) or ``eam`` (funcfl) form.

    Parameters
    ----------
    path : str, os.PathLike or sequence of them
        The potential file: ``eam/alloy`` or ``eam/fs``, told apart by the
        number of values, as an ``eam/fs`` file holds the density each
        element gives each element; or one or several funcfl files of
        ``pair_style eam``, one element each, mixed as LAMMPS mixes them.
    device : {"cpu", "cuda"}, optional
        Where it computes: on the processor, or on an NVIDIA GPU with a
        neighbor list of its own. Default "cpu".
    precision : {"mixed", "double"}, optional
        On the GPU: single precision for the terms of each pair and double
        for the positions and the sums, or double throughout. Default
        "mixed" on the GPU, "double" on the processor, which has no other.

    Attributes
    ----------
    path : str, tuple of str or None
        The file the potential was read from, the funcfl files, or None for
        a built one.
    elements : tuple of str or None
        The elements of the potential, in the order of the file; None for an
        average-atom potential, which handles atoms of any element.
    masses : numpy.ndarray
        (E,) float64, their masses in amu; the mass of the average atom for an
        average-atom potential.
    cutoff : float64
        Cutoff radius in Å.
    """

    _takes_wider_lists = True

    # its forces are pairwise, and their virials split in halves
    _heat_flux = True

    def __init__(
        self,
        path: str | os.PathLike | Sequence[str | os.PathLike],
        *,
        device: str = "cpu",
        precision: str | None = None,
    ):
        placement = _gpu.resolve(device, precision)
        paths = [path] if isinstance(path, (str, os.PathLike)) else list(path)
        if not paths:
            raise InvalidParameter("give the file of the potential")
        if len(paths) == 1 and not _is_funcfl(paths[0]):
            self.__set(_read_setfl(paths[0]), os.fspath(paths[0]), placement=placement)
            return
        if not all(_is_funcfl(p) for p in paths):
            raise InvalidParameter(
                "several files are funcfl files of pair_style eam, one element each"
            )
        names = tuple(os.fspath(p) for p in paths)
        self.__set(
            _funcfl_tables([_read_funcfl(p) for p in paths]),
            names[0] if len(names) == 1 else names,
            placement=placement,
        )

    def __set(
        self,
        tables: dict[str, Any],
        path: str | tuple[str, ...] | None,
        average: bool = False,
        placement: tuple[str, str] = ("cpu", "double"),
    ):
        self.path = path
        self.__tables = tables
        self.__average = average
        names = tables["elements"]
        self.cutoff = tables["cutoff"]
        self.__potential = _native.make_eam(
            tables["embedding"],
            tables["density"],
            tables["pair"],
            tables["drho"],
            tables["dr"],
            tables["cutoff"],
            rho_max=tables.get("rho_max"),
        )
        self.device, self.precision = placement
        self.__gpu: _gpu.OnDevice | None = None
        if self.device == "cuda":
            self.__gpu = _gpu.OnDevice(
                _gpu.cuda().make_eam(
                    tables["embedding"],
                    tables["density"],
                    tables["pair"],
                    tables["drho"],
                    tables["dr"],
                    tables["cutoff"],
                    self.__potential.rho_max,
                    self.precision,
                ),
                self.cutoff,
            )
        self.__species = np.full(len(SYMBOLS), -1, dtype=np.int32)
        if average:
            self.elements = None
            self.masses = tables["masses"][-1:].copy()
            self.__species[:] = len(names) - 1
        else:
            known = [k for k, name in enumerate(names) if name in SYMBOLS]
            self.elements = tuple(names[k] for k in known)
            self.masses = tables["masses"][known]
            for k in reversed(known):  # the first of repeated names
                self.__species[SYMBOLS.index(names[k])] = k

    @classmethod
    def __built(
        cls,
        tables: dict[str, Any],
        average: bool = False,
        placement: tuple[str, str] = ("cpu", "double"),
    ) -> "EAM":
        potential = cls.__new__(cls)
        potential.__set(tables, None, average, placement)
        return potential

    @classmethod
    def zhou(
        cls,
        elements: Sequence[str],
        *,
        nr: int = 2000,
        nrho: int = 2000,
        device: str = "cpu",
        precision: str | None = None,
    ) -> "EAM":
        """The potential of Zhou, Johnson and Wadley for elements, tabulated as
        the LAMMPS tool tools/eam_database does.

        Parameters
        ----------
        elements : sequence of str
            Any of Cu, Ag, Au, Ni, Pd, Pt, Al, Pb, Fe, Mo, Ta, W, Mg, Co, Ti,
            Zr and Cr.
        nr, nrho : int32, optional
            Number of points of the tables in distance and in density.
            Default 2000.
        device : {"cpu", "cuda"}, optional
            Where it computes, as for ``EAM``. Default "cpu".
        precision : {"mixed", "double"}, optional
            Its precision, as for ``EAM``.

        Returns
        -------
        EAM
            The potential, written to a file with ``write``.
        """
        placement = _gpu.resolve(device, precision)
        return cls.__built(_zhou.tables(list(elements), nr, nrho), placement=placement)

    def average(self, fractions: Mapping[str, float] | Sequence[float]) -> "EAM":
        """The average-atom potential of an alloy of the elements.

        Parameters
        ----------
        fractions : mapping or sequence of float64
            The fraction of each element, by element or in the order of
            ``elements``; elements not given have none. They sum to 1.

        Returns
        -------
        EAM
            A potential in which every atom is the average atom, whatever its
            element. ``write`` writes it with the elements of this
            potential and the average atom, named ``A``, last.
        """
        if self.__average:
            raise InvalidParameter("the potential is already an average-atom potential")
        if self.__tables.get("fs", False):
            raise InvalidParameter(
                "an average atom needs the density each element gives to be the "
                "same for every element it gives it to, as in eam/alloy; this "
                "eam/fs potential gives each element its own"
            )
        assert self.elements is not None
        if isinstance(fractions, Mapping):
            unknown = set(fractions) - set(self.elements)
            if unknown:
                raise InvalidParameter(
                    f"the potential has no {sorted(unknown)}; it has {list(self.elements)}"
                )
            given = [float(fractions.get(e, 0.0)) for e in self.elements]
        else:
            given = [float(c) for c in fractions]
            if len(given) != len(self.elements):
                raise InvalidParameter(
                    f"expected {len(self.elements)} fractions, one per element "
                    f"of {list(self.elements)}"
                )
        c = np.array(given)
        if (c < 0).any() or not np.isclose(c.sum(), 1.0, rtol=0, atol=1e-10):
            raise InvalidParameter(
                f"fractions must be non-negative and sum to 1, not {given}"
            )
        t = self.__tables
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
        mixed = np.einsum("a,abr->br", weights, pair)  # average atom with each element
        tables = dict(t)
        tables["elements"] = names + ("A",)
        tables["numbers"] = t["numbers"] + (0,)
        tables["masses"] = np.append(
            t["masses"], np.einsum("a,a->", weights, t["masses"])
        )
        tables["lattice_constants"] = np.append(t["lattice_constants"], 0.0)
        tables["lattices"] = t["lattices"] + ("",)
        tables["embedding"] = np.vstack(
            [t["embedding"], np.einsum("a,ar->r", weights, t["embedding"])]
        )
        tables["density"] = np.vstack(
            [t["density"], np.einsum("a,ar->r", weights, t["density"])]
        )
        tables["pair"] = np.vstack(
            [t["pair"], mixed, np.einsum("a,ar->r", weights, mixed)]
        )
        return EAM.__built(
            tables, average=True, placement=(self.device, self.precision)
        )

    def write(self, path: str | os.PathLike) -> None:
        """Writes the potential as a LAMMPS ``eam/alloy`` (setfl) file, or
        an ``eam/fs`` file for a potential read from one.

        Parameters
        ----------
        path : str or os.PathLike
            The file to write; it reads back to the same tables.
        """
        t = self.__tables
        names = t["elements"]
        per_element = len(names) if t.get("fs", False) else 1
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
            lines += _lines(t["embedding"][k])
            for row in t["density"][k * per_element : (k + 1) * per_element]:
                lines += _lines(row)
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
        properties = tuple(properties)
        heat = HEAT_FLUX_TERMS in properties
        wanted = self._check_properties(properties)
        species = self._species(system)
        if self.__gpu is not None:  # with a neighbor list of its own on the GPU
            return self.__gpu.compute(system, species, wanted, heat=heat)
        # the virials unless left out, as the dynamics may
        virial = WITHOUT_VIRIAL not in properties or heat
        neighbors = _given_or_built(system, self.cutoff, neighbors)
        with neighbors._reading():
            energies, forces, virials = _native.compute_eam(
                self.__potential,
                species,
                np.ascontiguousarray(system["positions"]),
                np.ascontiguousarray(system.box.matrix),
                neighbors.offsets,
                neighbors.j,
                neighbors.shifts,
                progress,
                _native_token(cancel),
                heat,
                virial,
            )
        return CalculatorResult(
            energy=float(energies.sum()),
            atomic_energies=energies,
            forces=forces,
            virial=virials.sum(axis=0) if virials is not None else None,
            atomic_virials=virials,
            volume=system.box.volume,
        )

    def _split(
        self,
        system: System,
        properties: tuple,
        neighbors: NeighborList,
        owned: int,
        forward: Any,
        reverse: Any,
        ids: np.ndarray,
    ) -> _Part:
        """The part of the first owned atoms of system, a process's of MPI:
        their densities, the slopes of their embedding functions sent to
        their ghosts by forward, then their forces, whose pairs reach the
        ghosts within the cutoff."""
        heat = HEAT_FLUX_TERMS in properties
        virial = (
            bool({"virial", "atomic_virials"} & set(properties))
            and WITHOUT_VIRIAL not in properties
        ) or heat
        arrays = (
            self._species(system),
            np.ascontiguousarray(system["positions"]),
            np.ascontiguousarray(system.box.matrix),
            neighbors.offsets,
            neighbors.j,
            neighbors.shifts,
        )
        if get_num_threads() == 1:
            # on one thread each pair once among the processes, as LAMMPS's
            # newton on: the densities of the ghosts back to their atoms, the
            # slopes to the ghosts, the forces on the ghosts back
            ids = np.ascontiguousarray(ids, dtype=np.int64)
            with neighbors._reading():
                rho = _native.eam_densities_once(self.__potential, *arrays, owned, ids)
            rho = np.ascontiguousarray(reverse(rho))
            embedding, slopes = _native.eam_embedding(self.__potential, arrays[0], rho)
            slopes = np.ascontiguousarray(forward(slopes))
            with neighbors._reading():
                energies, forces, virials = _native.eam_forces_once(
                    self.__potential, *arrays, owned, ids, slopes, virial, heat
                )
            energies[:owned] += embedding
            return _reversed_part(energies, forces, virials, properties, reverse)
        with neighbors._reading():
            embedding, slopes = _native.eam_densities(self.__potential, *arrays, owned)
            slopes = np.ascontiguousarray(forward(slopes))
            energies, forces, virials = _native.eam_forces(
                self.__potential, *arrays, slopes, embedding, virial, heat
            )
        return _Part(
            float(energies.sum()),
            forces,
            virials.sum(axis=0) if virials is not None else np.full((3, 3), np.nan),
            energies,
            virials,
        )

    def _species(self, system: System) -> np.ndarray:
        """The index of each atom in the tables, (N,) int32."""
        if self.__average:
            return np.full(len(system), len(self.__tables["elements"]) - 1, np.int32)
        if "element" not in system.table.columns:
            raise MissingProperty("EAM needs the element column")
        numbers = system.table["element"].to_physical().to_numpy()
        species = self.__species[numbers]
        if (species < 0).any():
            missing = sorted({SYMBOLS[z] for z in numbers[species < 0]})
            raise InvalidParameter(
                f"the potential has no parameters for {missing}; it has {list(self.elements or ())}"
            )
        return species

    @property
    def _on_device(self) -> _gpu.OnDevice | None:
        """The potential on the GPU, for its molecular dynamics there."""
        return self.__gpu

    def __repr__(self) -> str:
        source = f"{self.path!r}, " if self.path is not None else ""
        where = (
            f", device='cuda', precision={self.precision!r}"
            if self.device == "cuda"
            else ""
        )
        if self.__average:
            return f"EAM({source}average atom, cutoff={self.cutoff}{where})"
        return f"EAM({source}elements={self.elements}, cutoff={self.cutoff}{where})"
