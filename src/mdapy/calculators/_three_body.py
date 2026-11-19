# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Potentials of three-body terms read from LAMMPS files: Tersoff and
Stillinger–Weber."""

import os
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from mdapy import _core
from mdapy.calculators import _gpu
from mdapy.calculators._base import Calculator, CalculatorResult, _element_indices
from mdapy.core.system import System
from mdapy.core.threads import _native_token
from mdapy.elements import SYMBOLS
from mdapy.errors import InvalidParameter
from mdapy.neighbor import NeighborList
from mdapy.neighbor._neighbor_list import _given_or_built

_native = _core.calculators


def _entries(path: str | os.PathLike, size: int, kind: str) -> list[list[str]]:
    """The entries of a LAMMPS potential file of three-body terms: the words
    after the comments, size to an entry, an entry over one or more lines."""
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError as error:
        raise InvalidParameter(f"cannot read {os.fspath(path)!r}: {error}") from None
    lines = text.splitlines()
    first = lines[0] if lines else ""
    if "UNITS:" in first:
        units = first.split("UNITS:")[1].split()
        if units and units[0] != "metal":
            raise InvalidParameter(
                f"{os.fspath(path)!r} is in {units[0]} units; MDAPY reads {kind} files in metal units"
            )
    words = [w for line in lines for w in line.split("#")[0].split()]
    if len(words) % size:
        raise InvalidParameter(
            f"{os.fspath(path)!r} is not a {kind} file: its {len(words)} values are not "
            f"entries of {size}"
        )
    return [words[k : k + size] for k in range(0, len(words), size)]


def _triplets(
    entries: list[list[str]],
    elements: Sequence[str] | Mapping[str, str] | None,
    count: int,
    path: str,
) -> tuple[tuple[str, ...], np.ndarray]:
    """The elements and the numbers of each triplet of them, (i, j, k) in the
    order of the elements, from entries of three labels and count numbers;
    elements maps the elements of the atoms to labels of the file."""
    table: dict[tuple[str, str, str], np.ndarray] = {}
    found: dict[str, None] = {}
    for entry in entries:
        labels = tuple(entry[:3])
        for label in labels:
            found[label] = None
        try:
            numbers = np.array([float(v) for v in entry[3:]])
        except ValueError:
            raise InvalidParameter(
                f"{path!r}: the entry {' '.join(entry)} has a word for a number"
            ) from None
        if len(numbers) != count or not np.isfinite(numbers).all():
            raise InvalidParameter(
                f"{path!r}: the entry {' '.join(entry)} is not {count} numbers"
            )
        if labels in table:
            raise InvalidParameter(f"{path!r} has two entries for {' '.join(labels)}")
        table[labels] = numbers
    if elements is None:
        names = tuple(found)
        labels = names
    elif isinstance(elements, Mapping):
        names, labels = tuple(elements), tuple(elements.values())
    else:
        names = (elements,) if isinstance(elements, str) else tuple(elements)
        labels = names
    if not names or len(set(names)) != len(names):
        raise InvalidParameter("elements must name each element once")
    for name in names:
        if name not in SYMBOLS:
            hint = (
                "; map the elements of the atoms to the labels of the file, as "
                "elements={'Si': 'Si(D)'}"
                if elements is None
                else ""
            )
            raise InvalidParameter(f"{name!r} is not an element{hint}")
    for label in labels:
        if label not in found:
            raise InvalidParameter(
                f"{path!r} has no entries of {label!r}; it has {list(found)}"
            )
    rows = []
    for a in labels:
        for b in labels:
            for c in labels:
                if (a, b, c) not in table:
                    raise InvalidParameter(f"{path!r} has no entry for {a} {b} {c}")
                rows.append(table[a, b, c])
    return names, np.array(rows)


class _ThreeBody(Calculator):
    """A potential of three-body terms computed natively."""

    _name: str
    _compute: Any
    _gpu_potential: Any = None  # on the GPU

    def _species(self, system: System) -> np.ndarray:
        """The index of each atom in the tables, (N,) int32."""
        return _element_indices(system, self.elements, self._name)

    @property
    def _on_device(self) -> Any:
        """The potential on the GPU, for its molecular dynamics there."""
        return self._gpu_potential

    def compute(
        self,
        system: System,
        properties: Iterable[str] = ("energy", "forces"),
        *,
        neighbors: NeighborList | None = None,
        progress: Any = None,
        cancel: Any = None,
    ) -> CalculatorResult:
        wanted = self._check_properties(properties)
        species = self._species(system)
        if (
            self._gpu_potential is not None
        ):  # with a neighbor list of its own on the GPU
            return self._gpu_potential.compute(system, species, wanted)
        neighbors = _given_or_built(system, self.cutoff, neighbors)
        with neighbors._reading():
            energies, forces, virials = type(self)._compute(
                self._potential,
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
        where = (
            f", device='cuda', precision={self.precision!r}"
            if self.device == "cuda"
            else ""
        )
        return f"{self._name}({self.path!r}, elements={self.elements}, cutoff={self.cutoff}{where})"


class Tersoff(_ThreeBody):
    """Tersoff potential of a LAMMPS ``.tersoff`` file, computed as LAMMPS
    ``pair_style tersoff`` computes it.

    Parameters
    ----------
    path : str or os.PathLike
        The potential file, in metal units: for each triplet of elements,
        m, γ, λ3, c, d, cos θ0, n, β, λ2, B, R, D, λ1 and A.
    elements : sequence of str or mapping, optional
        The elements to use, as the names after the file in ``pair_coeff`` of
        LAMMPS; every triplet of them needs an entry. A mapping takes the
        element of the atoms to a label of the file, as {"Si": "Si(D)"}.
        Default every element of the file.
    device : {"cpu", "cuda"}, optional
        Where it computes: on the processor, or on an NVIDIA GPU with a
        neighbor list of its own. Default "cpu".
    precision : {"mixed", "double"}, optional
        On the GPU: single precision for the terms and double for the
        positions and the sums, or double throughout. Default "mixed" on the
        GPU, "double" on the processor, which has no other.

    Attributes
    ----------
    path : str
        The potential file.
    elements : tuple of str
        The elements of the potential.
    cutoff : float64
        The largest cutoff R + D in Å.
    """

    _name = "Tersoff"
    _compute = staticmethod(_native.compute_tersoff)

    def __init__(
        self,
        path: str | os.PathLike,
        elements: Sequence[str] | Mapping[str, str] | None = None,
        *,
        device: str = "cpu",
        precision: str | None = None,
    ):
        self.device, self.precision = _gpu.resolve(device, precision)
        name = os.fspath(path)
        entries = _entries(path, 17, "Tersoff")
        chosen, rows = _triplets(entries, elements, 14, name)
        for (a, b, c), row in zip(
            ((a, b, c) for a in chosen for b in chosen for c in chosen),
            rows,
            strict=True,
        ):
            m, gamma, lam3, c_, d, h, n, beta, lam2, big_b, r, big_d, lam1, big_a = row
            if m not in (1.0, 3.0):
                raise InvalidParameter(
                    f"{name!r}: m of {a} {b} {c} must be 1 or 3, not {m}"
                )
            if min(c_, d, n, beta, lam2, big_b, r, big_d, lam1, big_a) < 0 or big_d > r:
                raise InvalidParameter(
                    f"{name!r}: the entry {a} {b} {c} has a negative parameter or D > R"
                )
        self.path = name
        self.elements = chosen
        self._potential = _native.make_tersoff(
            len(chosen), np.ascontiguousarray(rows.ravel())
        )
        self.cutoff = float(self._potential.cutoff)
        if self.device == "cuda":
            self._gpu_potential = _gpu.OnDevice(
                _gpu.cuda().make_tersoff(
                    len(chosen), np.ascontiguousarray(rows.ravel()), self.precision
                ),
                self.cutoff,
            )


class StillingerWeber(_ThreeBody):
    """Stillinger–Weber potential of a LAMMPS ``.sw`` file, computed as
    LAMMPS ``pair_style sw`` computes it.

    Parameters
    ----------
    path : str or os.PathLike
        The potential file, in metal units: for each triplet of elements,
        ε, σ, a, λ, γ, cos θ0, A, B, p, q and tol.
    elements : sequence of str or mapping, optional
        The elements to use, as the names after the file in ``pair_coeff`` of
        LAMMPS; every triplet of them needs an entry. A mapping takes the
        element of the atoms to a label of the file. Default every element of
        the file.
    device : {"cpu", "cuda"}, optional
        Where it computes: on the processor, or on an NVIDIA GPU with a
        neighbor list of its own. Default "cpu".
    precision : {"mixed", "double"}, optional
        On the GPU: single precision for the terms and double for the
        positions and the sums, or double throughout. Default "mixed" on the
        GPU, "double" on the processor, which has no other.

    Attributes
    ----------
    path : str
        The potential file.
    elements : tuple of str
        The elements of the potential.
    cutoff : float64
        The largest radius of a pair in Å: a σ, or less where tol is given.
    """

    _name = "StillingerWeber"
    _compute = staticmethod(_native.compute_stillinger_weber)

    def __init__(
        self,
        path: str | os.PathLike,
        elements: Sequence[str] | Mapping[str, str] | None = None,
        *,
        device: str = "cpu",
        precision: str | None = None,
    ):
        self.device, self.precision = _gpu.resolve(device, precision)
        name = os.fspath(path)
        entries = _entries(path, 14, "Stillinger–Weber")
        chosen, rows = _triplets(entries, elements, 11, name)
        triplets = ((a, b, c) for a in chosen for b in chosen for c in chosen)
        for (a, b, c), row in zip(triplets, rows, strict=True):
            if np.delete(row, 5).min() < 0:  # all but cos theta0
                raise InvalidParameter(
                    f"{name!r}: the entry {a} {b} {c} has a negative parameter"
                )
        self.path = name
        self.elements = chosen
        self._potential = _native.make_stillinger_weber(
            len(chosen), np.ascontiguousarray(rows.ravel())
        )
        self.cutoff = float(self._potential.cutoff)
        if self.device == "cuda":
            self._gpu_potential = _gpu.OnDevice(
                _gpu.cuda().make_stillinger_weber(
                    len(chosen), np.ascontiguousarray(rows.ravel()), self.precision
                ),
                self.cutoff,
            )
