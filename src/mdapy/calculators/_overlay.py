# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Calculators added together, as LAMMPS pair_style hybrid/overlay."""

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from mdapy.calculators import _gpu
from mdapy.calculators._base import (
    HEAT_FLUX_TERMS,
    PROPERTIES,
    Calculator,
    CalculatorResult,
    _Part,
)
from mdapy.core.system import System
from mdapy.errors import InvalidParameter
from mdapy.neighbor import NeighborList
from mdapy.neighbor._neighbor_list import _given_or_built


class Overlay(Calculator):
    """Several calculators added together: the energy, the forces and the
    virials of each summed, as LAMMPS ``pair_style hybrid/overlay``. A
    `ZBL` overlaid on a potential gives it the close collisions of a cascade
    of radiation damage.

    Parameters
    ----------
    calculators : sequence of Calculator
        Two or more calculators, all on the processor or all on the GPU.
        On the GPU, MDAPY's potentials with their elements in the same order.

    Attributes
    ----------
    calculators : tuple of Calculator
        The calculators, in their order.
    cutoff : float64
        The largest of their cutoffs in Å.
    elements : tuple of str or None
        The elements of all of them, in the order they first come; None when
        one takes any element.
    implemented : frozenset of str
        The properties every one of them computes.
    """

    def __init__(self, calculators: Sequence[Calculator]):
        if isinstance(calculators, Calculator) or not isinstance(calculators, Iterable):
            raise InvalidParameter(
                f"calculators must be a sequence of calculators, not {calculators!r}"
            )
        parts = tuple(calculators)
        if len(parts) < 2 or not all(isinstance(p, Calculator) for p in parts):
            raise InvalidParameter("an Overlay adds two calculators or more")
        devices = {p.device for p in parts}
        if len(devices) != 1:
            raise InvalidParameter(
                "the calculators of an Overlay must all be on the processor or all "
                "on the GPU"
            )
        self.calculators = parts
        self.device = parts[0].device
        self.precision = (
            "mixed" if any(p.precision == "mixed" for p in parts) else "double"
        )
        self.cutoff = float(max(p.cutoff for p in parts))
        if any(p.elements is None for p in parts):
            self.elements = None
        else:
            self.elements = tuple(dict.fromkeys(e for p in parts for e in p.elements))
        self.implemented = frozenset.intersection(
            PROPERTIES, *(p.implemented for p in parts)
        )
        self._heat_flux = all(p._heat_flux for p in parts)
        self._takes_wider_lists = all(p._takes_wider_lists for p in parts)
        # on many processes, the parts of each process added, when every
        # calculator splits and they share their elements, so their species
        self._split = (
            self.__split
            if self.device == "cpu"
            and all(callable(getattr(p, "_split", None)) for p in parts)
            and len({p.elements for p in parts}) == 1
            else None
        )
        self._pairs_once = all(getattr(p, "_pairs_once", False) for p in parts)
        self._heat_needs_ghosts = any(
            getattr(p, "_heat_needs_ghosts", False) for p in parts
        )
        self._once_with_rows = any(getattr(p, "_once_with_rows", False) for p in parts)
        self.__gpu: _gpu.OnDevice | None = None
        if self.device == "cuda":
            on = [getattr(p, "_on_device", None) for p in parts]
            if any(o is None for o in on):
                raise InvalidParameter(
                    'on the GPU an Overlay adds MDAPY\'s potentials with device="cuda"'
                )
            if len({p.elements for p in parts}) != 1:
                raise InvalidParameter(
                    "on the GPU the potentials of an Overlay need the same elements "
                    "in the same order"
                )
            # the list of the device reaches the largest cutoff of those that
            # take it; D3 finds its pairs itself
            self.__gpu = _gpu.OnDevice(
                _gpu.cuda().make_overlay([o.potential for o in on]),
                max(o.cutoff for o in on),
            )

    @property
    def _on_device(self) -> _gpu.OnDevice | None:
        """The sum on the GPU, for its molecular dynamics there."""
        return self.__gpu

    def _species(self, system: System) -> np.ndarray:
        """On the GPU or split among processes, the species of the atoms,
        the same for every calculator."""
        return self.calculators[0]._species(system)

    def __split(
        self, atoms: tuple, properties: tuple, *rest: Any, **options: Any
    ) -> _Part:
        """The parts of a process of MPI of every calculator, added."""
        parts = [
            p._split(atoms, properties, *rest, **options) for p in self.calculators
        ]

        def total(name: str) -> Any:
            values = [getattr(part, name) for part in parts]
            if any(v is None for v in values):
                return None
            return sum(values[1:], values[0])

        return _Part(
            float(total("energy")),
            total("forces"),
            total("virial"),
            total("atomic_energies"),
            total("atomic_virials"),
        )

    def __neighbors(
        self, part: Calculator, neighbors: NeighborList | None
    ) -> NeighborList | None:
        """The list a calculator takes of the list of the Overlay: as it is
        for one that passes over the pairs beyond its cutoff, those within
        its cutoff for the others, none for one with a list of its own."""
        if neighbors is None or part.device == "cuda":
            return None
        if part._takes_wider_lists:
            return neighbors
        if not 0.0 < part.cutoff < np.inf:
            return None
        return neighbors.within(min(float(part.cutoff), neighbors.cutoff))

    def compute(
        self,
        system: System,
        properties: Iterable[str] = ("energy", "forces"),
        *,
        neighbors: NeighborList | None = None,
    ) -> CalculatorResult:
        properties = tuple(properties)
        wanted = self._check_properties(properties)
        if self.__gpu is not None:
            heat = HEAT_FLUX_TERMS in properties
            return self.__gpu.compute(system, self._species(system), wanted, heat)
        if 0.0 < self.cutoff < np.inf:
            neighbors = _given_or_built(system, self.cutoff, neighbors)
        else:
            neighbors = None
        results = [
            part.compute(
                system, properties, neighbors=self.__neighbors(part, neighbors)
            )
            for part in self.calculators
        ]

        def total(name: str) -> Any:
            values = [getattr(r, name) for r in results]
            if any(v is None for v in values):
                return None
            out = values[0]
            for v in values[1:]:
                out = out + v
            return out

        return CalculatorResult(
            energy=None if total("energy") is None else float(total("energy")),
            atomic_energies=total("atomic_energies"),
            forces=total("forces"),
            virial=total("virial"),
            atomic_virials=total("atomic_virials"),
            volume=system.box.volume,
        )

    def __repr__(self) -> str:
        return f"Overlay([{', '.join(repr(p) for p in self.calculators)}])"
