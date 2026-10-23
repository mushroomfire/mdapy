# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""mdapy calculators as ASE calculators, and ASE calculators as mdapy
calculators."""

from collections.abc import Iterable
from typing import Any

import numpy as np

from mdapy.calculators._base import Calculator, CalculatorResult
from mdapy.core.system import System
from mdapy.interop._ase import from_ase, to_ase

# ASE properties, and the mdapy properties they need.
_NEEDS = {
    "energy": "energy",
    "free_energy": "energy",
    "energies": "atomic_energies",
    "forces": "forces",
    "stress": "virial",
    "stresses": "atomic_virials",
}


def _voigt(tensors: np.ndarray) -> np.ndarray:
    """(..., 3, 3) tensors as (..., 6) in the order xx, yy, zz, yz, xz, xy."""
    return np.stack(
        [
            tensors[..., 0, 0],
            tensors[..., 1, 1],
            tensors[..., 2, 2],
            tensors[..., 1, 2],
            tensors[..., 0, 2],
            tensors[..., 0, 1],
        ],
        axis=-1,
    )


def to_ase_calculator(calculator: Calculator) -> Any:
    """An mdapy calculator as an ASE calculator.

    Parameters
    ----------
    calculator : Calculator
        Any mdapy calculator, such as ``EAM`` or ``NEP``.

    Returns
    -------
    ase.calculators.calculator.Calculator
        A calculator for ``ase.Atoms`` with the properties ``energy``,
        ``free_energy``, ``energies``, ``forces``, ``stress`` and
        ``stresses`` that the mdapy calculator can compute; stresses in
        eV/Å³ in ASE's order xx, yy, zz, yz, xz, xy, positive under tension.
    """
    try:
        from ase.calculators.calculator import Calculator as AseCalculator
        from ase.calculators.calculator import all_changes
    except ImportError:  # pragma: no cover - ASE is installed for the tests
        raise ImportError("an ASE calculator needs ASE") from None

    class MdapyCalculator(AseCalculator):
        implemented_properties = [
            name for name, needed in _NEEDS.items() if needed in calculator.implemented
        ]

        def __init__(self) -> None:
            super().__init__()
            self.mdapy = calculator

        def calculate(
            self, atoms=None, properties=("energy",), system_changes=all_changes
        ):
            super().calculate(atoms, properties, system_changes)
            system = from_ase(self.atoms)
            wanted = {_NEEDS[name] for name in self.implemented_properties}
            result = calculator.compute(system, sorted(wanted))
            volume = abs(np.linalg.det(self.atoms.cell.array))
            if result.energy is not None:
                self.results["energy"] = result.energy
                self.results["free_energy"] = result.energy
            if result.atomic_energies is not None:
                self.results["energies"] = np.array(result.atomic_energies)
            if result.forces is not None:
                self.results["forces"] = np.array(result.forces)
            if result.virial is not None:
                self.results["stress"] = _voigt(-np.asarray(result.virial) / volume)
            if result.atomic_virials is not None:
                self.results["stresses"] = _voigt(
                    -np.asarray(result.atomic_virials) / volume
                )

        def __repr__(self) -> str:
            return f"to_ase_calculator({calculator!r})"

    return MdapyCalculator()


# mdapy properties, and the ASE properties that give them.
_GIVES = {
    "energy": "energy",
    "atomic_energies": "energies",
    "forces": "forces",
    "virial": "stress",
    "atomic_virials": "stresses",
}


class _FromASE(Calculator):
    """An ASE calculator as an mdapy calculator."""

    cutoff = 0.0

    def __init__(self, calculator: Any) -> None:
        self.calculator = calculator
        offered = set(calculator.implemented_properties)
        self.implemented = frozenset(
            name for name, needed in _GIVES.items() if needed in offered
        )
        # the free energy is the one the forces derive from
        self._free = "free_energy" in offered

    def compute(
        self,
        system: System,
        properties: Iterable[str] = ("energy", "forces"),
        *,
        neighbors: Any = None,
    ) -> CalculatorResult:
        """Computes properties of a system with the ASE calculator.

        Parameters
        ----------
        system : System
            The atoms, with the columns the ASE calculator needs (usually
            ``element``).
        properties : iterable of str, optional
            The properties to compute.
        neighbors : NeighborList, optional
            Not used.

        Returns
        -------
        CalculatorResult
            The requested properties.

        Raises
        ------
        InvalidParameter
            If a property is not one the ASE calculator computes.
        """
        wanted = self._check_properties(properties)
        atoms = to_ase(system)
        atoms.calc = self.calculator
        values: dict[str, Any] = {}
        if "energy" in wanted:
            values["energy"] = float(
                atoms.get_potential_energy(force_consistent=self._free)
            )
        if "atomic_energies" in wanted:
            values["atomic_energies"] = np.array(
                atoms.get_potential_energies(), dtype=np.float64
            )
        if "forces" in wanted:
            values["forces"] = np.array(atoms.get_forces(), dtype=np.float64)
        volume = system.box.volume
        if "virial" in wanted:
            values["virial"] = -atoms.get_stress(voigt=False) * volume
        if "atomic_virials" in wanted:
            values["atomic_virials"] = -atoms.get_stresses(voigt=False) * volume
        return CalculatorResult(**values)

    def __repr__(self) -> str:
        return f"from_ase_calculator({self.calculator!r})"


def from_ase_calculator(calculator: Any) -> Calculator:
    """An ASE calculator as an mdapy calculator.

    Parameters
    ----------
    calculator : ase.calculators.calculator.Calculator
        Any ASE calculator, such as EMT or the calculator of a
        machine-learned potential.

    Returns
    -------
    Calculator
        An mdapy calculator with the properties that the ASE calculator
        computes: ``energy`` (the free energy where the ASE calculator gives
        one), ``atomic_energies``, ``forces``, ``virial`` from the stress and
        ``atomic_virials`` from the atomic stresses. Its ``cutoff`` is 0: it
        uses no neighbor list of mdapy.
    """
    return _FromASE(calculator)
