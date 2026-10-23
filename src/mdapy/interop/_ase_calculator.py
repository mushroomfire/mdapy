# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""An mdapy calculator as an ASE calculator."""

from typing import Any

import numpy as np

from mdapy.calculators._base import Calculator
from mdapy.interop._ase import from_ase

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
