# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Simulations: energy minimization and the energies of defects with a
calculator, and molecular dynamics in LAMMPS."""

from mdapy.simulation._defect_energies import (
    StackingFaultResult,
    SurfaceResult,
    VacancyResult,
    stacking_fault_energy,
    surface_energy,
    vacancy_formation_energy,
)
from mdapy.simulation._dynamics import MolecularDynamics
from mdapy.simulation._fault_free_energy import (
    FaultFreeEnergyResult,
    stacking_fault_free_energy,
)
from mdapy.simulation._free_energy import (
    FreeEnergyResult,
    PhaseCoexistence,
    free_energy,
)
from mdapy.simulation._melting import (
    MeltingAttempt,
    MeltingPointResult,
    melting_point,
)
from mdapy.simulation._minimize import MinimizeResult, minimize
from mdapy.simulation._phase_diagram import (
    BinaryPhaseDiagram,
    PhaseBoundary,
    PressureTemperatureDiagram,
    TieLine,
    binary_phase_diagram,
    pressure_temperature_diagram,
)

__all__ = [
    "BinaryPhaseDiagram",
    "FaultFreeEnergyResult",
    "FreeEnergyResult",
    "MeltingAttempt",
    "MeltingPointResult",
    "MinimizeResult",
    "MolecularDynamics",
    "PhaseBoundary",
    "PhaseCoexistence",
    "PressureTemperatureDiagram",
    "StackingFaultResult",
    "SurfaceResult",
    "TieLine",
    "VacancyResult",
    "binary_phase_diagram",
    "free_energy",
    "melting_point",
    "minimize",
    "pressure_temperature_diagram",
    "stacking_fault_energy",
    "stacking_fault_free_energy",
    "surface_energy",
    "vacancy_formation_energy",
]
