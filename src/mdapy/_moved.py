# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The names that moved in 2.2 from ``mdapy.lattice_dynamics`` and
``mdapy.simulation`` to the submodules of ``mdapy.properties``: still found at
their old places in 2.2 and 2.3, with a ``DeprecationWarning`` that names the
new one, and gone in 2.4, with this module, ``mdapy.lattice_dynamics`` and the
``__getattr__`` of ``mdapy.simulation``."""

import importlib
import warnings
from typing import Any

MOVED_IN = "2.2"
GOES_IN = "2.4"  # two minor releases later

MOVED = {
    "mdapy.lattice_dynamics": {
        "BandStructure": "mdapy.properties.phonons",
        "BondStiffness": "mdapy.properties.phonons",
        "ElasticResult": "mdapy.properties.elastic",
        "ForceConstants": "mdapy.properties.phonons",
        "GreenKubo": "mdapy.properties.transport",
        "PhononDOS": "mdapy.properties.phonons",
        "Phonons": "mdapy.properties.phonons",
        "QuasiHarmonicResult": "mdapy.properties.thermodynamics",
        "SheetElasticResult": "mdapy.properties.elastic",
        "SheetQuasiHarmonicResult": "mdapy.properties.thermodynamics",
        "ThermalConductivity": "mdapy.properties.transport",
        "ThermalElasticResult": "mdapy.properties.elastic",
        "ThermalProperties": "mdapy.properties.phonons",
        "bond_stiffness": "mdapy.properties.phonons",
        "displaced_supercells": "mdapy.properties.phonons",
        "elastic_constants": "mdapy.properties.elastic",
        "force_constants": "mdapy.properties.phonons",
        "green_kubo": "mdapy.properties.transport",
        "green_kubo_from_flux": "mdapy.properties.transport",
        "md_elastic_constants": "mdapy.properties.elastic",
        "phonons": "mdapy.properties.phonons",
        "phonons_from_frames": "mdapy.properties.phonons",
        "primitive_cell": "mdapy.properties.phonons",
        "quasi_harmonic": "mdapy.properties.thermodynamics",
        "read_band_structure": "mdapy.properties.phonons",
        "read_force_constants": "mdapy.properties.phonons",
        "thermal_conductivity": "mdapy.properties.transport",
        "thermal_elastic_constants": "mdapy.properties.elastic",
        "thermal_phonons": "mdapy.properties.phonons",
    },
    "mdapy.simulation": {
        "BinaryPhaseDiagram": "mdapy.properties.thermodynamics",
        "FaultFreeEnergyResult": "mdapy.properties.defects",
        "FreeEnergyResult": "mdapy.properties.thermodynamics",
        "MeltingAttempt": "mdapy.properties.thermodynamics",
        "MeltingPointResult": "mdapy.properties.thermodynamics",
        "PhaseBoundary": "mdapy.properties.thermodynamics",
        "PhaseCoexistence": "mdapy.properties.thermodynamics",
        "PressureTemperatureDiagram": "mdapy.properties.thermodynamics",
        "StackingFaultResult": "mdapy.properties.defects",
        "SurfaceResult": "mdapy.properties.defects",
        "TieLine": "mdapy.properties.thermodynamics",
        "VacancyResult": "mdapy.properties.defects",
        "binary_phase_diagram": "mdapy.properties.thermodynamics",
        "free_energy": "mdapy.properties.thermodynamics",
        "melting_point": "mdapy.properties.thermodynamics",
        "pressure_temperature_diagram": "mdapy.properties.thermodynamics",
        "stacking_fault_energy": "mdapy.properties.defects",
        "stacking_fault_free_energy": "mdapy.properties.defects",
        "surface_energy": "mdapy.properties.defects",
        "vacancy_formation_energy": "mdapy.properties.defects",
    },
}


def forward(module: str, name: str) -> Any:
    """The object a moved name of module names, at its new place."""
    new = MOVED.get(module, {}).get(name)
    if new is None:
        raise AttributeError(f"module {module!r} has no attribute {name!r}")
    warnings.warn(
        f"{module}.{name} moved to {new}.{name} in MDAPY {MOVED_IN}, and its old "
        f"import goes in {GOES_IN}",
        DeprecationWarning,
        stacklevel=3,
    )
    return getattr(importlib.import_module(new), name)
