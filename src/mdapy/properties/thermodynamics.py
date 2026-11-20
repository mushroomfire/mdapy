# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Thermodynamics: free energies, in the quasi-harmonic approximation or by
thermodynamic integration, melting points and phase diagrams.

>>> from mdapy.properties.thermodynamics import quasi_harmonic
"""

from mdapy.properties._free_energy import (
    FreeEnergyResult,
    PhaseCoexistence,
    free_energy,
)
from mdapy.properties._melting import (
    MeltingAttempt,
    MeltingPointResult,
    melting_point,
)
from mdapy.properties._phase_diagram import (
    BinaryPhaseDiagram,
    PhaseBoundary,
    PressureTemperatureDiagram,
    TieLine,
    binary_phase_diagram,
    pressure_temperature_diagram,
)
from mdapy.properties._quasi_harmonic import (
    QuasiHarmonicResult,
    SheetQuasiHarmonicResult,
    quasi_harmonic,
)

__all__ = [
    "BinaryPhaseDiagram",
    "FreeEnergyResult",
    "MeltingAttempt",
    "MeltingPointResult",
    "PhaseBoundary",
    "PhaseCoexistence",
    "PressureTemperatureDiagram",
    "QuasiHarmonicResult",
    "SheetQuasiHarmonicResult",
    "TieLine",
    "binary_phase_diagram",
    "free_energy",
    "melting_point",
    "pressure_temperature_diagram",
    "quasi_harmonic",
]
