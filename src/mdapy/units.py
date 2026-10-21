# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Physical constants and unit conversion factors.

mdapy works in one unit system: length in Å, energy in eV, time in ps,
mass in amu, temperature in K, charge in e, force in eV/Å, stress and
pressure in GPa. Every conversion in the package uses the factors defined
here.

A factor named ``A_TO_B`` converts a value expressed in unit A into unit B
by multiplication.
"""

__all__ = [
    "ELEMENTARY_CHARGE",
    "BOLTZMANN_SI",
    "AVOGADRO",
    "ATOMIC_MASS_UNIT_SI",
    "BOLTZMANN",
    "EV_PER_A3_TO_GPA",
    "GPA_TO_EV_PER_A3",
    "BAR_TO_GPA",
    "GPA_TO_BAR",
    "EV_PER_A2_TO_J_PER_M2",
    "GPA_A_TO_N_PER_M",
    "FS_TO_PS",
    "PS_TO_FS",
    "AMU_A2_PER_PS2_TO_EV",
    "KJ_PER_MOL_TO_EV",
    "KCAL_PER_MOL_TO_EV",
]

ELEMENTARY_CHARGE = 1.602176634e-19
"""Elementary charge in C (exact, 2019 SI)."""
BOLTZMANN_SI = 1.380649e-23
"""Boltzmann constant in J/K (exact, 2019 SI)."""
AVOGADRO = 6.02214076e23
"""Avogadro constant in 1/mol (exact, 2019 SI)."""
ATOMIC_MASS_UNIT_SI = 1.66053906892e-27
"""Atomic mass constant in kg (CODATA 2022)."""

BOLTZMANN = BOLTZMANN_SI / ELEMENTARY_CHARGE
"""Boltzmann constant in eV/K."""

EV_PER_A3_TO_GPA = ELEMENTARY_CHARGE * 1e30 / 1e9
"""Stress: eV/Å³ to GPa."""
GPA_TO_EV_PER_A3 = 1.0 / EV_PER_A3_TO_GPA
"""Stress: GPa to eV/Å³."""
BAR_TO_GPA = 1e-4
"""Pressure: bar to GPa (exact)."""
GPA_TO_BAR = 1e4
"""Pressure: GPa to bar (exact)."""

EV_PER_A2_TO_J_PER_M2 = ELEMENTARY_CHARGE * 1e20
"""Energy per area: eV/Å² to J/m² (equal to N/m)."""
GPA_A_TO_N_PER_M = 0.1
"""Two-dimensional stress: GPa·Å to N/m (exact)."""

FS_TO_PS = 1e-3
"""Time: fs to ps (exact)."""
PS_TO_FS = 1e3
"""Time: ps to fs (exact)."""

AMU_A2_PER_PS2_TO_EV = ATOMIC_MASS_UNIT_SI * 1e4 / ELEMENTARY_CHARGE
"""Kinetic energy: amu·Å²/ps² to eV (mass times velocity squared)."""

KJ_PER_MOL_TO_EV = 1e3 / (AVOGADRO * ELEMENTARY_CHARGE)
"""Energy: kJ/mol to eV."""
KCAL_PER_MOL_TO_EV = 4.184 * KJ_PER_MOL_TO_EV
"""Energy: kcal/mol (thermochemical calorie, 4.184 J) to eV."""
