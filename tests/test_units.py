# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

import pytest

from mdapy import units


def test_defining_constants_are_exact_si_values():
    assert units.ELEMENTARY_CHARGE == 1.602176634e-19
    assert units.BOLTZMANN_SI == 1.380649e-23
    assert units.AVOGADRO == 6.02214076e23


@pytest.mark.parametrize(
    "value, reference, rel",
    [
        # CODATA 2018/2022: k_B = 8.617333262...e-5 eV/K
        (units.BOLTZMANN, 8.617333262e-5, 1e-9),
        # 1 eV/Å^3 = 1.602176634e11 Pa
        (units.EV_PER_A3_TO_GPA, 160.2176634, 1e-14),
        # 1 eV/Å^2 = 16.02176634 J/m^2
        (units.EV_PER_A2_TO_J_PER_M2, 16.02176634, 1e-14),
        # LAMMPS metal units: mvv2e = 1.0364269e-4 eV per amu Å^2/ps^2
        (units.AMU_A2_PER_PS2_TO_EV, 1.0364269e-4, 1e-7),
        # 1 kJ/mol = 0.0103642697 eV
        (units.KJ_PER_MOL_TO_EV, 1.03642697e-2, 1e-8),
        # 1 kcal/mol = 0.0433641043 eV
        (units.KCAL_PER_MOL_TO_EV, 4.33641043e-2, 1e-8),
    ],
)
def test_derived_factors_match_reference_values(value, reference, rel):
    assert value == pytest.approx(reference, rel=rel)


@pytest.mark.parametrize(
    "forward, backward",
    [
        (units.EV_PER_A3_TO_GPA, units.GPA_TO_EV_PER_A3),
        (units.BAR_TO_GPA, units.GPA_TO_BAR),
        (units.FS_TO_PS, units.PS_TO_FS),
    ],
)
def test_inverse_pairs(forward, backward):
    assert forward * backward == pytest.approx(1.0, rel=1e-15)


def test_two_dimensional_stress_factor():
    # 1 GPa·Å = 1e9 Pa × 1e-10 m = 0.1 N/m
    assert units.GPA_A_TO_N_PER_M == pytest.approx(1e9 * 1e-10, rel=1e-15)


def test_all_names_are_floats():
    for name in units.__all__:
        assert isinstance(getattr(units, name), float), name
