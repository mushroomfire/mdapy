# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import polars as pl
import pytest

from mdapy import elements
from mdapy.errors import InvalidParameter


def test_symbol_table_layout():
    s = elements.SYMBOLS
    assert len(s) == 119  # "X" plus elements 1-118
    assert s[0] == "X"
    assert len(set(s)) == len(s)
    for z, symbol in [(1, "H"), (6, "C"), (26, "Fe"), (79, "Au"), (118, "Og")]:
        assert s[z] == symbol


# IUPAC 2016 standard atomic weights (abridged values for H, C, O, Cu).
@pytest.mark.parametrize(
    "symbol, mass",
    [
        ("H", 1.008),
        ("C", 12.011),
        ("O", 15.999),
        ("Al", 26.9815385),
        ("Fe", 55.845),
        ("Cu", 63.546),
        ("W", 183.84),
        ("Au", 196.966569),
        ("U", 238.02891),
    ],
)
def test_atomic_weights(symbol, mass):
    assert elements.atomic_masses([symbol])[0] == mass


def test_atomic_masses_table_is_read_only_and_positive():
    m = elements.ATOMIC_MASSES
    assert m.shape == (119,) and m[0] == 0.0
    assert np.all(m[1:] > 0)
    with pytest.raises(ValueError):
        m[1] = 2.0


def test_atomic_numbers_round_trip():
    z = elements.atomic_numbers(elements.SYMBOLS)
    np.testing.assert_array_equal(z, np.arange(119))


def test_masses_by_symbol_and_by_number_agree():
    z = np.arange(1, 119)
    symbols = [elements.SYMBOLS[i] for i in z]
    np.testing.assert_array_equal(
        elements.atomic_masses(symbols), elements.atomic_masses(z)
    )
    assert elements.atomic_masses([]).shape == (0,)


@pytest.mark.parametrize("bad", [["Cu", "CU"], ["Xx"], [1]])
def test_atomic_numbers_rejects_non_symbols(bad):
    with pytest.raises(InvalidParameter, match="unknown chemical symbol"):
        elements.atomic_numbers(bad)


def test_atomic_numbers_of_one_text():
    with pytest.raises(InvalidParameter, match=r"not the text 'Cu'; write \['Cu'\]"):
        elements.atomic_numbers("Cu")


@pytest.mark.parametrize(
    "bad, named", [(["X"], "'X'"), ([0], "'X'"), ([119], "119"), ([-1], "-1")]
)
def test_atomic_masses_rejects_unknown_elements(bad, named):
    """The message names the symbol where there is one, as X for 0."""
    with pytest.raises(
        InvalidParameter, match=f"no atomic weight for element {named}$"
    ):
        elements.atomic_masses(bad)


STABLE = [elements.SYMBOLS[z] for z in range(1, 84) if z not in (43, 61)]


def test_element_from_mass_recovers_every_stable_element():
    assert len(STABLE) == 81 and "Bi" in STABLE and "Tc" not in STABLE
    # and the radioactive elements with a standard atomic weight
    for symbol in STABLE + ["Th", "Pa", "U"]:
        m = elements.atomic_masses([symbol])[0]
        assert elements.element_from_mass(m) == symbol
        assert elements.element_from_mass(m + 0.009) == symbol
        assert elements.element_from_mass(m - 0.009) == symbol


def test_no_two_stable_elements_are_within_twice_the_tolerance():
    m = elements.atomic_masses(STABLE + ["Th", "Pa", "U"])
    gaps = np.abs(m[:, None] - m[None, :])[~np.eye(len(m), dtype=bool)]
    assert gaps.min() > 0.02


@pytest.mark.parametrize("mass", [0.5, 1000.0, 63.6])
def test_element_from_mass_without_match(mass):
    with pytest.raises(InvalidParameter, match="matches 0 elements"):
        elements.element_from_mass(mass)


def test_radioactive_elements_need_explicit_candidates():
    po = elements.atomic_masses(["Po"])[0]
    with pytest.raises(InvalidParameter, match="matches 0 elements"):
        elements.element_from_mass(po, tolerance=0.001)
    assert elements.element_from_mass(po, candidates=["Po", "Fe"]) == "Po"
    # 209Po and Bi differ by 0.002 amu.
    with pytest.raises(InvalidParameter, match=r"2 elements .* \['Bi', 'Po'\]"):
        elements.element_from_mass(po, candidates=["Bi", "Po"])


def test_element_dtype_physical_value_is_atomic_number():
    symbols = list(elements.SYMBOLS)
    s = pl.Series("element", symbols, dtype=elements.ELEMENT_DTYPE)
    np.testing.assert_array_equal(s.to_physical().to_numpy(), np.arange(119))
    with pytest.raises(pl.exceptions.InvalidOperationError):
        pl.Series("element", ["Zz"], dtype=elements.ELEMENT_DTYPE)


def test_xray_form_factors_count_the_electrons_at_q_zero():
    z = np.arange(1, 99)
    f = elements.xray_form_factors(z, 0.0)
    assert f.shape == (98,)
    np.testing.assert_allclose(f, z, atol=0.04)  # neutral atoms
    q = np.linspace(0.0, 4 * np.pi * 6, 50)
    curves = elements.xray_form_factors(["H", "Cu", "W"], q)
    assert curves.shape == (3, 50)
    assert (np.diff(curves, axis=1) < 1e-9).all()  # falling with q
    np.testing.assert_array_equal(curves[1], elements.xray_form_factors([29], q)[0])


def test_xray_form_factors_as_ase_transcribes_them():
    """ASE carries the coefficients of twelve elements, typed independently.
    Two have typing errors that put f(0) away from the number of electrons:
    oxygen a2 = 2.5088111 and a3 = 0.637053 for 2.508818 and 0.637853
    (f(0) = 7.9989), copper c = -3.774477 for -3.254477 (f(0) = 28.47)."""
    waasmaier = pytest.importorskip("ase.utils.xrdebye").waasmaier
    s = np.linspace(0.0, 6.0, 61)
    fixed = {"O": {2: 2.508818, 4: 0.637853}, "Cu": {10: -3.254477}}
    for symbol, abc in waasmaier.items():
        abc = list(abc)
        for k, value in fixed.get(symbol, {}).items():
            assert abs(abc[k] - value) > 1e-6  # still wrong there
            abc[k] = value
        theirs = abc[10] + sum(
            abc[2 * k] * np.exp(-abc[2 * k + 1] * s**2) for k in range(5)
        )
        ours = elements.xray_form_factors([symbol], 4 * np.pi * s)[0]
        np.testing.assert_allclose(ours, theirs, rtol=0, atol=1e-5, err_msg=symbol)


@pytest.mark.parametrize("bad", [["Es"], [0], ["X"], [99]])
def test_xray_form_factors_of_unknown_elements(bad):
    with pytest.raises(InvalidParameter, match="no X-ray form factor"):
        elements.xray_form_factors(bad, 1.0)


@pytest.mark.parametrize("q", [-0.1, 80.0, np.nan, [1.0, np.inf]])
def test_xray_form_factors_out_of_range(q):
    with pytest.raises(InvalidParameter, match="wave numbers must be from 0 to 75.40"):
        elements.xray_form_factors(["Cu"], q)


def test_neutron_scattering_lengths():
    lengths = elements.neutron_scattering_lengths(
        ["H", "H", "O", "Ni", "V", "Au"], [None, 2, None, 62, None, None]
    )
    # deuterium and hydrogen of opposite sign; vanadium near zero; gold of one
    # isotope, 197
    np.testing.assert_allclose(lengths, [-3.7409, 6.6681, 5.8037, -8.7, -0.443, 7.9])
    np.testing.assert_array_equal(
        elements.neutron_scattering_lengths([1, 8]),
        elements.neutron_scattering_lengths(["H", "O"]),
    )
    assert elements.neutron_scattering_lengths([]).shape == (0,)


def test_neutron_scattering_lengths_unknown():
    with pytest.raises(
        InvalidParameter, match="no neutron scattering length for Tc: give"
    ):
        elements.neutron_scattering_lengths(["Tc"])
    with pytest.raises(InvalidParameter, match="no neutron scattering length for H-5"):
        elements.neutron_scattering_lengths(["H"], [5])
    with pytest.raises(InvalidParameter, match="one mass number"):
        elements.neutron_scattering_lengths(["H", "O"], [2])
    with pytest.raises(
        InvalidParameter, match="no neutron scattering length for element 'X'"
    ):
        elements.neutron_scattering_lengths(["X"])
