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


@pytest.mark.parametrize("bad", [["X"], [0], [119], [-1]])
def test_atomic_masses_rejects_unknown_elements(bad):
    with pytest.raises(InvalidParameter):
        elements.atomic_masses(bad)


STABLE = [elements.SYMBOLS[z] for z in range(1, 84) if z not in (43, 61)]


def test_element_from_mass_recovers_every_stable_element():
    assert len(STABLE) == 81 and "Bi" in STABLE and "Tc" not in STABLE
    for symbol in STABLE:
        m = elements.atomic_masses([symbol])[0]
        assert elements.element_from_mass(m) == symbol
        assert elements.element_from_mass(m + 0.009) == symbol
        assert elements.element_from_mass(m - 0.009) == symbol


def test_no_two_stable_elements_are_within_twice_the_tolerance():
    m = elements.atomic_masses(STABLE)
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
