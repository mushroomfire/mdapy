# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The reference models of free energies: the Uhlenbeck-Ford model against
LAMMPS pair_style ufm atom by atom and against its formula and finite
differences; the springs of an Einstein crystal against theirs."""

import numpy as np
import pytest

import mdapy
from mdapy.calculators import Springs, UhlenbeckFord
from mdapy.calculators.testing import check_calculator
from mdapy.errors import InvalidParameter

from .test_pair import argon

UF = {"Ar": (0.25, 1.5), "Kr": (0.4, 1.7, 6.5)}


def test_one_pair_is_the_formula():
    box = mdapy.Box(np.eye(3) * 30.0, pbc=False)
    for r in (0.4, 1.0, 2.5):
        pair = mdapy.System([[0.0, 0.0, 0.0], [r, 0.0, 0.0]], box, elements=["Ar"] * 2)
        result = UhlenbeckFord({"Ar": (0.25, 1.5)}, 6.0).compute(
            pair, ("energy", "forces")
        )
        x = np.exp(-(r**2) / 1.5**2)
        assert result.energy == pytest.approx(-0.25 * np.log(1 - x), rel=1e-13)
        force = 2 * 0.25 / 1.5**2 * x / (1 - x) * r  # repulsive, on the second
        assert result.forces[1, 0] == pytest.approx(force, rel=1e-12)
    # any atom, and a shift that takes the energy at the cutoff off
    far = mdapy.System([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], box)
    shifted = UhlenbeckFord((0.25, 1.5), 5.0 + 1e-9, shift=True).compute(far)
    assert abs(shifted.energy) < 1e-12
    assert "UhlenbeckFord(" in repr(UhlenbeckFord(UF, 6.0))


def test_check_calculator():
    check_calculator(
        UhlenbeckFord(UF, 8.0, shift=True),
        argon(repeat=2, jitter=0.3, a=2.6),
        rtol=1e-6,
        atol=1e-7,
    )


@pytest.mark.parametrize("mixing", ["geometric", "arithmetic", "sixthpower"])
@pytest.mark.parametrize("shift", [False, True])
def test_against_lammps(tmp_path, mixing, shift):
    pytest.importorskip("lammps")
    from .lammps_reference import assert_matches, has_style, reference

    if not has_style("ufm"):
        pytest.skip("LAMMPS has no pair_style ufm")
    system = argon(seed=3, repeat=3, a=2.6)  # close, as in a dense liquid
    pair = f"""
pair_style ufm 6.0
pair_coeff 1 1 0.25 1.5
pair_coeff 2 2 0.4 1.7 6.5
pair_modify mix {mixing} shift {"yes" if shift else "no"}
"""
    expected = reference(system, ["Ar", "Kr"], pair, tmp_path)
    result = UhlenbeckFord(UF, 6.0, mixing=mixing, shift=shift).compute(
        system, ("energy", "forces", "virial", "atomic_virials")
    )
    assert_matches(result, expected)


def test_a_pair_of_one_element():
    system = argon(repeat=2, a=2.6).with_columns(element=["Ar"] * 32)
    alone = UhlenbeckFord({"Ar": (0.25, 1.5)}, 6.0).compute(system)
    paired = UhlenbeckFord({("Ar", "Ar"): (0.25, 1.5)}, 6.0).compute(system)
    assert paired.energy == pytest.approx(alone.energy, rel=1e-14)


def test_errors():
    with pytest.raises(InvalidParameter, match="mixing"):
        UhlenbeckFord(UF, 6.0, mixing="sum")
    with pytest.raises(InvalidParameter, match="sigma must be positive"):
        UhlenbeckFord({"Ar": (0.25, 0.0)}, 6.0)


def crystal(seed=0):
    system = argon(seed=seed, repeat=2, jitter=0.0)
    rng = np.random.default_rng(seed)
    return system, system.with_columns(
        positions=system["positions"] + rng.normal(0, 0.2, (len(system), 3))
    )


def test_springs_are_their_formula():
    sites, moved = crystal()
    k = np.linspace(1.0, 2.0, len(sites))
    springs = Springs(sites["positions"], k)
    result = springs.compute(moved, ("energy", "atomic_energies", "forces"))
    d = moved["positions"] - sites["positions"]
    np.testing.assert_allclose(result.atomic_energies, 0.5 * k * (d * d).sum(axis=1))
    np.testing.assert_allclose(result.forces, -k[:, None] * d, atol=1e-14)
    assert result.energy == pytest.approx(result.atomic_energies.sum())
    # an atom across the periodic box from its site is next to it
    positions = moved["positions"].copy()
    positions[0] += moved.box.matrix[0] - moved.box.matrix[2]
    across = springs.compute(moved.with_columns(positions=positions))
    assert across.energy == pytest.approx(result.energy, rel=1e-12)
    # the forces are minus the derivatives of the energy
    h = 1e-6
    for a in range(3):
        plus, minus = positions.copy(), positions.copy()
        plus[3, a] += h
        minus[3, a] -= h
        e = [
            springs.compute(moved.with_columns(positions=p)).energy
            for p in (plus, minus)
        ]
        assert -(e[0] - e[1]) / (2 * h) == pytest.approx(across.forces[3, a], rel=1e-7)
    assert repr(springs) == f"Springs({len(sites)} sites)"


def test_springs_errors():
    sites, moved = crystal()
    with pytest.raises(InvalidParameter, match="sites"):
        Springs(np.zeros((4, 2)), 1.0)
    with pytest.raises(InvalidParameter, match="positive"):
        Springs(sites["positions"], 0.0)
    with pytest.raises(InvalidParameter, match="sites, the system"):
        Springs(sites["positions"][:3], 1.0).compute(moved)
    with pytest.raises(InvalidParameter, match="virial"):
        Springs(sites["positions"], 1.0).compute(moved, ("virial",))
