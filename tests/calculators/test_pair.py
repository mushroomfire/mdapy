# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Lennard-Jones and Morse: against LAMMPS pair_style lj/cut and morse atom
by atom, against finite differences, and their parameters."""

import numpy as np
import pytest

import mdapy
from mdapy.calculators import LennardJones, Morse
from mdapy.calculators.testing import check_calculator
from mdapy.errors import InvalidParameter

from .eam_files import alloy


def argon(seed=0, repeat=3, jitter=0.2, a=5.26):
    """A shaken fcc crystal of argon and krypton."""
    system = alloy(seed=seed, repeat=repeat, jitter=jitter, a=a)
    rng = np.random.default_rng(seed + 1)
    return system.with_columns(element=list(rng.choice(["Ar", "Kr"], len(system))))


LJ = {"Ar": (0.0104, 3.40), "Kr": (0.0140, 3.65, 9.0)}


def test_one_pair_is_the_formula():
    calculator = LennardJones({"Ar": (0.0104, 3.4)}, 8.5)
    for r in (3.0, 3.8164, 6.0):
        system = mdapy.System(
            [[0, 0, 0], [r, 0, 0]], mdapy.Box(40.0, pbc=False), elements="Ar"
        )
        result = calculator.compute(system, ("energy", "forces"))
        x = (3.4 / r) ** 6
        assert result.energy == pytest.approx(4 * 0.0104 * (x * x - x), rel=1e-14)
        force = 4 * 0.0104 * (12 * x * x - 6 * x) / r
        assert result.forces[1, 0] == pytest.approx(force, rel=1e-13)
        np.testing.assert_allclose(result.forces[0], -result.forces[1], atol=1e-16)
    system = mdapy.System(
        [[0, 0, 0], [8.6, 0, 0]], mdapy.Box(40.0, pbc=False), elements="Ar"
    )
    assert calculator.compute(system).energy == 0.0
    morse = Morse({"Cu": (0.3429, 1.3588, 2.866)}, 9.0)
    for r in (2.4, 2.866, 4.0):
        system = mdapy.System(
            [[0, 0, 0], [r, 0, 0]], mdapy.Box(40.0, pbc=False), elements="Cu"
        )
        e = np.exp(-1.3588 * (r - 2.866))
        assert morse.compute(system).energy == pytest.approx(
            0.3429 * (e * e - 2 * e), rel=1e-14
        )


def test_shift_takes_the_energy_at_the_cutoff_off():
    pair = mdapy.System(
        [[0, 0, 0], [4.0, 0, 0]], mdapy.Box(40.0, pbc=False), elements="Ar"
    )
    plain = LennardJones({"Ar": (0.0104, 3.4)}, 8.5).compute(pair).energy
    shifted = LennardJones({"Ar": (0.0104, 3.4)}, 8.5, shift=True).compute(pair).energy
    x = (3.4 / 8.5) ** 6
    assert plain - shifted == pytest.approx(4 * 0.0104 * (x * x - x), rel=1e-12)


@pytest.mark.parametrize(
    "calculator",
    [
        LennardJones(LJ, 8.0, mixing="arithmetic", shift=True),
        Morse(
            {
                "Ar": (0.012, 1.3, 3.8),
                "Kr": (0.016, 1.2, 4.0),
                ("Ar", "Kr"): (0.014, 1.25, 3.9, 11.5),
            },
            12.0,
            shift=True,
        ),
    ],
)
def test_check_calculator(calculator):
    # shifted, and Morse cut off far out, so that a pair crossing the cutoff
    # under the finite differences of the strain does not jump
    check_calculator(calculator, argon(repeat=2, jitter=0.3), rtol=1e-6, atol=1e-7)


def test_mixing_rules():
    first, second = (0.01, 3.0, 8.0), (0.04, 4.0, 10.0)
    for rule, expected in (
        ("geometric", (0.02, np.sqrt(12.0), np.sqrt(80.0))),
        ("arithmetic", (0.02, 3.5, 9.0)),
        (
            "sixthpower",
            (
                2 * 0.02 * 27 * 64 / (3**6 + 4**6),
                (0.5 * (3**6 + 4**6)) ** (1 / 6),
                (0.5 * (8**6 + 10**6)) ** (1 / 6),
            ),
        ),
    ):
        lj = LennardJones({"Ar": first, "Kr": second}, 5.0, mixing=rule)
        np.testing.assert_allclose(
            (lj.epsilon[0, 1], lj.sigma[0, 1], lj.cutoffs[0, 1]), expected, rtol=1e-14
        )
        assert lj.cutoff == 10.0
    # a pair given as it is, not mixed
    lj = LennardJones({"Ar": first, "Kr": second, ("Kr", "Ar"): (0.5, 3.3)}, 5.0)
    assert (lj.epsilon[1, 0], lj.sigma[0, 1], lj.cutoffs[0, 1]) == (0.5, 3.3, 5.0)
    assert lj.elements == ("Ar", "Kr")


@pytest.mark.parametrize(
    "make, message",
    [
        (lambda: LennardJones({}, 5.0), "at least one entry"),
        (lambda: LennardJones({"Xx": (1, 1)}, 5.0), "not an element"),
        (lambda: LennardJones({"Ar": (1,)}, 5.0), "2 or 3 finite numbers"),
        (lambda: LennardJones({"Ar": (1, "a")}, 5.0), "must be numbers"),
        (lambda: LennardJones({"Ar": (1, 1)}, 0.0), "cutoff must be positive"),
        (lambda: LennardJones({"Ar": (1, 1)}, "x"), "cutoff must be a number"),
        (
            lambda: LennardJones({"Ar": (1, 1)}, 5.0, mixing="x"),
            "mixing must be one of",
        ),
        (lambda: LennardJones({"Ar": (-1, 1)}, 5.0), "epsilon must not be negative"),
        (lambda: LennardJones({("Ar", "Kr"): (1, 1)}, 5.0), "Ar–Ar has no parameters"),
        (
            lambda: LennardJones({("Ar", "Kr", "Ne"): (1, 1)}, 5.0),
            "an element or a pair",
        ),
        (
            lambda: LennardJones({("Ar", "Kr"): (1, 1), ("Kr", "Ar"): (2, 1)}, 5.0),
            "given twice",
        ),
        (
            lambda: Morse({"Cu": (1, 1, 1), "Ni": (1, 1, 1)}, 5.0),
            "Cu–Ni has no parameters",
        ),
        (lambda: Morse({"Cu": (1, 1)}, 5.0), "3 or 4 finite numbers"),
        (lambda: Morse({"Cu": (1, 0, 1)}, 5.0), "alpha and r0 must be positive"),
    ],
)
def test_errors(make, message):
    with pytest.raises(InvalidParameter, match=message):
        make()


def test_atoms_of_other_elements_are_refused():
    system = mdapy.System(
        [[0, 0, 0], [3, 0, 0]], mdapy.Box(20.0), elements=["Ar", "Ne"]
    )
    with pytest.raises(InvalidParameter, match="no parameters for \\['Ne'\\]"):
        LennardJones({"Ar": (0.01, 3.4)}, 8.0).compute(system)


class TestAgainstLammps:
    @pytest.fixture(autouse=True)
    def _lammps(self):
        pytest.importorskip("lammps")

    @pytest.mark.parametrize("mixing", ["geometric", "arithmetic", "sixthpower"])
    @pytest.mark.parametrize("shift", [False, True])
    def test_lennard_jones(self, tmp_path, mixing, shift):
        from .lammps_reference import assert_matches, reference

        system = argon(seed=3, repeat=3)
        pair = f"""
pair_style lj/cut 8.0
pair_coeff 1 1 0.0104 3.40
pair_coeff 2 2 0.0140 3.65 9.0
pair_modify mix {mixing} shift {"yes" if shift else "no"}
"""
        expected = reference(system, ["Ar", "Kr"], pair, tmp_path)
        result = LennardJones(LJ, 8.0, mixing=mixing, shift=shift).compute(
            system, ("energy", "forces", "virial", "atomic_virials")
        )
        assert_matches(result, expected)

    def test_lennard_jones_of_a_given_pair_in_a_small_box(self, tmp_path):
        from .lammps_reference import assert_matches, reference

        # a box of 10.5 Å with a cutoff of 9 Å: atoms meet their own images
        system = argon(seed=4, repeat=2, jitter=0.3)
        pair = """
pair_style lj/cut 8.0
pair_coeff 1 1 0.0104 3.40
pair_coeff 2 2 0.0140 3.65
pair_coeff 1 2 0.0300 3.30 9.0
"""
        expected = reference(system, ["Ar", "Kr"], pair, tmp_path)
        lj = LennardJones(
            {"Ar": LJ["Ar"], "Kr": (0.0140, 3.65), ("Ar", "Kr"): (0.03, 3.3, 9.0)}, 8.0
        )
        assert_matches(lj.compute(system, ("virial", "atomic_virials")), expected)

    @pytest.mark.parametrize("shift", [False, True])
    def test_morse(self, tmp_path, shift):
        from .lammps_reference import assert_matches, reference

        system = alloy(seed=5, repeat=3, jitter=0.15, a=3.6)
        pair = f"""
pair_style morse 7.0
pair_coeff 1 1 0.3429 1.3588 2.866
pair_coeff 2 2 0.4205 1.4199 2.780 7.5
pair_coeff 1 2 0.3800 1.3900 2.820
pair_modify shift {"yes" if shift else "no"}
"""
        expected = reference(system, ["Cu", "Ni"], pair, tmp_path)
        morse = Morse(
            {
                "Cu": (0.3429, 1.3588, 2.866),
                "Ni": (0.4205, 1.4199, 2.780, 7.5),
                ("Cu", "Ni"): (0.3800, 1.3900, 2.820),
            },
            7.0,
            shift=shift,
        )
        assert_matches(morse.compute(system, ("virial", "atomic_virials")), expected)

    def test_a_surface(self, tmp_path):
        from .lammps_reference import assert_matches, reference

        system = alloy(seed=6, repeat=3, jitter=0.1, a=3.6, pbc=(True, True, False))
        pair = "pair_style morse 6.0\npair_coeff * * 0.3429 1.3588 2.866"
        expected = reference(system, ["Cu", "Ni"], pair, tmp_path)
        morse = Morse(
            {
                ("Cu", "Cu"): (0.3429, 1.3588, 2.866),
                ("Ni", "Ni"): (0.3429, 1.3588, 2.866),
                ("Cu", "Ni"): (0.3429, 1.3588, 2.866),
            },
            6.0,
        )
        assert_matches(morse.compute(system, ("virial", "atomic_virials")), expected)
