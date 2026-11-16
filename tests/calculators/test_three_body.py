# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Tersoff, Stillinger–Weber and ADP: against LAMMPS pair_style tersoff, sw
and adp atom by atom, against finite differences, and their files."""

import numpy as np
import pytest

import mdapy
from mdapy.build import build_crystal
from mdapy.calculators import ADP, StillingerWeber, Tersoff
from mdapy.calculators.testing import check_calculator
from mdapy.errors import FileFormatError, InvalidParameter

from .eam_files import alloy
from .potential_files import write_adp, write_sw, write_tersoff


def diamond(elements, a=5.2, repeat=3, jitter=0.15, seed=0, pbc=(True, True, True)):
    """A shaken diamond crystal of random elements."""
    rng = np.random.default_rng(seed)
    system = build_crystal("diamond", a, elements="Si", repeat=(repeat, repeat, repeat))
    if not all(pbc):
        system = system.with_box(mdapy.Box(system.box.matrix, pbc=pbc))
    return system.with_columns(
        positions=system["positions"] + rng.normal(0, jitter, (len(system), 3)),
        element=list(rng.choice(elements, len(system))),
    )


def test_check_calculator(tmp_path):
    tersoff = Tersoff(write_tersoff(tmp_path / "a.tersoff", ("Si", "C", "Ge")))
    check_calculator(tersoff, diamond(("Si", "C", "Ge"), repeat=2, jitter=0.1))
    sw = StillingerWeber(write_sw(tmp_path / "a.sw", ("Si", "Ge", "C")))
    check_calculator(sw, diamond(("Si", "Ge", "C"), a=5.4, repeat=2, jitter=0.1))
    adp = ADP(write_adp(tmp_path / "a.adp"))
    # the angular terms make the third derivatives large: a shorter step
    check_calculator(adp, alloy(seed=1, repeat=2, jitter=0.1), displacement=1e-5)


def test_the_bond_order_of_a_lone_pair_is_one(tmp_path):
    # two atoms: no third atom, zeta = 0, b = 1: the energy is f_R + f_A
    tersoff = Tersoff(write_tersoff(tmp_path / "a.tersoff", ("Si",)))
    r = 2.35
    system = mdapy.System(
        [[0, 0, 0], [r, 0, 0]], mdapy.Box(30.0, pbc=False), elements="Si"
    )
    a, b, l1, l2 = 1830.8, 471.18, 2.4799, 1.7322
    assert tersoff.compute(system).energy == pytest.approx(
        a * np.exp(-l1 * r) - b * np.exp(-l2 * r), rel=1e-14
    )
    # Stillinger-Weber of two atoms: its two-body term alone
    sw = StillingerWeber(write_sw(tmp_path / "a.sw", ("Si",)))
    eps, sigma, a_, _, _, _, big_a, big_b, p, q, _ = (
        2.1683,
        2.0951,
        1.8,
        0,
        0,
        0,
        7.049556277,
        0.6022245584,
        4,
        0,
        0,
    )
    x = r / sigma
    expected = big_a * eps * (big_b * x**-p - x**-q) * np.exp(1 / (x - a_))
    assert sw.compute(system).energy == pytest.approx(expected, rel=1e-13)


def test_elements_of_the_file_and_labels(tmp_path):
    path = write_tersoff(tmp_path / "a.tersoff", ("Si", "C"))
    assert Tersoff(path).elements == ("Si", "C")
    assert Tersoff(path, ["C"]).elements == ("C",)
    assert Tersoff(path, "Si").elements == ("Si",)
    # labels that are not elements, mapped
    text = path.read_text().replace("Si", "Si(D)")
    labelled = tmp_path / "b.tersoff"
    labelled.write_text(text)
    with pytest.raises(InvalidParameter, match="map the elements of the atoms"):
        Tersoff(labelled)
    mapped = Tersoff(labelled, {"Si": "Si(D)", "C": "C"})
    system = diamond(("Si", "C"), repeat=2)
    assert mapped.compute(system).energy == Tersoff(path).compute(system).energy
    assert repr(mapped).startswith("Tersoff(") and "('Si', 'C')" in repr(mapped)


@pytest.mark.parametrize(
    "text, message",
    [
        ("Si Si Si 3 1 0 1 1 0 1 1 1 1 2.8 0.2 1\n", "not entries of 17"),
        ("Si Si Si 3 1 0 1 1 0 1 1 1 1 2.8 0.2 1 x\n", "word for a number"),
        ("Si Si Si 3 1 0 1 1 0 1 1 1 1 2.8 0.2 1 1\n" * 2, "two entries"),
        ("Si Si Si 2 1 0 1 1 0 1 1 1 1 2.8 0.2 1 1\n", "must be 1 or 3"),
        ("Si Si Si 3 1 0 1 1 0 1 1 1 1 2.8 3.0 1 1\n", "negative parameter or D > R"),
        ("Si Si Si 3 1 0 1 1 0 1 1 1 1 2.8 0.2 1 inf\n", "is not 14 numbers"),
        ("Si Si C 3 1 0 1 1 0 1 1 1 1 2.8 0.2 1 1\n", "no entry for Si Si Si"),
        ("# UNITS: real\nSi Si Si 3 1 0 1 1 0 1 1 1 1 2.8 0.2 1 1\n", "in real units"),
    ],
)
def test_tersoff_file_errors(tmp_path, text, message):
    path = tmp_path / "bad.tersoff"
    path.write_text(text)
    with pytest.raises(InvalidParameter, match=message):
        Tersoff(path)


def test_choosing_elements_errors(tmp_path):
    path = write_tersoff(tmp_path / "a.tersoff", ("Si", "C"))
    with pytest.raises(InvalidParameter, match="no entries of 'Ge'"):
        Tersoff(path, ["Si", "Ge"])
    with pytest.raises(InvalidParameter, match="each element once"):
        Tersoff(path, ["Si", "Si"])
    with pytest.raises(InvalidParameter, match="'Xx' is not an element"):
        Tersoff(path, {"Xx": "Si"})
    with pytest.raises(InvalidParameter, match="cannot read"):
        Tersoff(tmp_path / "missing.tersoff")
    sw = tmp_path / "bad.sw"
    sw.write_text("Si Si Si 2.1683 2.0951 1.8 21 1.2 -0.333 7.05 0.60 4 0 -1\n")
    with pytest.raises(InvalidParameter, match="negative parameter"):
        StillingerWeber(sw)


def test_adp_file_errors(tmp_path):
    good = write_adp(tmp_path / "a.adp").read_text().splitlines()
    for line, replacement, message in (
        (3, "x Cu Ni", "number of elements"),
        (3, "2 Cu", "2 element names"),
        (4, "1000 0.05", "Nrho drho Nr dr cutoff"),
        (4, "3 0.05 1000 0.0055 5.0", "at least 5"),
        (5, "29", "atomic number and the mass"),
    ):
        lines = list(good)
        lines[line] = replacement
        bad = tmp_path / "bad.adp"
        bad.write_text("\n".join(lines))
        with pytest.raises(FileFormatError, match=message):
            ADP(bad)
    lines = list(good)
    lines[3] = "2 Cu Xx"
    bad = tmp_path / "bad.adp"
    bad.write_text("\n".join(lines))
    with pytest.raises(InvalidParameter, match="'Xx' is not an element"):
        ADP(bad)
    with pytest.raises(InvalidParameter, match="cannot read"):
        ADP(tmp_path / "missing.adp")
    adp = ADP(tmp_path / "a.adp")
    assert adp.elements == ("Cu", "Ni") and adp.masses[0] == pytest.approx(63.546)
    assert repr(adp).startswith("ADP(")


class TestAgainstLammps:
    @pytest.fixture(autouse=True)
    def _lammps(self):
        pytest.importorskip("lammps")

    @pytest.mark.parametrize("elements", [("Si",), ("Si", "C"), ("Si", "C", "Ge")])
    @pytest.mark.parametrize("pbc", [(True, True, True), (True, True, False)])
    def test_tersoff(self, tmp_path, elements, pbc):
        from .lammps_reference import assert_matches, reference

        path = write_tersoff(tmp_path / "a.tersoff", elements)
        system = diamond(elements, a=5.0, seed=len(elements), pbc=pbc)
        pair = f"pair_style tersoff\npair_coeff * * {path} {' '.join(elements)}"
        expected = reference(system, list(elements), pair, tmp_path)
        result = Tersoff(path).compute(system, ("virial", "atomic_virials"))
        assert_matches(result, expected)

    def test_tersoff_in_a_small_box(self, tmp_path):
        from .lammps_reference import assert_matches, reference

        # a cell of 8 atoms, 5 Å across, with a cutoff of 3 Å: images in the bonds
        path = write_tersoff(tmp_path / "a.tersoff", ("Si", "C"))
        system = diamond(("Si", "C"), a=5.0, repeat=1, jitter=0.2, seed=9)
        pair = f"pair_style tersoff\npair_coeff * * {path} Si C"
        expected = reference(system, ["Si", "C"], pair, tmp_path)
        assert_matches(Tersoff(path).compute(system, ("atomic_virials",)), expected)

    @pytest.mark.parametrize("elements", [("Si",), ("Si", "Ge"), ("Si", "Ge", "C")])
    @pytest.mark.parametrize("tol", [0.0, 1e-3])
    def test_stillinger_weber(self, tmp_path, elements, tol):
        from .lammps_reference import assert_matches, reference

        path = write_sw(tmp_path / "a.sw", elements, tol=tol)
        system = diamond(elements, a=5.3, seed=len(elements))
        pair = f"pair_style sw\npair_coeff * * {path} {' '.join(elements)}"
        expected = reference(system, list(elements), pair, tmp_path)
        assert_matches(
            StillingerWeber(path).compute(system, ("atomic_virials",)), expected
        )

    @pytest.mark.parametrize("pbc", [(True, True, True), (False, True, True)])
    def test_adp(self, tmp_path, pbc):
        from .lammps_reference import assert_matches, reference

        path = write_adp(tmp_path / "a.adp")
        system = alloy(seed=7, repeat=3, jitter=0.15, pbc=pbc)
        pair = f"pair_style adp\npair_coeff * * {path} Cu Ni"
        expected = reference(system, ["Cu", "Ni"], pair, tmp_path)
        assert_matches(ADP(path).compute(system, ("atomic_virials",)), expected)


def _mean(first, second):
    return {key: 0.5 * (first[key] + second[key]) for key in first}


class TestEntriesThatDisagree:
    """Where two entries of a file meet in one term, LAMMPS takes one by the
    numbering of the atoms or the order of the neighbors; MDAPY takes their
    mean, the mean of LAMMPS with either entry in both places."""

    @pytest.fixture(autouse=True)
    def _lammps(self):
        pytest.importorskip("lammps")

    def test_the_three_body_entries_of_stillinger_weber(self, tmp_path):
        from .lammps_reference import assert_matches, reference
        from .potential_files import sw_entry

        def entry(order):
            def numbers(i, j, k):
                values = list(sw_entry(i, j, k))
                if j != k:  # lambda of i j k and i k j apart
                    first = order(j, k)
                    values[3] *= 1.1 if first else 0.9
                return values

            return numbers

        elements = ("Si", "Ge")
        own = write_sw(tmp_path / "own.sw", elements, entry(lambda j, k: j < k))
        one = write_sw(tmp_path / "one.sw", elements, entry(lambda j, k: True))
        other = write_sw(tmp_path / "other.sw", elements, entry(lambda j, k: False))
        system = diamond(elements, a=5.3, seed=5)
        expected = _mean(
            *(
                reference(
                    system,
                    list(elements),
                    f"pair_style sw\npair_coeff * * {f} Si Ge",
                    tmp_path,
                )
                for f in (one, other)
            )
        )
        assert_matches(
            StillingerWeber(own).compute(system, ("atomic_virials",)), expected
        )

    def test_the_repulsion_of_tersoff(self, tmp_path):
        from .lammps_reference import assert_matches, reference
        from .potential_files import tersoff_entry

        def entry(which):
            def numbers(i, j, k):
                values = list(tersoff_entry(i, j, k))
                if i != j and j == k:  # A and lambda1 of Si C C and C Si Si apart
                    side = which(i)
                    values[13] *= 1.02 if side else 0.98
                    values[12] *= 1.01 if side else 0.99
                return values

            return numbers

        elements = ("Si", "C")
        own = write_tersoff(
            tmp_path / "own.tersoff", elements, entry(lambda i: i == "Si")
        )
        one = write_tersoff(tmp_path / "one.tersoff", elements, entry(lambda i: True))
        other = write_tersoff(
            tmp_path / "other.tersoff", elements, entry(lambda i: False)
        )
        system = diamond(elements, a=4.6, seed=6)
        expected = _mean(
            *(
                reference(
                    system,
                    list(elements),
                    f"pair_style tersoff\npair_coeff * * {f} Si C",
                    tmp_path,
                )
                for f in (one, other)
            )
        )
        assert_matches(Tersoff(own).compute(system, ("atomic_virials",)), expected)
