# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""MEAM: against LAMMPS pair_style meam atom by atom on every option MDAPY
takes, the Rose energy of its reference structure, finite differences, and
its files."""

import numpy as np
import pytest

import mdapy
from mdapy.build import build_crystal
from mdapy.calculators import MEAM
from mdapy.calculators.testing import check_calculator
from mdapy.errors import InvalidParameter

from .meam_files import CASES, LIBRARY, write_library, write_parameters


def crystal(structure, a, elements, repeat=3, jitter=0.1, seed=0):
    """A shaken crystal of random elements."""
    rng = np.random.default_rng(seed)
    if structure == "zincblende":
        system = build_crystal("zincblende", a, elements=elements[:2], repeat=(repeat,) * 3)
    else:
        size = {"sc": 2 * repeat, "hcp": 2 * repeat, "bcc": repeat + 1}.get(structure, repeat)
        system = build_crystal(structure, a, elements=elements[0], repeat=(size,) * 3)
        system = system.with_columns(element=list(rng.choice(elements, len(system))))
    return system.with_columns(
        positions=system["positions"] + rng.normal(0, jitter, (len(system), 3))
    )


def make(tmp_path, names, lines):
    library = write_library(tmp_path / "library.meam")
    parameters = None if lines is None else write_parameters(tmp_path / "x.meam", lines)
    return MEAM(library, names, parameters), library, parameters


def test_the_reference_structure_has_the_rose_energy(tmp_path):
    # copper of nearest neighbours only, its second ones fully screened in
    # fcc: each atom has the energy of the Rose function of its distance
    meam, *_ = make(tmp_path, ["Cu"], None)
    alpha, alat, esub = 5.106, 3.62, 3.54
    re = alat / np.sqrt(2)
    for a in (3.5, 3.62, 3.75, 3.9):
        system = build_crystal("fcc", a, elements="Cu", repeat=(3, 3, 3))
        energy = meam.compute(system).energy / len(system)
        astar = alpha * (a / np.sqrt(2) / re - 1)
        assert energy == pytest.approx(-esub * (1 + astar) * np.exp(-astar), rel=1e-6)
    # at its lattice constant, no force, and no stress but for the
    # interpolation of the table of the pair function
    result = meam.compute(build_crystal("fcc", alat, elements="Cu", repeat=(3, 3, 3)), ("virial",))
    assert np.abs(result.forces).max() < 1e-10
    assert np.abs(result.virial).max() / len(result.forces) < 1e-7


@pytest.mark.parametrize(
    "case",
    [
        "copper and nickel, l12",
        "iron and copper, b2 of second neighbours, ialloy 2, erose_form 1",
        "silicon and carbon, dia, screening beyond rc",
    ],
)
def test_check_calculator(tmp_path, case):
    names, lines, structure, a = CASES[case]
    meam, *_ = make(tmp_path, names, lines)
    system = crystal(structure, a, meam.elements, repeat=2, jitter=0.15, seed=3)
    check_calculator(meam, system, displacement=1e-5, rtol=1e-5, atol=1e-6)


def test_the_neighbors_needed_reach_the_screening_atoms(tmp_path):
    meam, *_ = make(tmp_path, *CASES["silicon and carbon, dia, screening beyond rc"][:2])
    # Cmax = 4: an atom screens a pair up to rc C / (2 sqrt(C - 1)) from its atoms
    assert meam.rc == 4.0
    assert meam.cutoff == pytest.approx(4.0 * 4.0 / (2 * np.sqrt(3.0)))
    plain, *_ = make(tmp_path, ["Cu"], None)
    assert plain.cutoff == pytest.approx(4.0 * 2.8 / (2 * np.sqrt(1.8)))


def test_entries_and_elements(tmp_path):
    library = write_library(tmp_path / "library.meam")
    meam = MEAM(library, ["Cu", "Ni"])
    assert meam.elements == ("Cu", "Ni") and meam.masses[1] == pytest.approx(58.6934)
    # an entry not named by an element: by its atomic number, or mapped
    assert MEAM(library, ["CuX"]).elements == ("Cu",)
    assert MEAM(library, {"Cu": "CuX"}).elements == ("Cu",)
    with pytest.raises(InvalidParameter, match="two entries are of the same element"):
        MEAM(library, ["Cu", "CuX"])
    system = build_crystal("fcc", 3.62, elements="Cu", repeat=(2, 2, 2))
    with pytest.raises(InvalidParameter, match="no parameters for \\['Cu'\\]"):
        MEAM(library, ["Ni"]).compute(system)
    assert "MEAM(" in repr(meam) and "rc=4.0" in repr(meam)


@pytest.mark.parametrize(
    "names, lines, message",
    [
        (["Cu", "Cu"], None, "each entry of the library once"),
        (["Zz"], None, "no entries of \\['Zz'\\]"),
        (["Cu"], ["Ec(1,2) = 3"], "outside the 1 elements"),
        (["Cu", "Ni"], ["Cmin(1,2) = 1"], "Cmin needs 3 indices"),
        (["Cu", "Ni"], ["lattce(1,2) = 'c11'"], "c11 lattice is not supported"),
        (["Cu", "Ni"], ["lattce(1,2) = 'xyz'"], "unknown lattice 'xyz'"),
        (["Cu", "Ni"], ["theta(1,2) = 120"], "theta"),
        (["Cu"], ["nonsense = 1"], "unknown keyword"),
        (["Cu"], ["rho0(1) = 0"], "rho0 must be positive"),
        (["Cu"], ["rc = x"], "must be a number"),
        (["Cu"], ["Ec(a,1) = 3"], "whole numbers"),
        (["Cu"], ["ialloy = 3"], "ialloy must be one of"),
    ],
)
def test_errors(tmp_path, names, lines, message):
    with pytest.raises(InvalidParameter, match=message):
        make(tmp_path, names, lines)


@pytest.mark.parametrize(
    "replace, message",
    [
        (("'fcc' 12 29", "'c11' 12 29"), "unknown lattice 'c11'"),
        (("'fcc' 12 29", "'fcc' 8 29"), "not the coordination 12"),
        (("3.62 3.54 1.07\n1 3.14", "3.62 3.54 1.07\n2 3.14"), "t0 of 'Cu' must be 1"),
        (("2.95 1.0 3", "2.95 0.0 3"), "rho0 of 'Cu' must be positive"),
        (("2.95 1.0 3", "2.95 1.0 2"), "ibar must be one of"),
        (("5.106 3.634", "5.106 x"), "word for a number"),
    ],
)
def test_library_errors(tmp_path, replace, message):
    path = tmp_path / "library.meam"
    path.write_text(write_library(path, ["Cu"]).read_text().replace(*replace))
    with pytest.raises(InvalidParameter, match=message):
        MEAM(path, ["Cu"])
    with pytest.raises(InvalidParameter, match="cannot read"):
        MEAM(tmp_path / "missing", ["Cu"])
    with pytest.raises(InvalidParameter, match="cannot read"):
        MEAM(write_library(tmp_path / "ok.meam", ["Cu"]), ["Cu"], tmp_path / "missing")


def test_every_case_is_in_the_library():
    for names, *_ in CASES.values():
        assert all(name in LIBRARY for name in names)


class TestAgainstLammps:
    @pytest.fixture(autouse=True)
    def _lammps(self):
        pytest.importorskip("lammps")
        from .lammps_reference import has_style

        if not has_style("meam"):
            pytest.skip("LAMMPS without the MEAM package")

    @pytest.mark.parametrize("case", sorted(CASES))
    def test_the_cases(self, tmp_path, case):
        from .lammps_reference import assert_matches, reference

        names, lines, structure, a = CASES[case]
        meam, library, parameters = make(tmp_path, names, lines)
        system = crystal(structure, a, meam.elements, seed=len(case))
        pair = (
            f"pair_style meam\npair_coeff * * {library} {' '.join(names)} "
            f"{parameters or 'NULL'} {' '.join(names)}"
        )
        expected = reference(system, list(meam.elements), pair, tmp_path)
        assert_matches(
            meam.compute(system, ("atomic_virials",)),
            expected,
            energy=1e-11,
            forces=1e-9,
            virials=1e-9,
        )

    def test_a_surface_and_a_small_box(self, tmp_path):
        from .lammps_reference import assert_matches, reference

        names, lines, structure, a = CASES["copper and nickel, l12"]
        meam, library, parameters = make(tmp_path, names, lines)
        pair = f"pair_style meam\npair_coeff * * {library} Cu Ni {parameters} Cu Ni"
        slab = crystal(structure, a, meam.elements, seed=4)
        slab = slab.with_box(mdapy.Box(slab.box.matrix, pbc=(True, True, False)))
        expected = reference(slab, ["Cu", "Ni"], pair, tmp_path)
        assert_matches(meam.compute(slab, ("atomic_virials",)), expected, energy=1e-11, forces=1e-9)
        small = crystal(structure, a, meam.elements, repeat=1, jitter=0.15, seed=5)
        expected = reference(small, ["Cu", "Ni"], pair, tmp_path)
        assert_matches(meam.compute(small, ("atomic_virials",)), expected, energy=1e-11, forces=1e-9)
