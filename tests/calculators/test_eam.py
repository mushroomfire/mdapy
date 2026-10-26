# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import pytest

import mdapy
from mdapy.calculators import EAM, CalculatorResult
from mdapy.calculators.testing import check_calculator
from mdapy.errors import FileFormatError, InvalidParameter, MissingProperty

from .eam_files import CUTOFF, alloy, reference, write_setfl


@pytest.fixture
def potential(tmp_path):
    return EAM(write_setfl(tmp_path / "CuNi.eam.alloy"))


def test_reads_the_file(potential):
    assert potential.elements == ("Cu", "Ni")
    assert potential.cutoff == CUTOFF
    np.testing.assert_array_equal(potential.masses, [63.546, 58.6934])
    assert repr(potential).startswith("EAM(") and "('Cu', 'Ni')" in repr(potential)


@pytest.mark.parametrize("pbc", [True, (True, False, True), False])
def test_matches_a_numpy_reference(potential, pbc):
    system = alloy(seed=1, pbc=pbc)
    result = potential.compute(system, ("energy", "forces", "virial", "atomic_virials"))
    energies, forces, virials = reference(system)
    np.testing.assert_allclose(result.atomic_energies, energies, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.forces, forces, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(result.atomic_virials, virials, rtol=1e-11, atol=1e-11)
    assert result.energy == pytest.approx(energies.sum(), rel=1e-13)


def test_small_box_with_images(potential):
    # one Cu atom in a box smaller than the cutoff: it sees its own images
    system = mdapy.System([[0.1, 0.2, 0.3]], mdapy.Box(2.6), elements="Cu")
    result = potential.compute(system)
    energies, forces, _ = reference(system)
    np.testing.assert_allclose(result.atomic_energies, energies, rtol=1e-12)
    np.testing.assert_allclose(result.forces, 0.0, atol=1e-12)  # symmetric images


@pytest.mark.parametrize("pbc", [True, (True, True, False), False])
def test_passes_the_calculator_checks(potential, pbc):
    check_calculator(potential, alloy(seed=2, pbc=pbc), rtol=1e-6, atol=1e-6)


def test_same_result_for_any_thread_count(potential):
    system = alloy(seed=3, repeat=4)
    results = []
    for threads in (1, 4):
        with mdapy.num_threads(threads):
            results.append(potential.compute(system))
    np.testing.assert_array_equal(results[0].forces, results[1].forces)
    assert results[0].energy == results[1].energy


def test_results_attach_to_systems(potential):
    system = alloy(seed=4)
    result = potential.compute(system, ("energy", "forces", "virial"))
    attached = result.attach(system)
    np.testing.assert_array_equal(attached.table["forces"].to_numpy(), result.forces)
    assert attached.info["energy"] == result.energy
    np.testing.assert_array_equal(attached.info["virial"], result.virial)
    stress = result.stress(system)
    np.testing.assert_allclose(
        stress, -result.virial / system.box.volume * mdapy.units.EV_PER_A3_TO_GPA
    )
    assert not result.forces.flags.writeable
    with pytest.raises(InvalidParameter, match="virial was not computed"):
        CalculatorResult(energy=1.0).stress(system)


def test_errors(potential, tmp_path):
    system = alloy(seed=5)
    with pytest.raises(MissingProperty, match="element column"):
        potential.compute(mdapy.System(system["positions"], system.box))
    with pytest.raises(InvalidParameter, match=r"no parameters for \['Al'\]"):
        potential.compute(system.with_columns(element=["Al"] * len(system)))
    with pytest.raises(InvalidParameter, match="unknown properties"):
        potential.compute(system, ("charges",))
    small = mdapy.NeighborList.build(system, 3.0)
    with pytest.raises(InvalidParameter, match="below the required"):
        potential.compute(system, neighbors=small)
    other = mdapy.NeighborList.build(alloy(seed=6), CUTOFF)
    with pytest.raises(InvalidParameter, match="other positions or another box"):
        potential.compute(system, neighbors=other)
    nearest = mdapy.NeighborList.build(system, num_neighbors=40)
    with pytest.raises(InvalidParameter, match="nearest neighbors"):
        potential.compute(system, neighbors=nearest)


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda lines: lines.__setitem__(3, "x Cu Ni"), "number of elements"),
        (lambda lines: lines.__setitem__(3, "3 Cu Ni"), "expected 3 element names"),
        (
            lambda lines: lines.__setitem__(4, "1000 0.05 1000"),
            "Nrho drho Nr dr cutoff",
        ),
        (lambda lines: lines.__setitem__(4, "3 0.05 1000 0.0055 5.0"), "at least 5"),
        (lambda lines: lines.__setitem__(5, "29"), "atomic number and the mass"),
        (lambda lines: lines.__setitem__(6, "x " + lines[6]), "expected numbers"),
        (lambda lines: lines.__delitem__(slice(300, None)), "unexpected end of file"),
    ],
)
def test_invalid_files(tmp_path, edit, message):
    path = write_setfl(tmp_path / "bad.eam.alloy")
    lines = path.read_text().splitlines()
    edit(lines)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(FileFormatError, match=message):
        EAM(path)


def test_values_may_span_lines_and_have_comments(tmp_path):
    a = EAM(write_setfl(tmp_path / "a.eam.alloy", per_line=5))
    path = write_setfl(tmp_path / "b.eam.alloy", per_line=7)
    lines = path.read_text().replace("\n", "  # comment\n", 7).splitlines()
    lines[8:8] = ["", "# a comment line", "   "]  # inside the first table
    path.write_text("\n".join(lines) + "\n")
    b = EAM(path)
    system = alloy(seed=7)
    assert a.compute(system).energy == b.compute(system).energy
