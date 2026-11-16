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

from .eam_files import (
    CUTOFF,
    FUNCFL_GRIDS,
    alloy,
    charge,
    density,
    embedding,
    reference,
    write_funcfl,
    write_setfl,
)


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
    stress = result.stress
    np.testing.assert_allclose(
        stress, -result.virial / system.box.volume * mdapy.units.EV_PER_A3_TO_GPA
    )
    assert not result.forces.flags.writeable
    with pytest.raises(InvalidParameter, match="virial was not computed"):
        CalculatorResult(energy=1.0).stress
    with pytest.raises(InvalidParameter, match="volume"):
        CalculatorResult(virial=np.zeros((3, 3))).stress


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


@pytest.fixture
def fs_potential(tmp_path):
    return EAM(write_setfl(tmp_path / "CuNi.eam.fs", fs=True))


@pytest.mark.parametrize("pbc", [True, (True, False, True), False])
def test_fs_matches_a_numpy_reference(fs_potential, potential, pbc):
    # each element gives the other a density of its own: eam/fs
    system = alloy(seed=1, pbc=pbc)
    result = fs_potential.compute(
        system, ("energy", "forces", "virial", "atomic_virials")
    )
    energies, forces, virials = reference(system, fs=True)
    np.testing.assert_allclose(result.atomic_energies, energies, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.forces, forces, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(result.atomic_virials, virials, rtol=1e-11, atol=1e-11)
    assert abs(result.energy - potential.compute(system).energy) > 1.0


def test_fs_passes_the_calculator_checks(fs_potential):
    check_calculator(fs_potential, alloy(seed=2), rtol=1e-6, atol=1e-6)


def test_fs_written_and_read_back(fs_potential, tmp_path):
    fs_potential.write(tmp_path / "out.eam.fs")
    again = EAM(tmp_path / "out.eam.fs")
    system = alloy(seed=4)
    assert again.compute(system).energy == fs_potential.compute(system).energy
    # four densities of 1000 values, in four lines of the file each
    lines = (tmp_path / "out.eam.fs").read_text().splitlines()
    assert len(lines) == 5 + 2 * (1 + 200 + 2 * 200) + 3 * 200


def test_fs_of_the_same_densities_is_an_alloy(potential, tmp_path):
    # an eam/fs file in which each element gives every element the same
    # density: the eam/alloy potential of its functions, average atom and all
    same = EAM(write_setfl(tmp_path / "same.eam.fs", fs=True, factors={}))
    system = alloy(seed=8)
    assert same.compute(system).energy == potential.compute(system).energy
    average = {"Cu": 0.3, "Ni": 0.7}
    a, b = same.average(average), potential.average(average)
    assert a.compute(system).energy == b.compute(system).energy


def test_fs_has_no_average_atom(fs_potential):
    with pytest.raises(InvalidParameter, match="gives each element its own"):
        fs_potential.average([0.5, 0.5])


def test_funcfl_is_the_alloy_of_its_tables(tmp_path):
    # one funcfl file: LAMMPS keeps its values but the last, and makes the pair
    # r phi = 27.2 0.529 Z(r)^2; the same tables written by hand as setfl
    path = write_funcfl(tmp_path / "Cu.eam", "Cu")
    potential = EAM(path)
    assert potential.elements == ("Cu",) and potential.path == str(path)
    assert potential.cutoff == CUTOFF and potential.masses[0] == 63.546
    nrho, drho, nr, dr = FUNCFL_GRIDS["Cu"]
    r, rho = np.arange(nr - 1) * dr, np.arange(nrho - 1) * drho
    values = [
        embedding("Cu", rho),
        density("Cu", r),
        27.2 * 0.529 * charge("Cu", r) * charge("Cu", r),  # in the order of LAMMPS
    ]
    lines = ["", "", "", "1 Cu", f"{nrho - 1} {drho!r} {nr - 1} {dr!r} {CUTOFF!r}"]
    lines.insert(5, "29 63.546 3.6 fcc")
    lines += [" ".join(repr(float(x)) for x in v) for v in values]
    (tmp_path / "Cu.eam.alloy").write_text("\n".join(lines) + "\n")
    setfl = EAM(tmp_path / "Cu.eam.alloy")
    system = alloy(seed=1).with_columns(element=["Cu"] * 32)
    # the grid interpolated onto itself, as LAMMPS does: equal within rounding
    a, b = potential.compute(system), setfl.compute(system)
    np.testing.assert_allclose(a.forces, b.forces, rtol=1e-11, atol=1e-11)
    assert a.energy == pytest.approx(b.energy, rel=1e-13)


def test_funcfl_files_mix(tmp_path):
    # two files on grids of their own: an alloy on the larger spacings and
    # ranges, the cutoff the largest, the pair of Cu and Ni from both charges
    files = [write_funcfl(tmp_path / f"{e}.eam", e) for e in ("Cu", "Ni")]
    potential = EAM(files)
    assert potential.elements == ("Cu", "Ni")
    assert potential.path == tuple(str(f) for f in files)
    np.testing.assert_array_equal(potential.masses, [63.546, 58.6934])
    check_calculator(potential, alloy(seed=2), rtol=1e-6, atol=1e-6)
    average = potential.average({"Cu": 0.5, "Ni": 0.5})
    assert np.isfinite(average.compute(alloy(seed=2)).energy)
    # written as eam/alloy, which reads back to the same potential
    potential.write(tmp_path / "CuNi.eam.alloy")
    again = EAM(tmp_path / "CuNi.eam.alloy")
    system = alloy(seed=3)
    assert again.compute(system).energy == potential.compute(system).energy


def test_funcfl_without_lattice(tmp_path):
    # the lattice constant and the lattice may be left out, as LAMMPS reads
    # the atomic number and the mass only
    full = write_funcfl(tmp_path / "Cu.eam", "Cu")
    lines = full.read_text().splitlines()
    lines[1] = "29 63.546"
    (tmp_path / "short.eam").write_text("\n".join(lines) + "\n")
    system = alloy(seed=4).with_columns(element=["Cu"] * 32)
    short = EAM(tmp_path / "short.eam")
    assert short.compute(system).energy == EAM(full).compute(system).energy


def test_funcfl_errors(tmp_path):
    cu = write_funcfl(tmp_path / "Cu.eam", "Cu")
    with pytest.raises(InvalidParameter, match="repeat"):
        EAM([cu, cu])
    with pytest.raises(InvalidParameter, match="funcfl files"):
        EAM([cu, write_setfl(tmp_path / "CuNi.eam.alloy")])
    with pytest.raises(InvalidParameter, match="give the file"):
        EAM([])
    for line, text, message in [
        (1, "0 63.5 3.6 fcc", "0 is not the atomic number"),
        (1, "x", "atomic number and the mass"),
        (2, "900 0.06", "Nrho drho Nr dr cutoff"),
        (2, "3 0.06 1000 0.0052 5.0", "at least 5"),
    ]:
        lines = cu.read_text().splitlines()
        lines[line] = text
        bad = tmp_path / "bad.eam"
        bad.write_text("\n".join(lines) + "\n")
        with pytest.raises(FileFormatError, match=message):
            EAM(bad)


def test_progress_runs_once_from_start_to_end(potential):
    """The passes of the computation make one count: it never falls, and it
    reaches its total at the end."""
    from .eam_files import alloy

    calls = []
    potential.compute(
        alloy(seed=1, repeat=8), progress=lambda d, t: calls.append((d, t))
    )
    done = [d for d, _ in calls]
    assert len({t for _, t in calls}) == 1
    assert done == sorted(done)
    assert calls[-1][0] == calls[-1][1]
