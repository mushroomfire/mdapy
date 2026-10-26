# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import pytest

import mdapy
from mdapy.calculators import NEP
from mdapy.calculators._nep import _read_nep
from mdapy.calculators.testing import check_calculator
from mdapy.errors import FileFormatError, InvalidParameter, MissingProperty

from .eam_files import alloy
from .nep_files import write_nep
from .nep_reference import descriptors_and_energies

ZBL = [None, "universal", "typewise", "flexible"]
CUTOFFS = {"shared": ((5.0, 4.0),), "per element": ((5.0, 4.0), (4.4, 3.2))}


def compressed(seed, pbc=True):
    """Cu and Ni close enough for the ZBL repulsion."""
    return alloy(seed=seed, pbc=pbc, jitter=0.25, a=3.0)


def test_reads_the_file(tmp_path):
    path = write_nep(tmp_path / "nep.txt", cutoffs=CUTOFFS["per element"])
    potential = NEP(path)
    assert potential.elements == ("Cu", "Ni")
    assert potential.cutoff == 5.0
    assert repr(potential).startswith("NEP(") and "('Cu', 'Ni')" in repr(potential)
    model = _read_nep(path)
    assert model["dim"] == 5 + 4 * 6  # 5 radial, 4 n by (4 l + four- and five-body)
    np.testing.assert_array_equal(model["rc_radial"], [5.0, 4.4])
    np.testing.assert_array_equal(model["rc_angular"], [4.0, 3.2])


@pytest.mark.parametrize("zbl", ZBL)
@pytest.mark.parametrize("cutoffs", CUTOFFS)
def test_matches_a_numpy_reference(tmp_path, zbl, cutoffs):
    path = write_nep(tmp_path / "nep.txt", zbl=zbl, cutoffs=CUTOFFS[cutoffs])
    potential, model = NEP(path), _read_nep(path)
    for pbc in (True, (True, False, True), False):
        system = compressed(seed=1, pbc=pbc)
        descriptors, energies = descriptors_and_energies(model, system)
        np.testing.assert_allclose(
            potential.descriptors(system), descriptors, atol=1e-12
        )
        result = potential.compute(system, ("atomic_energies",))
        np.testing.assert_allclose(
            result.atomic_energies, energies, rtol=1e-12, atol=1e-12
        )


@pytest.mark.parametrize(
    "options",
    [
        {"l_max": (4, 0, 0)},
        {"l_max": (2, 2, 0)},
        {"l_max": (8, 2, 1)},
        {"l_max": (1, 0, 0), "n_max": (0, 0), "basis": (0, 0)},
    ],
)
def test_descriptor_sizes(tmp_path, options):
    path = write_nep(tmp_path / "nep.txt", **options)
    potential, model = NEP(path), _read_nep(path)
    system = alloy(seed=2)
    descriptors, energies = descriptors_and_energies(model, system)
    assert potential.descriptors(system).shape == (len(system), model["dim"])
    np.testing.assert_allclose(potential.descriptors(system), descriptors, atol=1e-12)
    np.testing.assert_allclose(
        potential.compute(system).energy, energies.sum(), rtol=1e-12
    )
    check_calculator(potential, system, rtol=1e-6, atol=1e-6)


def test_small_box_with_images(tmp_path):
    potential = NEP(write_nep(tmp_path / "nep.txt", zbl="universal"))
    system = mdapy.System([[0.1, 0.2, 0.3]], mdapy.Box(2.1), elements="Cu")
    _, energies = descriptors_and_energies(_read_nep(potential.path), system)
    result = potential.compute(system, ("energy", "forces", "virial"))
    np.testing.assert_allclose(result.atomic_energies, energies, rtol=1e-12)
    np.testing.assert_allclose(result.forces, 0.0, atol=1e-10)  # symmetric images


@pytest.mark.parametrize("zbl", ZBL)
@pytest.mark.parametrize("pbc", [True, (True, True, False), False])
def test_passes_the_calculator_checks(tmp_path, zbl, pbc):
    path = write_nep(tmp_path / "nep.txt", zbl=zbl, cutoffs=CUTOFFS["per element"])
    check_calculator(NEP(path), compressed(seed=3, pbc=pbc), rtol=1e-6, atol=1e-6)


def test_same_result_for_any_thread_count(tmp_path):
    potential = NEP(write_nep(tmp_path / "nep.txt", zbl="typewise"))
    system = alloy(seed=4, repeat=4)
    results = []
    for threads in (1, 4):
        with mdapy.num_threads(threads):
            results.append(
                (
                    potential.compute(system, ("forces", "atomic_virials")),
                    potential.descriptors(system),
                )
            )
    np.testing.assert_array_equal(results[0][0].forces, results[1][0].forces)
    np.testing.assert_array_equal(
        results[0][0].atomic_virials, results[1][0].atomic_virials
    )
    np.testing.assert_array_equal(results[0][1], results[1][1])


def test_errors(tmp_path):
    potential = NEP(write_nep(tmp_path / "nep.txt"))
    system = alloy(seed=5)
    with pytest.raises(MissingProperty, match="NEP needs the element column"):
        potential.compute(mdapy.System(system["positions"], system.box))
    with pytest.raises(InvalidParameter, match=r"no parameters for \['Al'\]"):
        potential.descriptors(system.with_columns(element=["Al"] * len(system)))
    with pytest.raises(InvalidParameter, match="below the required"):
        potential.compute(system, neighbors=mdapy.NeighborList.build(system, 3.0))
    shared = mdapy.NeighborList.build(system, 6.0)
    np.testing.assert_array_equal(
        potential.descriptors(system, neighbors=shared), potential.descriptors(system)
    )


@pytest.mark.parametrize(
    "edit, message",
    [
        (
            lambda lines: lines.__setitem__(0, "nep3 2 Cu Ni"),
            "unsupported model 'nep3'",
        ),
        (lambda lines: lines.__setitem__(0, "nep4 x Cu Ni"), "number of elements"),
        (
            lambda lines: lines.__setitem__(0, "nep4 3 Cu Ni"),
            "expected 3 element names",
        ),
        (lambda lines: lines.__setitem__(0, "nep4 2 Cu Qq"), "unknown element 'Qq'"),
        (
            lambda lines: lines.__setitem__(1, "cut 5 4 100 40"),
            "expected the cutoff line",
        ),
        (
            lambda lines: lines.__setitem__(1, "cutoff 5 4 3 100 40"),
            "needs 4 or 6 values",
        ),
        (
            lambda lines: lines.__setitem__(1, "cutoff 5 x 100 40"),
            "expected float values",
        ),
        (
            lambda lines: lines.__setitem__(1, "cutoff 5 -4 100 40"),
            "cutoffs must be positive",
        ),
        (lambda lines: lines.__setitem__(2, "n_max 4 x"), "expected int values"),
        (lambda lines: lines.__setitem__(4, "l_max 9 0 0"), "l_max needs 1 to 8"),
        (
            lambda lines: lines.__setitem__(4, "l_max 4 0 1"),
            "five-body terms only with",
        ),
        (lambda lines: lines.__setitem__(5, "ANN 0 0"), "neurons positive"),
        (lambda lines: lines.__setitem__(8, "1.0 x"), "expected numbers"),
        (lambda lines: lines.__delitem__(slice(20, None)), "expected 602 parameters"),
        (lambda lines: lines.__delitem__(slice(3, None)), "unexpected end of file"),
        (lambda lines: lines.__setitem__(slice(None), []), "unsupported model ''"),
    ],
)
def test_invalid_files(tmp_path, edit, message):
    path = write_nep(tmp_path / "nep.txt", l_max=(4, 2, 1))
    lines = path.read_text().splitlines()
    edit(lines)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(FileFormatError, match=message):
        NEP(path)


@pytest.mark.parametrize(
    "line, message",
    [
        ("zbl 2.0 1.0", "0 <= inner < outer"),
        ("zbl 1.0", "needs 2 or 3 values"),
    ],
)
def test_invalid_zbl(tmp_path, line, message):
    path = write_nep(tmp_path / "nep.txt", zbl="universal")
    lines = path.read_text().splitlines()
    lines[1] = line
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(FileFormatError, match=message):
        NEP(path)


def test_invalid_flexible_zbl(tmp_path):
    path = write_nep(tmp_path / "nep.txt", zbl="flexible")
    lines = path.read_text().splitlines()
    lines[-10] = "3.0"  # an inner cutoff beyond the outer one
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(FileFormatError, match="flexible ZBL cutoffs"):
        NEP(path)


def test_blank_lines_before_the_header(tmp_path):
    path = write_nep(tmp_path / "nep.txt")
    a = NEP(path)
    path.write_text("\n\n" + path.read_text().replace("\nn_max", "\n\nn_max"))
    b = NEP(path)
    system = alloy(seed=6)
    assert a.compute(system).energy == b.compute(system).energy


def test_neighbor_lists_must_be_symmetric(tmp_path):
    potential = NEP(write_nep(tmp_path / "nep.txt"))
    system = alloy(seed=7)
    nearest = mdapy.NeighborList.build(system, num_neighbors=120)
    assert nearest.cutoff > potential.cutoff
    with pytest.raises(InvalidParameter, match="nearest neighbors"):
        potential.compute(system, neighbors=nearest)
    # a list within a cutoff without one of the two directions of a pair
    full = mdapy.NeighborList.build(system, potential.cutoff)
    keep = np.ones(len(full), dtype=bool)
    keep[0] = False
    kept = np.concatenate([[0], np.cumsum(keep)])
    broken = mdapy.NeighborList(
        system,
        full.cutoff,
        kept[full.offsets],
        full.j[keep],
        full.shifts[keep],
        full.distances[keep],
    )
    with pytest.raises(ValueError, match="not symmetric"):
        potential.compute(system, neighbors=broken)
