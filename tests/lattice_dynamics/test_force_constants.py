# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Force constants and their FORCE_CONSTANTS files: round trips, files that
leave out atoms, and the errors. The files against phonopy itself are in
test_force_constants_phonopy.py."""

import dataclasses

import numpy as np
import pytest

import mdapy
from mdapy.build import build_crystal
from mdapy.errors import FileFormatError, InvalidParameter
from mdapy.lattice_dynamics import ForceConstants, phonons, read_force_constants
from mdapy.lattice_dynamics._force_constants import _phonopy_atoms

from calculators.test_testing import Harmonic

HEXAGONAL = np.array([[2, -1, 0], [1, 2, 0], [0, 0, 2]])


def springs(structure, a, matrix, elements="Ar"):
    """The force constants of a crystal of springs at rest between nearest
    neighbors."""
    cell = build_crystal(structure, a, elements=elements)
    r0 = a / np.sqrt(2) if structure in ("fcc", "l1_2") else a
    result = phonons(cell, Harmonic(k=0.7, r0=r0), supercell=matrix)
    return ForceConstants(result.cell, result.supercell, result.force_constants)


@pytest.fixture(scope="module")
def cubic():
    """fcc: the cubic cell of four atoms, each a translation of the first."""
    return springs("fcc", 3.6, (2, 2, 3))


@pytest.fixture(scope="module")
def hexagonal():
    """hcp of two atoms, in a supercell of a matrix that is not diagonal."""
    return springs("hcp", 2.5, HEXAGONAL)


def blocks(path):
    """The header and the blocks of four lines of a FORCE_CONSTANTS."""
    lines = path.read_text().splitlines()
    return lines[0], [lines[k : k + 4] for k in range(1, len(lines), 4)]


def rewritten(path, keep):
    """The file with only the rows of the atoms that keep accepts, by their
    label in the file."""
    header, found = blocks(path)
    columns = int(header.split()[1])
    kept = [b for b in found if keep(int(b[0].split()[0]))]
    text = [f"{len(kept) // columns:4d} {columns:4d}"] + [x for b in kept for x in b]
    path.write_text("\n".join(text) + "\n")


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("crystal", ["cubic", "hexagonal"])
def test_the_round_trip(crystal, full, request, tmp_path):
    constants = request.getfixturevalue(crystal)
    path = tmp_path / "FORCE_CONSTANTS"
    constants.write(path, full=full)
    n, size = len(constants.cell), round(np.linalg.det(constants.supercell))
    header, found = blocks(path)
    assert header.split() == [str(n * size if full else n), str(n * size)]
    assert len(found) == (n * size if full else n) * n * size
    back = read_force_constants(path, constants.cell, constants.supercell)
    np.testing.assert_array_equal(back.supercell, constants.supercell)
    # the file holds 15 decimals
    np.testing.assert_allclose(back.second, constants.second, rtol=0, atol=1e-14)


def test_the_phonons_are_those_of_the_force_constants():
    cell = build_crystal("fcc", 3.6, elements="Ar")
    result = phonons(cell, Harmonic(k=0.7, r0=3.6 / np.sqrt(2)), supercell=2)
    constants = ForceConstants(result.cell, result.supercell, result.force_constants)
    q = np.random.default_rng(0).random((10, 3)) - 0.5
    np.testing.assert_array_equal(
        constants.phonons().frequencies(q), result.frequencies(q)
    )
    assert repr(constants) == "ForceConstants(4 atoms, supercell of 8 cells)"
    one = build_crystal("sc", 2.5, elements="Ar")
    assert repr(ForceConstants(one, 1, np.zeros((1, 1, 3, 3)))) == (
        "ForceConstants(1 atom, supercell of 1 cells)"
    )


def test_the_order_of_phonopy_for_repeats():
    """Atom by atom of the cell, the cells with the first vector fastest."""
    counts, n = (2, 3, 4), 2
    ours = _phonopy_atoms(np.diag(counts), n)
    expected = [
        ((i * counts[1] + j) * counts[2] + k) * n + u
        for u in range(n)
        for k in range(counts[2])
        for j in range(counts[1])
        for i in range(counts[0])
    ]
    np.testing.assert_array_equal(ours, expected)


def test_rows_left_out_come_from_translations(cubic, hexagonal, tmp_path):
    """A file with the rows of atoms of other cells only, or of one atom of
    the cubic cell only, as phonopy writes for its primitive cell, gives the
    same force constants: those of the atoms of the cubic cell, equivalent by
    translations, are the same to rounding."""
    path = tmp_path / "FORCE_CONSTANTS"
    for constants in (cubic, hexagonal):
        size = round(np.linalg.det(constants.supercell))
        constants.write(path, full=True)
        rewritten(path, lambda label: (label - 1) % size != 0)  # not the first cell
        back = read_force_constants(path, constants.cell, constants.supercell)
        np.testing.assert_allclose(back.second, constants.second, rtol=0, atol=1e-12)
    cubic.write(path)
    rewritten(path, lambda label: label == 1)
    back = read_force_constants(path, cubic.cell, cubic.supercell)
    np.testing.assert_allclose(back.second, cubic.second, rtol=0, atol=1e-12)


def test_a_sheet(tmp_path):
    """A triangular sheet of carbon, periodic along its plane only."""
    a, height = 2.5, 10.0
    plane = np.array([[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, height]])
    sheet = mdapy.System(
        np.array([[0, 0, height / 2]]),
        mdapy.Box(plane, pbc=(True, True, False)),
        elements=["C"],
    )
    result = phonons(sheet, Harmonic(k=2.0, r0=a), supercell=3, symmetry=False)
    constants = ForceConstants(result.cell, 3, result.force_constants)
    np.testing.assert_array_equal(constants.supercell, np.diag([3, 3, 1]))
    path = tmp_path / "FORCE_CONSTANTS"
    constants.write(path, full=True)
    back = read_force_constants(path, constants.cell, 3)
    np.testing.assert_allclose(back.second, constants.second, rtol=0, atol=1e-14)


def test_a_header_of_one_number(cubic, tmp_path):
    """phonopy reads a header of one number as the rows and the columns."""
    path = tmp_path / "FORCE_CONSTANTS"
    cubic.write(path, full=True)
    lines = path.read_text().splitlines()
    path.write_text("\n".join([lines[0].split()[0], *lines[1:]]))
    back = read_force_constants(path, cubic.cell, cubic.supercell)
    np.testing.assert_allclose(back.second, cubic.second, rtol=0, atol=1e-14)


def test_the_force_constants_cannot_change(cubic):
    with pytest.raises(dataclasses.FrozenInstanceError):
        cubic.second = cubic.second
    for array in (cubic.supercell, cubic.second):
        with pytest.raises(ValueError):
            array[0] = 0


def test_errors(cubic, tmp_path):
    cell, matrix = cubic.cell, cubic.supercell
    with pytest.raises(InvalidParameter, match="shape"):
        ForceConstants(cell, matrix, cubic.second[:, :-1])
    with pytest.raises(InvalidParameter, match="finite"):
        ForceConstants(cell, matrix, np.full_like(cubic.second, np.nan))
    with pytest.raises(InvalidParameter, match="supercell"):
        ForceConstants(cell, 0, cubic.second)
    with pytest.raises(InvalidParameter, match="with atoms"):
        ForceConstants(
            cell.filter(np.zeros(len(cell), dtype=bool)), 1, np.zeros((0, 0, 3, 3))
        )
    closed = mdapy.System(
        cell["positions"], mdapy.Box(cell.box.matrix, pbc=False), elements=["Ar"] * 4
    )
    with pytest.raises(InvalidParameter, match="or a sheet"):
        ForceConstants(closed, 1, np.zeros((4, 4, 3, 3)))
    with pytest.raises(InvalidParameter, match="mass of every atom"):
        ForceConstants(
            mdapy.System(cell["positions"], cell.box), matrix, cubic.second
        ).phonons()

    path = tmp_path / "FORCE_CONSTANTS"
    cubic.write(path)
    good = path.read_text().splitlines()

    def broken(lines, match="not a FORCE_CONSTANTS of phonopy"):
        path.write_text("\n".join(lines) + "\n")
        with pytest.raises(FileFormatError, match=match):
            read_force_constants(path, cell, matrix)

    broken(["four atoms"])
    broken(good[:-1])  # a line short
    broken([good[0], "first", *good[2:]])  # no atom
    broken([good[0], "", *good[2:]])  # a line short, without the empty one
    broken([good[0], *good[1:5], "2 2", *good[6:]])  # two atoms in a row
    broken([good[0], *good[1:4], "1.0 x 2.0", *good[5:]])
    broken(["   4    0", *good[1:]])
    labels = [k for k in range(1, len(good), 4)]
    out = list(good)
    for k in labels[: len(labels) // 4]:
        out[k] = f"999 {out[k].split()[1]}"
    broken(out, "atom 999 of the file is not in the supercell")

    # as phonopy, only the first number of the first line of a block counts
    loose = [
        f"{line.split()[0]} anything" if k % 4 == 1 else line
        for k, line in enumerate(good)
    ]
    path.write_text("\n".join(loose) + "\n")
    back = read_force_constants(path, cell, matrix)
    np.testing.assert_allclose(back.second, cubic.second, rtol=0, atol=1e-14)

    path.write_text("\n".join(good) + "\n")
    with pytest.raises(
        InvalidParameter, match="48 atoms; the supercell of 27 cells of 4 atoms has 108"
    ):
        read_force_constants(path, cell, 3)


def test_an_atom_that_no_translation_reaches(tmp_path):
    """Nothing in the file gives the force constants of the Ni atoms."""
    constants = springs("l1_2", 3.6, 2, elements=("Cu", "Ni"))
    size = round(np.linalg.det(constants.supercell))
    path = tmp_path / "FORCE_CONSTANTS"
    constants.write(path, full=True)
    cu = np.flatnonzero(constants.cell.table["element"].cast(str).to_numpy() == "Cu")
    rewritten(path, lambda label: (label - 1) // size in cu)
    with pytest.raises(InvalidParameter, match="no force constants of atom"):
        read_force_constants(path, constants.cell, constants.supercell)
