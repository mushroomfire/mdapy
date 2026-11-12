# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""FORCE_CONSTANTS files against phonopy: its order of the atoms of a
supercell, its reader on the files of MDAPY, and MDAPY's reader on its
files, compact and full. Skipped where phonopy is not installed."""

import warnings

import numpy as np
import pytest

import mdapy
from mdapy.build import build_crystal
from mdapy.calculators import EAM
from mdapy.elements import atomic_masses
from mdapy.lattice_dynamics import ForceConstants, phonons, read_force_constants
from mdapy.lattice_dynamics._force_constants import _phonopy_atoms
from mdapy.lattice_dynamics._phonons import _supercell
from mdapy.simulation import minimize

from calculators.eam_files import write_setfl

phonopy = pytest.importorskip("phonopy")

# small enough that phonopy's force constants, from other displacements than
# those of MDAPY, are the same to the cube of it
DISPLACEMENT = 2e-5

CRYSTALS = {
    "fcc cubic cell": ("fcc", 3.29, "Cu", np.diag([3, 3, 3])),
    "L1_2 Cu3Ni": (
        "l1_2",
        3.29,
        ("Cu", "Ni"),
        np.array([[0, 2, 2], [2, 0, 2], [2, 2, 0]]),
    ),
    "hcp": ("hcp", 2.33, "Cu", np.array([[2, -1, 0], [1, 2, 0], [0, 0, 2]])),
}


@pytest.fixture(scope="module")
def potential(tmp_path_factory):
    return EAM(write_setfl(tmp_path_factory.mktemp("eam") / "CuNi.eam.alloy"))


def of_phonopy(cell, matrix, primitive):
    """phonopy's calculation of the cell, its supercell matrix the transpose
    of MDAPY's."""
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    names = cell.table["element"].cast(str).to_list()
    unit = PhonopyAtoms(
        symbols=names,
        cell=cell.box.matrix,
        scaled_positions=cell.box.to_fractional(cell["positions"]),
        masses=atomic_masses(names),
    )
    # phonopy warns of its primitive cells and of symmetries that differ
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return Phonopy(unit, supercell_matrix=matrix.T, primitive_matrix=primitive)


@pytest.fixture(scope="module", params=CRYSTALS)
def crystal(request, potential):
    structure, a, elements, matrix = CRYSTALS[request.param]
    cell = minimize(
        build_crystal(structure, a, elements=elements),
        potential,
        cell="full",
        max_force=1e-10,
        max_stress=1e-8,
        max_steps=10000,
    ).system
    result = phonons(cell, potential, supercell=matrix, displacement=DISPLACEMENT)
    return cell, matrix, result


@pytest.mark.parametrize(
    "matrix",
    [
        np.diag([2, 3, 4]),
        np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]]),
        np.array([[0, 2, 2], [2, 0, 2], [2, 2, 0]]),
        np.array([[2, 1, 0], [0, 2, 1], [1, 0, 3]]),
        np.array([[2, -1, 0], [1, 2, 0], [0, 0, 2]]),
    ],
)
def test_the_order_of_the_atoms_of_phonopy(matrix):
    cell = build_crystal("hcp", 2.5, elements="Cu")
    supercell = of_phonopy(cell, matrix, "P").supercell
    big = _supercell(cell, matrix)
    np.testing.assert_allclose(supercell.cell, big.box.matrix, atol=1e-12)
    ours = _phonopy_atoms(matrix, len(cell))
    apart = big.box.to_fractional(big["positions"][ours]) - supercell.scaled_positions
    np.testing.assert_allclose(apart - np.round(apart), 0, atol=1e-12)


@pytest.mark.parametrize("full", [False, True])
def test_phonopy_reads_the_files(crystal, full, tmp_path):
    """Compact files for phonopy with the cell as its primitive cell, full
    files whatever its primitive cell."""
    from phonopy.file_IO import parse_FORCE_CONSTANTS

    cell, matrix, result = crystal
    path = tmp_path / "FORCE_CONSTANTS"
    ForceConstants(cell, matrix, result.force_constants).write(path, full=full)
    theirs = of_phonopy(cell, matrix, "auto" if full else "P")
    theirs.force_constants = parse_FORCE_CONSTANTS(path, theirs.primitive.p2s_map)
    q = np.random.default_rng(0).random((20, 3)) - 0.5
    # phonopy's wave vectors are of its primitive cell
    to_primitive = theirs.primitive.cell @ np.linalg.inv(cell.box.matrix)
    theirs.run_qpoints(q @ to_primitive.T)
    found, ours = theirs.qpoints.frequencies, result.frequencies(q)
    if found.shape != ours.shape:
        # the modes of a primitive cell smaller than the cell are among those
        # of the cell at the same wave vector
        assert full and found.shape[1] < ours.shape[1]
        nearest = np.abs(found[:, :, None] - ours[:, None, :]).argmin(axis=2)
        ours = np.take_along_axis(ours, nearest, axis=1)
    # phonopy's factor from eV/(Å² amu) to THz differs by a millionth
    np.testing.assert_allclose(found, ours, rtol=1e-6, atol=1e-6)


def test_the_text_of_phonopy(crystal, tmp_path):
    from phonopy.file_IO import write_FORCE_CONSTANTS

    cell, matrix, result = crystal
    n = len(cell)
    size = round(np.linalg.det(matrix))
    ours, theirs = tmp_path / "ours", tmp_path / "theirs"
    ForceConstants(cell, matrix, result.force_constants).write(ours)
    constants = result.force_constants[:, _phonopy_atoms(matrix, n)]
    write_FORCE_CONSTANTS(constants, theirs, p2s_map=np.arange(n) * size)
    assert ours.read_text() == theirs.read_text() + "\n"


@pytest.mark.parametrize("full", [False, True])
def test_the_files_of_phonopy(crystal, potential, full, tmp_path):
    """phonopy's force constants of the same potential, on its own primitive
    cell, which is smaller than the cubic cell of fcc: the compact file has
    one row, the others come from translations."""
    from phonopy.file_IO import write_FORCE_CONSTANTS

    cell, matrix, result = crystal
    theirs = of_phonopy(cell, matrix, "auto")
    theirs.generate_displacements(distance=DISPLACEMENT, is_plusminus=True)
    theirs.forces = np.array(
        [
            potential.compute(
                mdapy.System(s.positions, mdapy.Box(s.cell), elements=s.symbols),
                ("forces",),
            ).forces
            for s in theirs.supercells_with_displacements
        ]
    )
    theirs.produce_force_constants(calculate_full_force_constants=full)
    rows = theirs.force_constants.shape[0]
    assert rows == (len(theirs.supercell) if full else len(theirs.primitive))
    path = tmp_path / "FORCE_CONSTANTS"
    write_FORCE_CONSTANTS(theirs.force_constants, path, theirs.primitive.p2s_map)
    read = read_force_constants(path, cell, matrix)
    np.testing.assert_allclose(read.second, result.force_constants, atol=1e-7)


def test_group_velocities_of_phonopy(crystal, tmp_path):
    """phonopy on the same force constants, with its symmetry: the same
    velocities at any wave vector and at those of equal frequencies, to its
    constants."""
    from phonopy.file_IO import parse_FORCE_CONSTANTS

    cell, matrix, result = crystal
    path = tmp_path / "FORCE_CONSTANTS"
    ForceConstants(cell, matrix, result.force_constants).write(path)
    theirs = of_phonopy(cell, matrix, "P")
    theirs.force_constants = parse_FORCE_CONSTANTS(path, theirs.primitive.p2s_map)
    q = np.vstack(
        [
            np.random.default_rng(0).random((10, 3)) - 0.5,
            [[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0.25, 0, 0], [0.5, 0.5, 0.5]],
            [[0.1, 0.1, 0.1], [0.2, 0.0, 0.2], [1 / 3, 1 / 3, 0], [0, 0, 0.25]],
        ]
    )
    theirs.run_qpoints(q, with_group_velocities=True)
    expected = theirs.qpoints.group_velocities  # THz Å, which is Å/ps
    found = result.group_velocities(q)
    np.testing.assert_allclose(
        found, expected, rtol=0, atol=1e-6 * np.abs(expected).max()
    )


def test_thermal_displacements_of_phonopy(crystal, tmp_path):
    """phonopy on the same force constants and mesh through Γ, without the
    modes below 1e-3 THz."""
    from phonopy.file_IO import parse_FORCE_CONSTANTS

    cell, matrix, result = crystal
    path = tmp_path / "FORCE_CONSTANTS"
    ForceConstants(cell, matrix, result.force_constants).write(path)
    theirs = of_phonopy(cell, matrix, "P")
    theirs.force_constants = parse_FORCE_CONSTANTS(path, theirs.primitive.p2s_map)
    theirs.run_mesh(
        [6, 6, 6], is_gamma_center=True, is_mesh_symmetry=False, with_eigenvectors=True
    )
    temperatures = [0.0, 100.0, 300.0, 900.0]
    theirs.run_thermal_displacement_matrices(temperatures=temperatures, freq_min=1e-3)
    expected = theirs.thermal_displacement_matrices.thermal_displacement_matrices
    found = result.thermal_displacements(temperatures, mesh=6)
    # phonopy's physical constants are older, by a few parts in 10^7
    np.testing.assert_allclose(
        found, expected, rtol=0, atol=1e-5 * np.abs(expected).max()
    )


def test_gruneisen_of_phonopy(crystal, potential, tmp_path):
    """phonopy's mode Grüneisen parameters from the same force constants at
    three volumes."""
    from phonopy.file_IO import parse_FORCE_CONSTANTS
    from phonopy.gruneisen.core import GruneisenBase

    cell, matrix, result = crystal

    def scaled(volume):
        s = volume ** (1 / 3)
        return cell.with_box(cell.box.matrix * s).with_columns(
            positions=cell["positions"] * s
        )

    cells = [scaled(0.99), cell, scaled(1.01)]
    results = [
        phonons(c, potential, supercell=matrix, displacement=DISPLACEMENT)
        for c in (cells[0], cells[2])
    ]
    results.insert(1, result)
    q = np.vstack(
        [
            np.random.default_rng(0).random((8, 3)) - 0.5,
            [[0.5, 0, 0], [0.5, 0.5, 0], [0.5, 0.5, 0.5], [0.25, 0, 0], [0, 0, 0.25]],
        ]
    )
    found = result.gruneisen(q, smaller=results[0], larger=results[2])
    matrices = []
    for k, (c, r) in enumerate(zip(cells, results)):
        path = tmp_path / f"FORCE_CONSTANTS_{k}"
        ForceConstants(c, matrix, r.force_constants).write(path)
        theirs = of_phonopy(c, matrix, "P")
        theirs.force_constants = parse_FORCE_CONSTANTS(path, theirs.primitive.p2s_map)
        matrices.append(theirs.dynamical_matrix)
    expected = GruneisenBase(
        matrices[1], matrices[2], matrices[0], qpoints=q
    ).get_gruneisen()
    np.testing.assert_allclose(found, expected, rtol=0, atol=1e-10)
