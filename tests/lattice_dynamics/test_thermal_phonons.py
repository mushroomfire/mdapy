# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Phonons from displaced atoms and their forces: small displacements give the
force constants of the displacement method; molecular dynamics is skipped where
LAMMPS is not installed."""

import numpy as np
import pytest

import mdapy
from mdapy import units
from mdapy.build import build_crystal
from mdapy.calculators import EAM, LAMMPS
from mdapy.core.threads import CancelToken
from mdapy.elements import atomic_masses
from mdapy.errors import Cancelled, InvalidParameter, MissingProperty
from mdapy.lattice_dynamics import (
    phonons,
    phonons_from_frames,
    primitive_cell,
    thermal_phonons,
)
from mdapy.lattice_dynamics._phonons import _supercell
from mdapy.lattice_dynamics._thermal_phonons import _cell_table
from mdapy.simulation import minimize

from calculators.eam_files import write_setfl

Q = [[0.5, 0.5, 0.5], [0.5, 0, 0.5], [0.25, 0, 0.25], [0.1, 0.2, 0.3]]


@pytest.fixture(scope="module")
def setfl(tmp_path_factory):
    return write_setfl(tmp_path_factory.mktemp("eam") / "CuNi.eam.alloy")


@pytest.fixture(scope="module")
def potential(setfl):
    return EAM(setfl)


def relaxed(potential, structure, a=3.29, **options):
    start = build_crystal(structure, a, elements="Cu", **options)
    return minimize(
        start, potential, cell="full", max_force=1e-10, max_stress=1e-8, max_steps=10000
    ).system


def shaken(cell, potential, counts, size, frames, seed=1):
    """Frames of the supercell with atoms displaced at random and their forces."""
    big = _supercell(cell, np.diag(counts))
    rng = np.random.default_rng(seed)
    result = []
    for _ in range(frames):
        moved = big.with_columns(
            positions=big["positions"] + rng.normal(0, size, (len(big), 3))
        )
        result.append(
            moved.with_columns(forces=potential.compute(moved, ("forces",)).forces)
        )
    return result


def test_the_cell_table():
    matrix = np.array([[2, 0, 0], [0, 2, 0], [1, 1, 2]])
    table = _cell_table(matrix)
    size = round(np.linalg.det(matrix))
    assert table.shape == (size, size)
    assert (table[0] == np.arange(size)).all()  # adding nothing
    assert (np.sort(table, axis=1) == np.arange(size)).all()  # each a permutation
    assert (table == table.T).all()  # the sum does not depend on the order


@pytest.mark.parametrize(
    "structure, primitive, counts",
    [("fcc", True, (4, 4, 4)), ("hcp", False, (3, 3, 2))],
)
def test_small_displacements_give_the_force_constants(
    potential, structure, primitive, counts
):
    cell = relaxed(potential, structure)
    if primitive:
        cell = primitive_cell(cell)
    reference = phonons(cell, potential, supercell=counts, symmetry=False)
    frames = shaken(cell, potential, counts, 0.003, 40)
    fitted = phonons_from_frames(cell, frames, supercell=counts, symmetry=False)
    # within the cubic terms of displacements of this size
    np.testing.assert_allclose(
        fitted.frequencies(Q), reference.frequencies(Q), atol=0.03
    )
    # the crystal can translate, and the force constants are symmetric
    np.testing.assert_allclose(fitted.frequencies([0, 0, 0])[:3], 0.0, atol=1e-3)
    np.testing.assert_allclose(fitted.force_constants.sum(axis=1), 0.0, atol=1e-9)


def test_symmetry_removes_the_noise_of_large_displacements(potential):
    cell = primitive_cell(relaxed(potential, "fcc"))
    frames = shaken(cell, potential, (4, 4, 4), 0.1, 6)
    plain = phonons_from_frames(cell, frames, supercell=4, symmetry=False)
    known = phonons_from_frames(cell, frames, supercell=4, symmetry=True)
    # the two transverse modes at L are the same by symmetry
    low = known.frequencies([0.5, 0.5, 0.5])
    assert low[0] == pytest.approx(low[1], abs=1e-6)
    noisy = plain.frequencies([0.5, 0.5, 0.5])
    assert abs(noisy[1] - noisy[0]) > abs(low[1] - low[0])


def test_a_supercell_matrix(potential):
    cell = primitive_cell(relaxed(potential, "fcc"))
    matrix = np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]])
    frames = shaken(cell, potential, (4, 4, 4), 0.003, 6)
    by_matrix = phonons_from_frames(cell, frames, supercell=matrix, symmetry=False)
    by_number = phonons_from_frames(cell, frames, supercell=4, symmetry=False)
    np.testing.assert_allclose(by_matrix.force_constants, by_number.force_constants)


def test_invalid_frames(potential):
    cell = primitive_cell(relaxed(potential, "fcc"))
    frames = shaken(cell, potential, (3, 3, 3), 0.01, 3)
    with pytest.raises(InvalidParameter, match="too few"):
        phonons_from_frames(cell, frames[:2], supercell=3, symmetry=False)
    with pytest.raises(InvalidParameter, match="atoms"):
        phonons_from_frames(cell, frames, supercell=2, symmetry=False)
    with pytest.raises(MissingProperty, match="forces"):
        phonons_from_frames(
            cell, [f.drop("forces") for f in frames], supercell=3, symmetry=False
        )
    open_box = cell.with_box(mdapy.Box(cell.box.matrix, pbc=(True, True, False)))
    with pytest.raises(InvalidParameter, match="periodic"):
        phonons_from_frames(open_box, frames, supercell=3)


def test_against_symfc(potential):
    """The same frames in phonopy, whose fit with symfc is a mature version of this."""
    phonopy = pytest.importorskip("phonopy")
    pytest.importorskip("symfc")
    from phonopy.structure.atoms import PhonopyAtoms

    cell = relaxed(potential, "fcc")
    counts = (3, 3, 3)
    frames = shaken(cell, potential, counts, 0.15, 240)
    ours = phonons_from_frames(cell, frames, supercell=np.diag(counts))
    names = cell.table["element"].cast(str).to_list()
    unit = PhonopyAtoms(
        symbols=names,
        cell=cell.box.matrix,
        scaled_positions=cell.box.to_fractional(cell["positions"]),
        masses=atomic_masses(names),
    )
    theirs = phonopy.Phonopy(
        unit, supercell_matrix=np.diag(counts), primitive_matrix="P"
    )
    big = _supercell(cell, np.diag(counts))
    apart = (
        big.box.to_fractional(big["positions"])[:, None]
        - theirs.supercell.scaled_positions
    )
    apart -= np.round(apart)
    order = np.argmin(np.abs(apart).sum(axis=2), axis=0)
    theirs.dataset = {
        "displacements": np.array(
            [(f["positions"] - big["positions"])[order] for f in frames]
        ),
        "forces": np.array([f["forces"][order] for f in frames]),
    }
    theirs.produce_force_constants(fc_calculator="symfc")
    q = np.random.default_rng(0).random((20, 3)) - 0.5
    theirs.run_qpoints(q)
    np.testing.assert_allclose(
        ours.frequencies(q), theirs.qpoints.frequencies, atol=0.05
    )


lammps = pytest.importorskip("lammps")


@pytest.fixture(scope="module")
def dynamics(setfl):
    return LAMMPS(f"pair_style eam/alloy\npair_coeff * * {setfl} Cu Ni", ["Cu", "Ni"])


FAST = dict(
    supercell=4, timestep=0.002, equilibration=2.0, production=20.0, num_frames=40
)


def test_at_a_low_temperature_as_at_rest(dynamics, potential):
    cell = primitive_cell(relaxed(potential, "fcc"))
    cold = thermal_phonons(cell, dynamics, 5.0, pressure=None, **FAST)
    rest = phonons(cell, potential, supercell=4)
    np.testing.assert_allclose(cold.frequencies(Q), rest.frequencies(Q), rtol=0.03)
    assert cold.cell.box.allclose(cell.box)


def test_a_hot_crystal_is_softer_and_expands(dynamics, potential):
    cell = primitive_cell(relaxed(potential, "fcc"))
    hot = thermal_phonons(cell, dynamics, 600.0, **FAST)
    rest = phonons(cell, potential, supercell=4)
    assert hot.cell.box.volume > cell.box.volume
    ratio = hot.frequencies(Q)[:, -1] / rest.frequencies(Q)[:, -1]
    assert (ratio < 1).all() and (ratio > 0.7).all()
    # with the symmetry of the crystal, whatever the box did on average
    low = hot.frequencies([0.5, 0.5, 0.5])
    assert low[0] == pytest.approx(low[1], abs=1e-6)


def test_the_displacements_of_the_dynamics(dynamics, potential):
    """Classical harmonic vibrations of the fitted force constants give about
    the mean square displacement of the dynamics they were fitted to."""
    cell = primitive_cell(relaxed(potential, "fcc"))
    temperature = 300.0
    result = thermal_phonons(cell, dynamics, temperature, pressure=None, **FAST)
    big = _supercell(cell, result.supercell)
    from mdapy.simulation import MolecularDynamics

    with MolecularDynamics(big, dynamics, timestep=0.002) as md:
        md.velocities(2 * temperature, seed=3)
        md.command(f"fix t all langevin {temperature} {temperature} 0.1 4 zero yes")
        md.run(1000, ensemble="nve", thermo=1000)
        md.run(5000, ensemble="nve", thermo=5000, save_every=50)
        frames = md.frames
    squares = []
    for frame in frames:
        fractional = big.box.to_fractional(frame["positions"]) - big.box.to_fractional(
            big["positions"]
        )
        shifts = (fractional - np.round(fractional)) @ big.box.matrix
        squares.append(((shifts - shifts.mean(axis=0)) ** 2).sum(axis=1).mean())
    grid = (np.arange(4) + 0.5) / 4  # the wave vectors of a box with no drift
    q = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), -1).reshape(-1, 3)
    omega = 2 * np.pi * result.frequencies(q)
    mass = float(mdapy.elements.atomic_masses(["Cu"])[0])
    expected = (
        units.BOLTZMANN
        * temperature
        * 3
        * np.mean(1 / (mass * units.AMU_A2_PER_PS2_TO_EV * omega**2))
    )
    assert np.mean(squares) == pytest.approx(expected, rel=0.2)


def test_progress_and_cancel(dynamics, potential):
    cell = primitive_cell(relaxed(potential, "fcc"))
    seen = []
    options = dict(FAST, equilibration=0.1, production=0.2, num_frames=4)
    thermal_phonons(
        cell, dynamics, 100.0, progress=lambda d, t: seen.append((d, t)), **options
    )
    assert seen[-1][0] == seen[-1][1]
    token = CancelToken()
    token.cancel()
    with pytest.raises(Cancelled):
        thermal_phonons(cell, dynamics, 100.0, cancel=token, **options)


def test_the_default_supercell_is_thick_enough(dynamics, potential):
    cell = primitive_cell(relaxed(potential, "fcc"))
    result = thermal_phonons(
        cell,
        dynamics,
        50.0,
        pressure=None,
        timestep=0.002,
        equilibration=0.1,
        production=0.4,
        num_frames=8,
    )
    assert np.linalg.det(result.supercell) > 1


def test_invalid_input(dynamics, potential):
    cell = primitive_cell(relaxed(potential, "fcc"))
    for wrong, match in (
        (dict(timestep=0.0), "positive"),
        (dict(equilibration=0.0), "positive"),
        (dict(production=-1.0), "positive"),
        (dict(num_frames=0), "positive"),
        (dict(seed=0), "positive"),
    ):
        with pytest.raises(InvalidParameter, match=match):
            thermal_phonons(cell, dynamics, 100.0, **wrong)
    with pytest.raises(InvalidParameter, match="temperature"):
        thermal_phonons(cell, dynamics, 0.0)
    with pytest.raises(InvalidParameter, match="LAMMPS"):
        thermal_phonons(cell, potential, 100.0)
    with pytest.raises(MissingProperty, match="element"):
        thermal_phonons(cell.drop("element"), dynamics, 100.0)
    with pytest.raises(InvalidParameter, match="periodic"):
        thermal_phonons(
            cell.with_box(mdapy.Box(cell.box.matrix, pbc=(True, True, False))),
            dynamics,
            100.0,
        )
