# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Force constants fitted within a cutoff: springs between near neighbors,
whose force constants are known in closed form, come back exactly, in
supercells that keep the symmetry of the crystal and in those that break
it; the frequencies keep the symmetry of the crystal; a shell at the cutoff
is whole. symfc with the same cutoff is in test_force_constants_symfc.py."""

import numpy as np
import pytest

import mdapy
from mdapy.build import build_crystal
from mdapy.calculators import EAM
from mdapy.errors import InvalidParameter
from mdapy.lattice_dynamics import (
    displaced_supercells,
    force_constants,
    primitive_cell,
)
from mdapy.lattice_dynamics._force_constants import _supercell_matrix
from mdapy.lattice_dynamics._phonons import _THZ, _masses, _supercell
from mdapy.simulation import minimize

from calculators.eam_files import write_setfl

# the fits use the symmetry of the crystals, found by spglib, which the
# development install has
pytest.importorskip("spglib")

K, R0 = 0.9, 2.4  # eV/Å² and Å: springs in tension


def springs(cell, matrix, reach):
    """(n, nS, 3, 3) the force constants of springs of stiffness K and rest
    length R0 between atoms closer than reach: along a bond of length r the
    second derivative K, across it the first over r, K (r - R0) / r."""
    big = _supercell(cell, matrix)
    n = len(cell)
    vectors = big.box.minimum_image(
        (big["positions"][None, :, :] - cell["positions"][:, None, :]).reshape(-1, 3)
    ).reshape(n, len(big), 3)
    lengths = np.linalg.norm(vectors, axis=2)
    bonded = (lengths > 1e-9) & (lengths < reach)
    unit = np.where(
        bonded[..., None], vectors / np.where(bonded, lengths, 1)[..., None], 0
    )
    along = np.einsum("ija,ijb->ijab", unit, unit)
    across = np.where(bonded, K * (lengths - R0) / np.where(bonded, lengths, 1), 0)
    constants = -(K * along + across[..., None, None] * (np.eye(3) - along))
    constants *= bonded[..., None, None]
    for a in range(n):
        constants[a, a] = -constants[a].sum(axis=0)
    return constants


def with_forces(frames, cell, matrix, constants):
    """The frames with the forces of the force constants, exactly harmonic."""
    big = _supercell(cell, matrix)
    n, size = len(cell), len(big) // len(cell)
    out = []
    for frame in frames:
        u = big.box.minimum_image(frame["positions"] - big["positions"])
        forces = np.zeros_like(u)
        for c in range(size):
            # the atoms of cell c see the supercell moved by c
            moved = _supercell(cell, matrix)
            shift = moved["positions"][c * n] - moved["positions"][0]
            sites = big.box.wrap(big["positions"] + shift)[0]
            order = np.argmin(
                np.linalg.norm(
                    big.box.minimum_image(
                        (sites[:, None, :] - big["positions"][None, :, :]).reshape(
                            -1, 3
                        )
                    ).reshape(len(big), len(big), 3),
                    axis=2,
                ),
                axis=1,
            )
            forces[c * n : (c + 1) * n] = -np.einsum("ijab,jb->ia", constants, u[order])
        out.append(frame.with_columns(forces=forces))
    return out


def frequencies_of_pairs(constants, cell, matrix, q):
    """THz from a sum over the pair vectors of the force constants, without
    periodic images."""
    big = _supercell(cell, matrix)
    n = len(cell)
    vectors = big.box.minimum_image(
        (big["positions"][None, :, :] - cell["positions"][:, None, :]).reshape(-1, 3)
    ).reshape(n, len(big), 3)
    reciprocal = 2 * np.pi * np.linalg.inv(cell.box.matrix).T
    phase = np.exp(1j * np.einsum("qx,ijx->qij", q @ reciprocal, vectors))
    blocks = np.zeros((len(q), n, n, 3, 3), dtype=complex)
    for j in range(len(big)):
        blocks[:, :, j % n] += phase[:, :, j, None, None] * constants[:, j]
    masses = np.sqrt(np.outer(_masses(cell), _masses(cell)))
    blocks /= masses[None, :, :, None, None]
    matrices = blocks.transpose(0, 1, 3, 2, 4).reshape(len(q), 3 * n, 3 * n)
    values = np.linalg.eigvalsh(0.5 * (matrices + matrices.conj().transpose(0, 2, 1)))
    return np.sign(values) * np.sqrt(np.abs(values)) * _THZ


CASES = {
    "fcc 4x4x4": ("fcc", 3.6, 4, True, 3.0),
    "fcc 4x4x3, of less symmetry": ("fcc", 3.6, (4, 4, 3), True, 3.0),
    "fcc, skewed": ("fcc", 3.6, [[4, 1, 0], [0, 4, 1], [1, 0, 4]], True, 3.0),
    "hcp 4x4x3": ("hcp", 2.55, (4, 4, 3), False, 2.8),
    "diamond 3x3x3": ("diamond", 5.43, 3, True, 2.5),
}


@pytest.mark.parametrize("symmetry", [True, False])
@pytest.mark.parametrize("name", CASES)
def test_springs_come_back_exactly(name, symmetry):
    structure, a, supercell, primitive, reach = CASES[name]
    cell = build_crystal(structure, a, elements="Cu")
    if primitive:
        cell = primitive_cell(cell)
    matrix = _supercell_matrix(supercell)
    constants = springs(cell, matrix, reach)
    frames = displaced_supercells(
        cell,
        supercell=supercell,
        method="random",
        count=1 if symmetry else 3 * len(cell),
        seed=4,
    )
    frames = with_forces(frames, cell, matrix, constants)
    fit = force_constants(cell, frames, cutoff=reach, symmetry=symmetry)
    assert fit.cutoff == reach
    assert fit.error < 1e-12
    np.testing.assert_allclose(fit.second, constants, rtol=0, atol=1e-12)
    # the phonons put each pair on its shortest image, the pair itself
    q = np.random.default_rng(1).random((10, 3)) - 0.5
    np.testing.assert_allclose(
        fit.phonons().frequencies(q),
        frequencies_of_pairs(fit.second, cell, matrix, q),
        rtol=0,
        atol=1e-10,
    )


@pytest.fixture(scope="module")
def copper(tmp_path_factory):
    potential = EAM(write_setfl(tmp_path_factory.mktemp("eam") / "CuNi.eam.alloy"))
    cell = minimize(
        build_crystal("fcc", 3.29, elements="Cu"),
        potential,
        cell="full",
        max_force=1e-10,
        max_stress=1e-8,
        max_steps=10000,
    ).system
    return potential, primitive_cell(cell)


def computed(frames, potential):
    return [
        f.with_columns(forces=potential.compute(f, ("forces",)).forces) for f in frames
    ]


def test_a_supercell_of_less_symmetry_keeps_that_of_the_crystal(copper):
    """In 4x4x3 cells of fcc, with every operation of the crystal: the same
    free force constants as in 4x4x4, and frequencies that the rotations of
    the crystal do not change."""
    potential, cell = copper
    frames = computed(
        displaced_supercells(cell, supercell=(4, 4, 3), method="random", seed=2),
        potential,
    )
    fit = force_constants(cell, frames, cutoff=2.8)
    assert fit.parameters == 3  # the springs along and across, and one more
    # the rotations of the crystal on wave vectors of the cell
    from mdapy.analysis._symmetry import _symmetry

    rotations = _symmetry(cell, True).cartesian
    reciprocal = 2 * np.pi * np.linalg.inv(cell.box.matrix).T
    q = np.random.default_rng(3).random((5, 3)) - 0.5
    reference = fit.phonons().frequencies(q)
    for rotation in rotations:
        turned = (q @ reciprocal @ rotation.T) @ np.linalg.inv(reciprocal)
        np.testing.assert_allclose(
            fit.phonons().frequencies(turned), reference, rtol=0, atol=1e-10
        )


def test_the_error_falls_as_the_cutoff_grows(copper):
    """The force constants of EAM reach twice its cutoff: the longer the
    cutoff of the fit, the smaller its error."""
    potential, cell = copper
    frames = computed(
        displaced_supercells(cell, supercell=5, method="random", count=2, seed=5),
        potential,
    )
    errors = [force_constants(cell, frames, cutoff=c).error for c in (2.8, 4.8, 5.5)]
    torus = force_constants(cell, frames)
    assert errors[0] > errors[1] > errors[2] > torus.error
    assert torus.cutoff is None


def test_a_shell_at_the_cutoff_is_whole():
    """A cutoff typed at the distance of the nearest neighbors holds all
    twelve, with the atom itself."""
    cell = primitive_cell(build_crystal("fcc", 3.6, elements="Cu"))
    reach = 3.6 / np.sqrt(2)
    constants = springs(cell, _supercell_matrix(4), reach + 1e-3)
    frames = with_forces(
        displaced_supercells(cell, supercell=4, method="random", seed=6),
        cell,
        _supercell_matrix(4),
        constants,
    )
    fit = force_constants(cell, frames, cutoff=reach)
    assert (np.abs(fit.second[0]).sum(axis=(1, 2)) > 1e-9).sum() == 13
    np.testing.assert_allclose(fit.second, constants, rtol=0, atol=1e-12)


def test_a_sheet():
    """A triangular sheet with springs between nearest neighbors."""
    a, height = 2.5, 10.0
    plane = np.array([[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, height]])
    sheet = mdapy.System(
        np.array([[0, 0, height / 2]]),
        mdapy.Box(plane, pbc=(True, True, False)),
        elements=["C"],
    )
    matrix = _supercell_matrix((6, 6, 1))
    constants = springs(sheet, matrix, 2.6)
    frames = with_forces(
        displaced_supercells(sheet, supercell=6, method="random", seed=7),
        sheet,
        matrix,
        constants,
    )
    fit = force_constants(sheet, frames, cutoff=2.6)
    np.testing.assert_allclose(fit.second, constants, rtol=0, atol=1e-12)
    with pytest.raises(InvalidParameter, match="at most 7.499"):
        force_constants(sheet, frames, cutoff=8.0)


def test_errors():
    cell = primitive_cell(build_crystal("fcc", 3.6, elements="Cu"))
    frames = [
        f.with_columns(forces=np.zeros((len(f), 3)))
        for f in displaced_supercells(cell, supercell=4, method="random", seed=0)
    ]
    # the shortest vector of the supercell: four nearest neighbors
    limit = 2 * 3.6 / np.sqrt(2) - 1e-3
    with pytest.raises(InvalidParameter, match=f"at most {limit:.4g}"):
        force_constants(cell, frames, cutoff=5.2)
    for cutoff in (0.0, -1.0):
        with pytest.raises(InvalidParameter, match="cutoff must be positive"):
            force_constants(cell, frames, cutoff=cutoff)
