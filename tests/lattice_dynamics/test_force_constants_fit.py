# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Force constants fitted to the forces of displaced supercells: forces that
are exactly harmonic give back their force constants, finite displacements
give those of phonons, and the frames may come in any order. symfc on the
same data is in test_force_constants_symfc.py."""

import numpy as np
import pytest

import mdapy
from mdapy.build import build_crystal
from mdapy.calculators import EAM
from mdapy.errors import InvalidParameter
from mdapy.lattice_dynamics import (
    ForceConstants,
    displaced_supercells,
    force_constants,
    phonons,
    primitive_cell,
)
from mdapy.lattice_dynamics._force_constants import _cell_of
from mdapy.lattice_dynamics._phonons import _lattice_points, _supercell
from mdapy.simulation import minimize

from calculators.eam_files import write_setfl
from calculators.test_testing import Harmonic

# most fits use the symmetry of the crystals, found by spglib, which the
# development install has
pytest.importorskip("spglib")

# the crystal, its supercell, the free parameters that symfc counts, the
# supercells of finite displacements, as the page of phonons says, and
# whether the operations turn axes onto axes only
CRYSTALS = {
    "fcc primitive cell": ("fcc", 3.29, "Cu", 4, True, 17, 1, True),
    "fcc cubic cell": ("fcc", 3.29, "Cu", 2, False, 11, 1, True),
    "bcc": ("bcc", 2.7, "Cu", 3, False, 15, 1, True),
    "hcp": ("hcp", 2.33, "Cu", (3, 3, 2), False, 26, 2, False),
    "L1_2 Cu3Ni": ("l1_2", 3.29, ("Cu", "Ni"), 2, False, 31, 3, False),
}


@pytest.fixture(scope="module")
def potential(tmp_path_factory):
    return EAM(write_setfl(tmp_path_factory.mktemp("eam") / "CuNi.eam.alloy"))


@pytest.fixture(scope="module", params=CRYSTALS)
def crystal(request, potential):
    """A relaxed cell, its supercell, the phonons of the potential and the
    count of free parameters."""
    structure, a, elements, supercell, primitive, *expected = CRYSTALS[request.param]
    cell = minimize(
        build_crystal(structure, a, elements=elements),
        potential,
        cell="full",
        max_force=1e-10,
        max_stress=1e-8,
        max_steps=10000,
    ).system
    if primitive:
        cell = primitive_cell(cell)
    return cell, supercell, phonons(cell, potential, supercell=supercell), *expected


def harmonic(frames, cell, matrix, constants):
    """The frames with the forces of these force constants: minus the force
    constants of each pair times the displacement of the second atom."""
    n = len(cell)
    points = _lattice_points(matrix)
    size = len(points)
    big = _supercell(cell, matrix)
    apart = _cell_of(points[None, :, :] - points[:, None, :], matrix)
    full = np.empty((n * size, n * size, 3, 3))
    for c in range(size):
        columns = (apart[c][:, None] * n + np.arange(n)).reshape(-1)
        full[c * n : (c + 1) * n] = constants[:, columns]
    out = []
    for frame in frames:
        u = big.box.minimum_image(frame["positions"] - big["positions"])
        out.append(frame.with_columns(forces=-np.einsum("ijab,jb->ia", full, u)))
    return out


def computed(frames, calculator):
    return [
        f.with_columns(forces=calculator.compute(f, ("forces",)).forces) for f in frames
    ]


def test_harmonic_forces_give_their_force_constants(crystal):
    """The force constants of phonons, symmetric and summing to zero, and
    forces exactly of them: the fit of random displacements gives them back."""
    cell, supercell, result, count, _, _ = crystal
    frames = displaced_supercells(
        cell, supercell=supercell, method="random", displacement=0.03, seed=1
    )
    frames = harmonic(frames, cell, result.supercell, result.force_constants)
    fit = force_constants(cell, frames)
    assert fit.parameters == count
    assert fit.error < 1e-12
    np.testing.assert_array_equal(fit.supercell, result.supercell)
    np.testing.assert_allclose(fit.second, result.force_constants, rtol=0, atol=1e-12)
    # the same without symmetry, from more frames: at Γ a pattern gives 3n
    # of the (3n)(3n + 1)/2 constants of the dynamical matrix, so 3n are
    # needed
    frames = displaced_supercells(
        cell,
        supercell=supercell,
        method="random",
        displacement=0.03,
        count=3 * len(cell),
        seed=2,
    )
    frames = harmonic(frames, cell, result.supercell, result.force_constants)
    plain = force_constants(cell, frames, symmetry=False)
    assert plain.parameters > count
    np.testing.assert_allclose(plain.second, result.force_constants, rtol=0, atol=1e-11)


def test_finite_displacements_give_the_phonons(crystal, potential):
    """The moves of phonons, the forces of the potential: the same force
    constants where the operations turn axes onto axes, as for fcc and bcc,
    within the error of the differences otherwise."""
    cell, supercell, result, _, _, onto_axes = crystal
    frames = computed(displaced_supercells(cell, supercell=supercell), potential)
    fit = force_constants(cell, frames)
    q = np.random.default_rng(0).random((20, 3)) - 0.5
    apart = np.abs(fit.phonons().frequencies(q) - result.frequencies(q)).max()
    if onto_axes:
        assert apart < 1e-11
    else:
        assert apart < 3e-4
        # the differences are of the square of the displacement
        small = computed(
            displaced_supercells(cell, supercell=supercell, displacement=1e-3),
            potential,
        )
        closer = force_constants(cell, small).phonons().frequencies(q)
        reference = phonons(cell, potential, supercell=supercell, displacement=1e-3)
        assert np.abs(closer - reference.frequencies(q)).max() < apart / 50


def test_the_moves_are_those_of_phonons(crystal):
    """One atom moves in each supercell, as many supercells as phonons
    computes; without symmetry each atom of the cell along x, y and z by plus
    and minus."""
    cell, supercell, result, _, moves, _ = crystal
    frames = displaced_supercells(cell, supercell=supercell, displacement=0.02)
    assert len(frames) == moves
    big = _supercell(cell, result.supercell)
    for frame in frames:
        moved = np.linalg.norm(frame["positions"] - big["positions"], axis=1)
        assert (moved > 1e-12).sum() == 1
        assert moved.max() == pytest.approx(0.02)
    assert len(displaced_supercells(cell, supercell=supercell, symmetry=False)) == (
        6 * len(cell)
    )


def test_random_displacements():
    cell = build_crystal("fcc", 3.6, elements="Ar")
    frames = displaced_supercells(
        cell, supercell=2, method="random", displacement=0.05, count=3, seed=7
    )
    assert len(frames) == 6
    big = _supercell(cell, np.diag([2, 2, 2]))
    moves = [f["positions"] - big["positions"] for f in frames]
    for plus, minus in zip(moves[::2], moves[1::2]):
        np.testing.assert_allclose(plus, -minus)
        np.testing.assert_allclose(np.linalg.norm(plus, axis=1), 0.05)
    assert not np.allclose(moves[0], moves[2])
    again = displaced_supercells(
        cell, supercell=2, method="random", displacement=0.05, count=3, seed=7
    )
    np.testing.assert_array_equal(again[4]["positions"], frames[4]["positions"])
    assert frames[0].table.columns == big.table.columns


def test_the_atoms_of_the_frames_may_be_in_any_order(tmp_path):
    """Frames read back from POSCAR, which sorts the atoms by element and
    wraps them into the box, give the same force constants."""
    cell = build_crystal("l1_2", 3.6, elements=("Cu", "Ni"))
    springs = Harmonic(k=0.7, r0=3.6 / np.sqrt(2))
    frames = computed(
        displaced_supercells(cell, supercell=2, method="random", seed=3), springs
    )
    direct = force_constants(cell, frames)
    shuffled = []
    for k, frame in enumerate(frames):
        order = np.random.default_rng(k).permutation(len(frame))
        path = tmp_path / f"POSCAR-{k}"
        mdapy.io.write(path, frame.gather(order))
        read = mdapy.io.read(path)
        elements = read.table["element"].cast(str).to_list()
        assert elements == sorted(elements, key=elements.index)  # in groups
        forces = springs.compute(read, ("forces",)).forces
        shuffled.append(read.with_columns(forces=forces))
    again = force_constants(cell, shuffled)
    np.testing.assert_allclose(again.second, direct.second, rtol=0, atol=1e-8)
    # boxes with another origin, the atoms where they are in the box
    shift = np.array([1.0, -2.0, 0.5])
    elsewhere = [
        f.with_box(mdapy.Box(f.box.matrix, origin=shift)).with_columns(
            positions=f["positions"] + shift
        )
        for f in frames
    ]
    np.testing.assert_allclose(
        force_constants(cell, elsewhere).second, direct.second, rtol=0, atol=1e-12
    )


def test_a_sheet():
    """A triangular sheet with springs: the fit of finite displacements,
    symmetric under the operations of the sheet, and exactly harmonic forces
    of it, in its plane and across it."""
    a, height = 2.5, 10.0
    plane = np.array([[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, height]])
    sheet = mdapy.System(
        np.array([[0, 0, height / 2]]),
        mdapy.Box(plane, pbc=(True, True, False)),
        elements=["C"],
    )
    springs = Harmonic(k=2.0, r0=2.4)
    exact = force_constants(
        sheet, computed(displaced_supercells(sheet, supercell=4), springs)
    )
    np.testing.assert_array_equal(exact.supercell, np.diag([4, 4, 1]))
    frames = displaced_supercells(sheet, supercell=4, method="random", seed=5)
    frames = harmonic(frames, sheet, exact.supercell, exact.second)
    fit = force_constants(sheet, frames)
    assert fit.parameters == exact.parameters
    np.testing.assert_allclose(fit.second, exact.second, rtol=0, atol=1e-12)
    # and the phonons of the sheet, those of phonons to the error of the
    # differences
    result = phonons(sheet, springs, supercell=4)
    q = [[0.1, 0.2, 0.0], [0.5, 0.0, 0.0], [1 / 3, 1 / 3, 0.0]]
    np.testing.assert_allclose(
        fit.phonons().frequencies(q), result.frequencies(q), atol=1e-4
    )


def test_force_constants_of_no_fit_have_no_error():
    cell = build_crystal("sc", 2.5, elements="Ar")
    constants = ForceConstants(cell, 1, np.zeros((1, 1, 3, 3)))
    assert constants.parameters is None and constants.error is None


def test_errors(potential):
    cell = build_crystal("fcc", 3.6, elements="Ar")
    springs = Harmonic(k=0.7, r0=3.6 / np.sqrt(2))
    frames = computed(displaced_supercells(cell, supercell=2), springs)
    with pytest.raises(InvalidParameter, match="determine 93 of the 591 free"):
        force_constants(cell, frames, symmetry=False)
    with pytest.raises(InvalidParameter, match="need frames"):
        force_constants(cell, [])
    with pytest.raises(InvalidParameter, match="with atoms"):
        force_constants(cell.filter(np.zeros(len(cell), dtype=bool)), frames)
    with pytest.raises(InvalidParameter, match="not a supercell"):
        force_constants(
            primitive_cell(cell), [frames[0].with_box(cell.box.matrix * 1.5)]
        )
    other = frames[0].with_box(frames[0].box.matrix * 1.01)
    with pytest.raises(InvalidParameter, match="box of frame 1"):
        force_constants(cell, [frames[0], frames[0].with_box(other.box)])
    with pytest.raises(InvalidParameter, match="with forces"):
        force_constants(cell, [frames[0].select(["positions", "element"])])
    with pytest.raises(InvalidParameter, match="with forces"):
        force_constants(cell, [frames[0].filter(np.arange(len(frames[0])) > 0)])
    positions = frames[0]["positions"].copy()
    positions[1] = positions[0]
    with pytest.raises(InvalidParameter, match="one at each atom"):
        force_constants(cell, [frames[0].with_columns(positions=positions)])
    swapped = frames[0].with_columns(element=["Kr"] + ["Ar"] * (len(frames[0]) - 1))
    with pytest.raises(InvalidParameter, match="element"):
        force_constants(cell, [swapped])
    with pytest.raises(InvalidParameter, match="method"):
        displaced_supercells(cell, supercell=2, method="smart")
    with pytest.raises(InvalidParameter, match="displacement"):
        displaced_supercells(cell, supercell=2, displacement=0.0)
    for count in (0, 1.5, True):
        with pytest.raises(InvalidParameter, match="count"):
            displaced_supercells(cell, supercell=2, method="random", count=count)
    # a cell of 32 atoms without symmetry, in a supercell of 27 cells
    big = mdapy.System(
        np.random.default_rng(0).random((32, 3)) * 8,
        mdapy.Box(np.eye(3) * 8),
        elements=["Ar"] * 32,
    )
    moved = displaced_supercells(big, supercell=3, method="random", seed=0)
    with pytest.raises(InvalidParameter, match="GB"):
        force_constants(
            big, [m.with_columns(forces=np.zeros((len(m), 3))) for m in moved]
        )
