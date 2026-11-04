# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Elastic constants and phonons of sheets: a network of springs, whose
results are known in closed form, and the same sheet in a periodic box."""

import numpy as np
import pytest

import mdapy
from mdapy import units
from mdapy.errors import InvalidParameter
from mdapy.lattice_dynamics import (
    SheetElasticResult,
    elastic_constants,
    phonons,
    primitive_cell,
    quasi_harmonic,
)
from mdapy.simulation import minimize

from calculators.test_testing import Harmonic

A, HEIGHT, K, MASS = 2.5, 10.0, 2.0, 12.011
PLANE = np.array([[A, 0, 0], [-A / 2, A * np.sqrt(3) / 2, 0], [0, 0, HEIGHT]])
SHEET = (True, True, False)


def triangular(height=HEIGHT, repeat=(1, 1), pbc=SHEET):
    """A triangular sheet of one atom per cell, with springs at rest."""
    matrix = PLANE.copy()
    matrix[2, 2] = height
    positions = [
        i * PLANE[0] + j * PLANE[1] + [0, 0, height / 2]
        for i in range(repeat[0])
        for j in range(repeat[1])
    ]
    matrix[0] *= repeat[0]
    matrix[1] *= repeat[1]
    return mdapy.System(
        np.array(positions),
        mdapy.Box(matrix, pbc=pbc),
        elements=["C"] * len(positions),
    )


@pytest.fixture
def springs():
    return Harmonic(k=K, r0=A)


def test_the_constants_of_a_network_of_springs(springs):
    """Nearest neighbors on a triangular lattice: Lamé's constants are both
    sqrt(3) k / 4, whatever the height of the box."""
    for height in (10.0, 31.0):
        result = elastic_constants(
            triangular(height, (4, 4)), springs, relax=False, symmetry=False
        )
        assert isinstance(result, SheetElasticResult)
        modulus = np.sqrt(3) * K / 4 * units.EV_PER_A3_TO_GPA * 0.1  # N/m
        expected = modulus * np.array([[3, 1, 0], [1, 3, 0], [0, 0, 1]])
        # within the error of the differences, of the order of the strain squared
        np.testing.assert_allclose(result.constants, expected, rtol=1e-5, atol=1e-5)
        young = 8 / 3 * modulus
        for axis in ("x", "y"):
            assert result.young_modulus(axis) == pytest.approx(young, rel=1e-5)
            assert result.poisson_ratio(axis) == pytest.approx(1 / 3, rel=1e-5)
        assert result.shear_modulus() == pytest.approx(modulus, rel=1e-5)
        assert result.area_modulus() == pytest.approx(2 * modulus, rel=1e-5)
        assert not result.constants.flags.writeable


def test_the_moduli_of_an_anisotropic_sheet():
    c = np.array([[100.0, 20, 0], [20, 50, 0], [0, 0, 30]])
    result = SheetElasticResult(c)
    assert result.young_modulus("x") == pytest.approx(100 - 20**2 / 50)
    assert result.young_modulus("y") == pytest.approx(50 - 20**2 / 100)
    assert result.poisson_ratio("x") == pytest.approx(20 / 50)
    assert result.poisson_ratio("y") == pytest.approx(20 / 100)
    assert result.shear_modulus() == pytest.approx(30)
    assert result.area_modulus() == pytest.approx(1 / (np.linalg.inv(c)[:2, :2].sum()))
    with pytest.raises(InvalidParameter, match="axis"):
        result.young_modulus("z")
    with pytest.raises(InvalidParameter, match="stiffness"):
        SheetElasticResult(np.zeros((3, 3))).shear_modulus()


def test_symmetry_and_relaxation_change_nothing_here(springs):
    system = triangular(repeat=(4, 4))
    plain = elastic_constants(system, springs, relax=False, symmetry=False)
    pytest.importorskip("spglib")
    known = elastic_constants(system, springs, relax=False, symmetry=True)
    relaxed = elastic_constants(system, springs, relax=True, symmetry=False)
    np.testing.assert_allclose(known.constants, plain.constants, atol=1e-8)
    np.testing.assert_allclose(relaxed.constants, plain.constants, atol=1e-6)


def test_a_periodic_box_with_vacuum_gives_the_same_constants(springs):
    sheet = elastic_constants(
        triangular(repeat=(4, 4)), springs, relax=False, symmetry=False
    )
    periodic = elastic_constants(
        triangular(repeat=(4, 4), pbc=True), springs, relax=False, symmetry=False
    )
    index = [0, 1, 5]
    np.testing.assert_allclose(
        periodic.constants[np.ix_(index, index)] * HEIGHT * 0.1,
        sheet.constants,
        atol=1e-9,
    )
    with pytest.raises(InvalidParameter, match="sheet"):
        periodic.bulk_modulus()
    with pytest.raises(InvalidParameter, match="sheet"):
        periodic.shear_modulus()


def analytic_frequencies(q):
    """The frequencies in THz of the network of springs: its dynamical matrix
    in closed form, with no stiffness across the plane."""
    reciprocal = 2 * np.pi * np.linalg.inv(PLANE).T
    wave = np.asarray(q) @ reciprocal
    neighbors = [
        s * vector
        for s in (1, -1)
        for vector in (PLANE[0], PLANE[1], PLANE[0] + PLANE[1])
    ]
    matrix = np.zeros((3, 3))
    for vector in neighbors:
        unit_vector = vector / np.linalg.norm(vector)
        matrix += (1 - np.cos(wave @ vector)) * np.outer(unit_vector, unit_vector)
    values = np.linalg.eigvalsh(K / MASS * matrix)
    return np.sqrt(np.abs(values) / units.AMU_A2_PER_PS2_TO_EV) / (2 * np.pi)


def carbon(system):
    return system.with_columns(mass=[MASS] * len(system))


def test_the_frequencies_of_a_network_of_springs(springs):
    # springs at rest resist a displacement across the plane only by its cube:
    # a small displacement, and frequencies across the plane near zero
    result = phonons(
        carbon(triangular()), springs, supercell=4, displacement=1e-3, symmetry=False
    )
    np.testing.assert_array_equal(np.diag(result.supercell), (4, 4, 1))
    for q in ([0, 0, 0], [0.5, 0, 0], [1 / 3, 1 / 3, 0], [0.13, 0.31, 0]):
        np.testing.assert_allclose(
            result.frequencies(q), analytic_frequencies(q), atol=1e-2
        )
    # nothing along z: the wave vectors along z change nothing
    np.testing.assert_allclose(
        result.frequencies([0.13, 0.31, 0.4]), result.frequencies([0.13, 0.31, 0])
    )


def test_a_sheet_has_a_two_dimensional_mesh(springs):
    result = phonons(carbon(triangular()), springs, supercell=4, symmetry=False)
    # the two modes in the plane; the third has no frequency in this network
    dos = result.dos(16)
    above = dos.frequencies > 0.5
    assert np.trapezoid(dos.dos[above], dos.frequencies[above]) == pytest.approx(
        2.0, rel=0.03
    )
    one, three = (
        result.thermal([300.0], mesh=16),
        result.thermal([300.0], mesh=(16, 16, 1)),
    )
    assert one.free_energy[0] == pytest.approx(three.free_energy[0])
    with pytest.raises(InvalidParameter, match="along z"):
        result.thermal([300.0], mesh=(16, 16, 2))
    with pytest.raises(InvalidParameter, match="along z"):
        result.dos((4, 4, 4))


def test_the_default_supercell_is_in_the_plane(springs):
    result = phonons(carbon(triangular()), springs, symmetry=False)
    matrix = result.supercell
    assert (matrix[2] == (0, 0, 1)).all() and (matrix[:2, 2] == 0).all()
    plane = matrix[:2, :2] @ PLANE[:2, :2]
    widths = abs(np.linalg.det(plane)) / np.linalg.norm(plane[[1, 0]], axis=1)
    assert (widths >= 2 * springs.cutoff - 1e-9).all()


def test_the_supercell_of_a_sheet_stays_in_its_plane(springs):
    for wrong in ((4, 4, 2), [[4, 0, 1], [0, 4, 0], [0, 0, 1]]):
        with pytest.raises(InvalidParameter, match="plane"):
            phonons(carbon(triangular()), springs, supercell=wrong, symmetry=False)
    matrix = phonons(
        carbon(triangular()),
        springs,
        supercell=[[4, 1, 0], [0, 4, 0], [0, 0, 1]],
        symmetry=False,
    ).supercell
    assert matrix[0, 1] == 1


def test_a_small_box_along_z_makes_no_images(springs):
    """A sheet needs no vacuum: the atoms of a layer do not meet themselves
    across a box lower than the cutoff."""
    low = phonons(carbon(triangular(1.0)), springs, supercell=4, symmetry=False)
    high = phonons(carbon(triangular()), springs, supercell=4, symmetry=False)
    q = [[0.13, 0.31, 0], [0.5, 0, 0]]
    np.testing.assert_allclose(low.frequencies(q), high.frequencies(q), atol=1e-9)


def test_the_symmetry_of_a_sheet_gives_the_same_phonons(springs):
    pytest.importorskip("spglib")
    system = carbon(triangular())
    plain = phonons(system, springs, supercell=4, symmetry=False)
    known = phonons(system, springs, supercell=4, symmetry=True)
    # the same within the error of the differences, of the order of h squared
    np.testing.assert_allclose(known.force_constants, plain.force_constants, atol=1e-4)


def test_band_paths_stay_in_the_plane(springs):
    result = phonons(carbon(triangular()), springs, supercell=4, symmetry=False)
    bands = result.band_structure("hexagonal", points=5)
    assert [label for _, label in bands.labels] == ["Γ", "M", "K", "Γ"]
    assert np.abs(bands.q[:, 2]).max() < 1e-9
    pytest.importorskip("seekpath")
    auto = result.band_structure("auto", points=5)
    assert [label for _, label in auto.labels] == ["Γ", "M", "K", "Γ"]
    assert np.abs(auto.q[:, 2]).max() < 1e-9
    np.testing.assert_allclose(auto.frequencies, bands.frequencies, atol=1e-6)


def test_the_primitive_cell_of_a_sheet():
    big = carbon(triangular(repeat=(3, 2)))
    cell = primitive_cell(big)
    assert len(cell) == 1
    assert cell.box.pbc.tolist() == [True, True, False]
    assert cell.box.matrix[2, 2] == HEIGHT
    assert abs(np.linalg.det(cell.box.matrix)) == pytest.approx(
        abs(np.linalg.det(PLANE))
    )
    assert np.linalg.det(cell.box.matrix) > 0
    small = carbon(triangular())
    assert primitive_cell(small) is small


@pytest.mark.parametrize(
    "matrix, pbc",
    [
        (PLANE, (True, False, True)),
        (PLANE, (False, False, False)),
        (PLANE + [[0, 0, 0], [0, 0, 0], [1.0, 0, 0]], SHEET),
        (PLANE + [[0, 0, 1.0], [0, 0, 0], [0, 0, 0]], SHEET),
    ],
)
def test_other_boxes_are_refused(springs, matrix, pbc):
    system = carbon(
        mdapy.System([[0, 0, 5.0]], mdapy.Box(matrix, pbc=pbc), elements=["C"])
    )
    with pytest.raises(InvalidParameter, match="sheet"):
        elastic_constants(system, springs)
    with pytest.raises(InvalidParameter, match="sheet"):
        phonons(system, springs)
    with pytest.raises(InvalidParameter, match="sheet"):
        primitive_cell(system)


def test_no_atoms(springs):
    system = triangular()
    empty = system.gather(np.zeros(0, dtype=int))
    with pytest.raises(InvalidParameter, match="atoms"):
        primitive_cell(empty)
    with pytest.raises(InvalidParameter, match="atoms"):
        phonons(empty, springs)


def test_the_quasi_harmonic_approximation_is_for_crystals(springs):
    with pytest.raises(InvalidParameter, match="periodic in every direction"):
        quasi_harmonic(carbon(triangular()), springs, [300.0])


def test_minimize_relaxes_the_area_of_a_sheet():
    start = triangular(repeat=(4, 4))
    stretched = start.with_box(
        mdapy.Box(start.box.matrix * [[1.01], [1.01], [1.0]], pbc=SHEET),
        scale_positions=True,
    )
    result = minimize(
        stretched,
        Harmonic(k=K, r0=A),
        cell="full",
        max_force=1e-5,
        max_stress=1e-3,
        max_steps=20000,
    )
    assert result.converged
    assert np.linalg.norm(result.system.box.matrix[0]) == pytest.approx(4 * A, rel=1e-3)
    assert result.system.box.matrix[2, 2] == pytest.approx(HEIGHT)
