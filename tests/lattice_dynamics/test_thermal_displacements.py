# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Thermal displacements of atoms: springs of a simple cubic lattice, whose
modes are known in closed form, the classical limit, the symmetry of the
crystal. phonopy on the same force constants is in
test_force_constants_phonopy.py."""

import numpy as np
import pytest

from mdapy import units
from mdapy.build import build_crystal
from mdapy.elements import atomic_masses
from mdapy.errors import InvalidParameter
from mdapy.lattice_dynamics import ForceConstants, phonons

from calculators.test_testing import Harmonic

A, K = 2.5, 1.3


@pytest.fixture(scope="module")
def cubic():
    cell = build_crystal("sc", A, elements="Ar")
    return phonons(cell, Harmonic(k=K, r0=A), displacement=1e-5, symmetry=False)


def test_springs_of_a_simple_cubic_lattice(cubic):
    """The mode along x moves the atom along x alone: its mean square
    displacement is the mean over the mesh of hbar (n + 1/2) / (m omega)."""
    mesh = 6
    temperatures = [0.0, 50.0, 300.0]
    found = cubic.thermal_displacements(temperatures, mesh=mesh)
    mass = atomic_masses(["Ar"])[0]
    q = np.arange(mesh) / mesh
    grid = np.stack(np.meshgrid(q, q, q, indexing="ij"), -1).reshape(-1, 3)
    # omega in rad/ps of the mode along x
    omega = (
        2
        * np.sqrt(K / mass / units.AMU_A2_PER_PS2_TO_EV)
        * np.abs(np.sin(np.pi * grid[:, 0]))
    )
    hbar = units.PLANCK / (2 * np.pi)
    for t, temperature in enumerate(temperatures):
        moving = omega > 2 * np.pi * 1e-3
        quanta = hbar * omega[moving]
        occupation = (
            1 / np.expm1(quanta / (units.BOLTZMANN * temperature))
            if temperature
            else 0.0
        )
        expected = (hbar * (occupation + 0.5) / omega[moving]).sum() / len(grid)
        expected /= mass * units.AMU_A2_PER_PS2_TO_EV
        np.testing.assert_allclose(
            found[t, 0], expected * np.eye(3), rtol=1e-6, atol=1e-14
        )


def test_the_classical_limit(cubic):
    """Hot, each mode holds k T: the displacements grow as T, less a
    constant of the zero point motion that vanishes."""
    hot = cubic.thermal_displacements([2000.0, 4000.0], mesh=6)
    assert hot[1, 0, 0, 0] == pytest.approx(2 * hot[0, 0, 0, 0], rel=1e-3)
    cold = cubic.thermal_displacements([0.0, 1e-6], mesh=6)
    np.testing.assert_allclose(cold[0], cold[1])
    assert cold[0, 0, 0, 0] > 0  # the zero point motion


def test_the_shapes_and_the_errors(cubic):
    found = cubic.thermal_displacements(300.0, mesh=(4, 4, 4))
    assert found.shape == (1, 1, 3, 3)
    with pytest.raises(InvalidParameter, match="temperatures"):
        cubic.thermal_displacements([-1.0])
    with pytest.raises(InvalidParameter, match="mesh"):
        cubic.thermal_displacements([300.0], mesh=0)
    unstable = ForceConstants(cubic.cell, cubic.supercell, -cubic.force_constants)
    with pytest.raises(InvalidParameter, match="imaginary"):
        unstable.phonons().thermal_displacements([300.0], mesh=4)
