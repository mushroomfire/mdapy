# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Group velocities of phonons: springs of a simple cubic lattice in closed
form, finite differences of the frequencies, and the symmetry of the
crystal. phonopy on the same force constants is in
test_force_constants_phonopy.py."""

import sys

import numpy as np
import pytest

import mdapy
from mdapy.build import build_crystal
from mdapy.elements import atomic_masses
from mdapy.errors import InvalidParameter
from mdapy.lattice_dynamics import phonons, primitive_cell
from mdapy.lattice_dynamics._phonons import _THZ

from calculators.test_testing import Harmonic


def test_springs_of_a_simple_cubic_lattice():
    """Each mode moves along an axis, of frequency 2 sqrt(k/m) |sin(k_α a/2)|:
    its velocity is a sqrt(k/m) cos(k_α a/2) along the axis."""
    a, k = 2.5, 1.3
    cell = build_crystal("sc", a, elements="Ar")
    result = phonons(cell, Harmonic(k=k, r0=a), displacement=1e-5, symmetry=False)
    mass = atomic_masses(["Ar"])[0]
    q = np.random.default_rng(0).random((20, 3)) - 0.5
    omega = 2 * np.sqrt(k / mass) * np.abs(np.sin(np.pi * q))  # √(eV/(Å² amu))
    speed = a * np.sqrt(k / mass) * np.cos(np.pi * q) * np.sign(q) * 2 * np.pi * _THZ
    order = np.argsort(omega, axis=1)
    expected = np.zeros((20, 3, 3))
    for m in range(3):
        axis = order[:, m]
        expected[np.arange(20), m, axis] = speed[np.arange(20), axis]
    np.testing.assert_allclose(result.group_velocities(q), expected, atol=1e-6)
    np.testing.assert_allclose(result.group_velocities(q[0]), expected[0], atol=1e-6)


@pytest.mark.parametrize("structure, a", [("fcc", 3.6), ("hcp", 2.55)])
def test_finite_differences_of_the_frequencies(structure, a):
    cell = build_crystal(structure, a, elements="Ar")
    if structure == "fcc":
        cell = primitive_cell(cell)
    springs = Harmonic(k=0.7, r0=a / np.sqrt(2) if structure == "fcc" else a)
    result = phonons(cell, springs, supercell=4, symmetry=False)
    q = np.random.default_rng(1).random((6, 3)) - 0.5
    reciprocal = 2 * np.pi * np.linalg.inv(cell.box.matrix).T
    h = 1e-5
    differences = np.zeros((6, 3 * len(cell), 3))
    for x in range(3):
        step = np.eye(3)[x] * h @ np.linalg.inv(reciprocal)
        rise = result.frequencies(q + step) - result.frequencies(q - step)
        differences[:, :, x] = 2 * np.pi * rise / (2 * h)
    np.testing.assert_allclose(
        result.group_velocities(q, symmetry=False), differences, atol=1e-6
    )


def test_the_symmetry_of_the_crystal():
    """Γ: no velocity. A wave vector along z of hcp: the velocities along z,
    each degenerate pair of modes the same. A sheet: in its plane."""
    pytest.importorskip("spglib")
    cell = build_crystal("hcp", 2.55, elements="Ar")
    result = phonons(cell, Harmonic(k=0.7, r0=2.55), supercell=(4, 4, 3))
    np.testing.assert_array_equal(result.group_velocities([0, 0, 0]), 0)
    frequencies = result.frequencies([0, 0, 0.2])
    v = result.group_velocities([0, 0, 0.2])
    np.testing.assert_allclose(v[:, :2], 0, atol=1e-10)
    pairs = np.flatnonzero(np.diff(frequencies) < 1e-6)
    np.testing.assert_allclose(v[pairs], v[pairs + 1], atol=1e-10)
    plain = result.group_velocities([0, 0, 0.2], symmetry=False)
    np.testing.assert_allclose(plain[:, 2], v[:, 2], atol=1e-10)

    side, height = 2.5, 10.0
    plane = np.array(
        [[side, 0, 0], [-side / 2, side * np.sqrt(3) / 2, 0], [0, 0, height]]
    )
    sheet = mdapy.System(
        np.array([[0, 0, height / 2]]),
        mdapy.Box(plane, pbc=(True, True, False)),
        elements=["C"],
    )
    flat = phonons(sheet, Harmonic(k=2.0, r0=2.4), supercell=4)
    velocities = flat.group_velocities([[0.1, 0.2, 0], [0.3, -0.1, 0]])
    np.testing.assert_allclose(velocities[..., 2], 0, atol=1e-10)
    assert np.abs(velocities[..., :2]).max() > 1


def test_errors(monkeypatch):
    cell = build_crystal("sc", 2.5, elements="Ar")
    result = phonons(cell, Harmonic(k=1.0, r0=2.5), symmetry=False)
    with pytest.raises(InvalidParameter, match="wave vector"):
        result.group_velocities([0.1, 0.2])
    monkeypatch.setitem(sys.modules, "spglib", None)
    with pytest.raises(InvalidParameter, match="spglib"):
        result.group_velocities([0.1, 0.2, 0.3], symmetry=True)
