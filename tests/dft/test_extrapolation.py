# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The extrapolation of the density between the steps of molecular dynamics
(restart="second_order"): the coefficients of the fit are exact for atoms that
move with a constant acceleration each; the steps of dynamics give the same
energies and forces as with the atomic restart, in fewer iterations."""

import os

import numpy as np
import pytest

import mdapy
from mdapy.calculators import DFT
from mdapy.calculators._dft.calculator import HARTREE
from mdapy.errors import InvalidParameter

from dft import espresso, gpu

_dft = gpu.module()

CELL = 5.431 * np.eye(3)
DIAMOND = (
    np.array(
        [[0, 0, 0], [0, 2, 2], [2, 0, 2], [2, 2, 0], [1, 1, 1], [1, 3, 3], [3, 1, 3], [3, 3, 1]]
    )
    / 4.0
)  # fmt: skip


def coefficients(path, cell=CELL):
    return _dft.extrapolation_coefficients(*path, np.asarray(cell, dtype=float))


def test_the_fit_is_exact_for_a_constant_acceleration_each():
    rng = np.random.default_rng(3)
    start = rng.random((6, 3))
    velocity = 0.01 * rng.standard_normal((6, 3))
    acceleration = 0.002 * rng.standard_normal((6, 3))
    path = [start + velocity * t + 0.5 * acceleration * t * t for t in range(4)]
    alpha, beta = coefficients(path)
    assert alpha == pytest.approx(2.0, abs=1e-9)
    assert beta == pytest.approx(-1.0, abs=1e-9)


def test_the_fit_of_a_line_and_of_images():
    rng = np.random.default_rng(4)
    start = rng.random((5, 3))
    velocity = 0.01 * rng.standard_normal((5, 3))
    # equal steps: the fit has no unique answer and takes first order
    assert coefficients([start + velocity * t for t in range(4)]) == (1.0, 0.0)
    # atoms that stay put
    assert coefficients([start] * 4) == (1.0, 0.0)
    # the displacements are those of the nearest images: atoms wrapped by a
    # lattice vector give the same coefficients
    acceleration = 0.002 * rng.standard_normal((5, 3))
    path = [start + velocity * t + 0.5 * acceleration * t * t for t in range(4)]
    wrapped = [p % 1.0 for p in path]
    assert coefficients(wrapped) == pytest.approx(coefficients(path), abs=1e-9)
    # the least squares when the third step is not on the parabola: the
    # residual is orthogonal to both previous displacements
    path[3] = path[3] + 0.001 * rng.standard_normal((5, 3))
    alpha, beta = coefficients(path)
    a, b, c = (np.diff(path, axis=0)[i] @ CELL for i in (1, 0, 2))
    residual = c - alpha * a - beta * b
    assert np.sum(residual * a) == pytest.approx(0.0, abs=1e-9)
    assert np.sum(residual * b) == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(ValueError, match="the positions of the same atoms"):
        _dft.extrapolation_coefficients(start, start, start, start[:3], CELL)


def test_the_state_steps():
    state = _dft.State()
    assert state.steps == 1
    state.steps = 3
    state.clear()
    assert state.steps == 3
    with pytest.raises(ValueError, match="steps must be 1 or 3"):
        state.steps = 2
    with pytest.raises(InvalidParameter, match="restart must be"):
        DFT(12 * HARTREE, restart="third_order")


@pytest.fixture(scope="module")
def folder():
    espresso.pseudo("Si_std.upf")
    return os.environ["MDAPY_PSEUDO_DIR"]


def silicon(fractions):
    return mdapy.System(
        fractions @ CELL, mdapy.Box(CELL), elements=["Si"] * len(fractions)
    )


@pytest.mark.skipif(_dft.device_count() == 0, reason="no GPU")
def test_molecular_dynamics_of_silicon(folder):
    """Eight atoms at 1000 K for 10 steps of velocity Verlet with the DFT:
    the positions of a run with the atomic restart, computed again with the
    second-order one, give the same energies and forces to the tolerance of
    the field, in fewer iterations."""
    kB, MASS, CONVERT, dt, steps = 8.617333e-5, 28.0855, 9.648533e-3, 2.0, 10
    settings = dict(encut=12 * HARTREE, kpoints=(2, 2, 2), smearing="fixed", functional="pbe_pw",
                    tolerance=1e-10, pseudopotentials=folder)  # fmt: skip
    atomic = DFT(restart=True, **settings)
    rng = np.random.default_rng(7)
    positions = DIAMOND @ CELL + 0.05 * rng.standard_normal((8, 3))
    velocities = np.sqrt(kB * 1000.0 / MASS * CONVERT) * rng.standard_normal((8, 3))
    path, energies, forces, iterations = [], [], [], []

    def compute(x):
        r = atomic.compute(silicon(x @ np.linalg.inv(CELL)))
        path.append(x.copy())
        energies.append(r.energy)
        forces.append(r.forces)
        iterations.append(atomic.last["iterations"])
        return r.forces * CONVERT / MASS

    acceleration = compute(positions)
    for _ in range(steps):
        positions = positions + velocities * dt + 0.5 * acceleration * dt * dt
        new = compute(positions)
        velocities = velocities + 0.5 * (acceleration + new) * dt
        acceleration = new
    second = DFT(restart="second_order", **settings)
    second_iterations = []
    for x, e, f in zip(path, energies, forces):
        r = second.compute(silicon(x @ np.linalg.inv(CELL)))
        second_iterations.append(second.last["iterations"])
        assert r.energy == pytest.approx(e, abs=1e-7)
        np.testing.assert_allclose(r.forces, f, atol=1e-5)
    assert second.last["restarted"]
    print(
        f"\niterations per step, atomic {iterations}, second order {second_iterations}"
    )
    print(f"total {sum(iterations)} and {sum(second_iterations)}")
    assert sum(second_iterations) < sum(iterations)
