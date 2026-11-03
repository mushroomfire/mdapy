# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The free energy of a stacking fault against its energy at rest, its harmonic
vibrations and a slip that leaves the crystal as it is. Skipped where LAMMPS or
its package EXTRA-FIX is missing."""

import numpy as np
import pytest

import mdapy
from mdapy import units
from mdapy.build import build_crystal
from mdapy.calculators import LAMMPS
from mdapy.core.threads import CancelToken
from mdapy.errors import Cancelled, InvalidParameter, MissingProperty
from mdapy.lattice_dynamics import phonons
from mdapy.simulation import minimize, stacking_fault_energy, stacking_fault_free_energy

pytest.importorskip("lammps")

A = 5.26  # about the lattice constant of the model of argon
ORIENTATION = [(1, 1, -2), (-1, 1, 0), (1, 1, 1)]
FAST = dict(timestep=0.004, equilibration=2.0, switching=8.0, repeats=3, seed=2)


@pytest.fixture(scope="module")
def potential():
    return LAMMPS(
        "pair_style lj/cut 8.5\npair_coeff * * 0.0104 3.4\npair_modify shift yes",
        ["Ar"],
    )


@pytest.fixture(scope="module")
def layers(potential):
    start = build_crystal(
        "fcc", A, elements="Ar", orientation=ORIENTATION, repeat=(2, 3, 3)
    )
    return minimize(start, potential, cell="full", max_stress=1e-3).system


def partial(system):
    """1/6 [11-2], in a box that holds two periods of 1/2 [11-2] along x."""
    return np.array([system.box.matrix[0, 0] / 6, 0, 0])


def test_a_whole_lattice_vector_makes_no_fault(potential, layers):
    # half of the box along x, which holds two periods of 1/2 [11-2]
    whole = [layers.box.matrix[0, 0] / 2, 0, 0]
    result = stacking_fault_free_energy(layers, potential, whole, 30.0, **FAST)
    assert abs(result.energy[0]) < 3 * result.error[0] + 2e-3


def test_cold_as_at_rest(potential, layers):
    direction = partial(layers)
    rest = stacking_fault_energy(layers, potential, direction, points=2).energy[1]
    result = stacking_fault_free_energy(
        layers, potential, direction, 5.0, pressure=0.0, **FAST
    )
    assert result.energy[0] == pytest.approx(rest, abs=2 * result.error[0] + 1e-3)
    assert result.temperatures.tolist() == [5.0]
    assert result.area == pytest.approx(
        np.linalg.norm(np.cross(layers.box.matrix[0], layers.box.matrix[1])), rel=0.01
    )
    assert result.dissipation[0] < 1e-3
    assert not result.energy.flags.writeable


def test_warm_as_its_harmonic_vibrations(potential, layers):
    """At a moderate temperature in a fixed size: the classical vibrations of the
    two cells, by the phonons of each."""
    temperature = 25.0
    direction = partial(layers)
    result = stacking_fault_free_energy(
        layers,
        potential,
        direction,
        temperature,
        **dict(FAST, switching=20.0, repeats=4),
    )
    assert result.volume[0] > layers.box.volume / len(layers)
    scale = (result.volume[0] * len(layers) / layers.box.volume) ** (1 / 3)
    box = mdapy.Box(layers.box.matrix * scale)
    grown = layers.with_box(box, scale_positions=True)
    from mdapy.build._defects import slip
    from mdapy.simulation._defect_energies import _options, _relaxed

    moved = slip(grown, direction)
    movable = np.zeros((len(grown), 3), dtype=bool)
    movable[:, 2] = True
    energy, fault = _relaxed(
        moved.with_columns(movable=movable), potential, _options(1e-6, 5000)
    )
    rest = potential.compute(grown).energy
    kt = units.BOLTZMANN * temperature

    def vibrations(cell):
        result = phonons(cell, potential, supercell=(2, 2, 1), symmetry=False)
        q = np.stack(
            np.meshgrid(*[(np.arange(n) + 0.5) / n for n in (6, 6, 2)], indexing="ij"),
            -1,
        ).reshape(-1, 3)
        return (
            len(cell)
            * 3
            * kt
            * np.log(units.PLANCK * result.frequencies(q) / kt).mean()
        )

    area = np.linalg.norm(np.cross(grown.box.matrix[0], grown.box.matrix[1]))
    expected = (
        (energy - rest + vibrations(fault.drop("movable")) - vibrations(grown))
        / area
        * units.EV_PER_A2_TO_J_PER_M2
    )
    assert result.energy[0] == pytest.approx(expected, abs=4 * result.error[0] + 2e-3)


def test_one_repeat_has_no_error_and_several_temperatures(potential, layers):
    result = stacking_fault_free_energy(
        layers,
        potential,
        partial(layers),
        [20.0, 40.0],
        **dict(FAST, repeats=1, switching=1.0, equilibration=0.2),
    )
    assert (result.error == 0).all() and len(result.energy) == 2
    assert result.volume[0] < result.volume[1]


def test_progress_and_cancel(potential, layers):
    seen = []
    options = dict(FAST, repeats=1, switching=0.2, equilibration=0.1)
    stacking_fault_free_energy(
        layers,
        potential,
        partial(layers),
        20.0,
        progress=lambda d, t: seen.append((d, t)),
        **options,
    )
    assert seen[-1][0] == seen[-1][1]
    token = CancelToken()
    token.cancel()
    with pytest.raises(Cancelled):
        stacking_fault_free_energy(
            layers, potential, partial(layers), 20.0, cancel=token, **options
        )


def test_invalid_input(potential, layers):
    direction = partial(layers)
    for wrong, match in (
        (dict(timestep=0.0), "positive"),
        (dict(equilibration=0.0), "positive"),
        (dict(switching=0.0), "positive"),
        (dict(repeats=0), "positive"),
        (dict(seed=0), "positive"),
    ):
        with pytest.raises(InvalidParameter, match=match):
            stacking_fault_free_energy(layers, potential, direction, 20.0, **wrong)
    with pytest.raises(InvalidParameter, match="temperatures"):
        stacking_fault_free_energy(layers, potential, direction, 0.0)
    with pytest.raises(InvalidParameter, match="three"):
        stacking_fault_free_energy(layers, potential, [1.0, 0.0], 20.0)
    with pytest.raises(InvalidParameter, match="LAMMPS"):
        stacking_fault_free_energy(layers, object(), direction, 20.0)
    with pytest.raises(MissingProperty, match="element"):
        stacking_fault_free_energy(layers.drop("element"), potential, direction, 20.0)
    with pytest.raises(InvalidParameter, match="periodic"):
        stacking_fault_free_energy(
            layers.with_box(mdapy.Box(layers.box.matrix, pbc=(True, True, False))),
            potential,
            direction,
            20.0,
        )
    tilted = build_crystal("fcc", A, elements="Cu")
    with pytest.raises(InvalidParameter, match="normal"):
        stacking_fault_free_energy(
            tilted.with_box(
                mdapy.Box(tilted.box.matrix + [[0, 0, 1.0], [0, 0, 0], [0, 0, 0]])
            ),
            potential,
            direction,
            20.0,
        )
