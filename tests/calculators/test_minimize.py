# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

import numpy as np
import pytest

import mdapy
from mdapy.calculators import EAM, minimize
from mdapy.errors import InvalidParameter

from .eam_files import alloy, write_setfl
from .test_testing import EnergyOnly, Harmonic


@pytest.fixture
def potential(tmp_path):
    return EAM(write_setfl(tmp_path / "CuNi.eam.alloy"))


def test_a_dimer_reaches_its_bond_length():
    dimer = mdapy.System(
        [[0.0, 0, 0], [3.0, 0.4, 0]], mdapy.Box(20.0, pbc=False), elements="H"
    )
    result = minimize(dimer, Harmonic(k=2.0, r0=2.5), max_force=1e-8)
    assert result.converged and 0 < result.steps < 1000
    distance = np.linalg.norm(result.system.positions[1] - result.system.positions[0])
    assert distance == pytest.approx(2.5, abs=1e-8)
    assert result.energy == pytest.approx(0.0, abs=1e-15)
    assert result.max_force < 1e-8


def test_a_perturbed_lattice_relaxes_back(potential):
    # near the minimum of E(a), where fcc is stable
    perfect = alloy(seed=1, repeat=2, jitter=0.0, a=3.3).with_columns(
        element=["Cu"] * 32
    )
    rng = np.random.default_rng(2)
    rattled = perfect.with_positions(
        perfect.positions + rng.normal(scale=0.05, size=(32, 3))
    )
    result = minimize(rattled, potential, max_force=1e-6, max_steps=5000)
    assert result.converged
    assert result.energy < potential.compute(rattled).energy
    assert result.energy == pytest.approx(potential.compute(perfect).energy, abs=1e-8)
    nl = mdapy.NeighborList.build(result.system, 2.8)
    np.testing.assert_allclose(nl.distances, 3.3 / np.sqrt(2), atol=1e-5)
    assert (nl.counts == 12).all()
    table = result.system.table
    np.testing.assert_allclose(
        np.linalg.norm(table["forces"].to_numpy(), axis=1).max(), result.max_force
    )
    assert result.system.info["energy"] == result.energy
    assert "atomic_energies" in table.columns


def test_results_do_not_depend_on_threads(potential):
    system = alloy(seed=3, repeat=2, jitter=0.1)
    runs = []
    for threads in (1, 4):
        with mdapy.num_threads(threads):
            runs.append(minimize(system, potential, max_force=1e-3))
    assert runs[0].steps == runs[1].steps
    np.testing.assert_array_equal(runs[0].system.positions, runs[1].system.positions)


def test_step_limit_callback_and_empty_systems(potential):
    system = alloy(seed=4, repeat=2, jitter=0.2)
    stopped = minimize(system, potential, max_steps=3)
    assert not stopped.converged and stopped.steps == 3
    seen = []

    def callback(step, current, energy, largest):
        seen.append(step)
        return step == 5

    result = minimize(system, potential, callback=callback)
    assert seen == [1, 2, 3, 4, 5] and result.steps == 5 and not result.converged
    empty = mdapy.System(np.zeros((0, 3)), mdapy.Box(5.0), elements=[])
    assert minimize(empty, potential).converged


def test_large_steps_are_limited():
    dimer = mdapy.System(
        [[0.0, 0, 0], [2.9, 0, 0]], mdapy.Box(20.0, pbc=False), elements="H"
    )
    moves = []

    def callback(step, current, energy, largest):
        moves.append(current.positions.copy())

    minimize(
        dimer, Harmonic(k=1000.0, r0=2.5), max_step=0.01, max_steps=5, callback=callback
    )
    steps = np.diff(np.array([dimer.positions] + moves), axis=0)
    assert np.linalg.norm(steps, axis=2).max() <= 0.01 + 1e-15


@pytest.mark.parametrize(
    "options, message",
    [
        ({"max_force": 0.0}, "max_force must be positive"),
        ({"max_steps": -1}, "max_steps must be at least 0"),
        ({"max_step": 0.0}, "max_step positive"),
    ],
)
def test_invalid_options(potential, options, message):
    with pytest.raises(InvalidParameter, match=message):
        minimize(alloy(), potential, **options)


def test_calculators_without_forces():
    with pytest.raises(InvalidParameter, match="both energies and forces"):
        minimize(alloy(), EnergyOnly())
