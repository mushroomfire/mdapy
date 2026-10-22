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


class Searching(EAM):
    """EAM that always searches its own neighbors."""

    def compute(self, system, properties=("energy", "forces"), *, neighbors=None):
        return super().compute(system, properties)


def test_one_search_serves_several_steps(potential, monkeypatch):
    # the lists from a longer search are those of new searches, to the last bit
    system = alloy(seed=4, repeat=2, jitter=0.15)
    calls = []
    build = mdapy.NeighborList.__dict__["build"]

    def counted(cls, *args, **kwargs):
        calls.append(1)
        return build.__func__(cls, *args, **kwargs)

    monkeypatch.setattr(mdapy.NeighborList, "build", classmethod(counted))
    result = minimize(system, potential, max_force=1e-4, max_steps=300)
    searches = len(calls)
    reference = minimize(
        system, Searching(potential.path), max_force=1e-4, max_steps=300
    )
    assert result.steps == reference.steps > 20
    np.testing.assert_array_equal(result.system.positions, reference.system.positions)
    assert result.energy == reference.energy
    assert searches < result.steps / 2


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
    with pytest.raises(
        InvalidParameter, match=r"does not compute \['energy', 'forces'\]"
    ):
        minimize(alloy(), EnergyOnly())


def copper(a, repeat=2):
    system = alloy(seed=0, repeat=repeat, jitter=0.0, a=a)
    return system.with_columns(element=["Cu"] * len(system))


def energy_at(potential, a, repeat=2):
    return potential.compute(copper(a, repeat)).energy


@pytest.mark.parametrize("cell", ["isotropic", "axes", "full"])
def test_the_box_relaxes_to_zero_stress(potential, cell):
    start = copper(3.4)  # the minimum of fcc Cu is near 3.29 Å
    if cell != "isotropic":
        strain = [[1.02, 0.0, 0.0], [0.0, 0.98, 0.0], [0.0, 0.0, 1.01]]
        if cell == "full":
            strain = [[1.02, 0.03, 0.0], [0.0, 0.98, 0.02], [0.01, 0.0, 1.01]]
        start = start.with_box(
            mdapy.Box(start.box.matrix @ np.array(strain).T), scale_positions=True
        )
    result = minimize(
        start, potential, cell=cell, max_force=1e-6, max_stress=1e-5, max_steps=5000
    )
    assert result.converged
    assert np.abs(result.stress).max() < 1e-5
    matrix = result.system.box.matrix
    a = np.sqrt(np.trace(matrix @ matrix.T) / 3) / 2
    # cubic again (a stress of 1e-5 GPa leaves strains of about 1e-6), at the
    # minimum of E(a) by finite differences of the energy
    np.testing.assert_allclose(matrix @ matrix.T, (2 * a) ** 2 * np.eye(3), atol=1e-4)
    h = 1e-4
    slope = (energy_at(potential, a + h) - energy_at(potential, a - h)) / (2 * h)
    assert abs(slope) < 1e-4
    assert result.energy == pytest.approx(energy_at(potential, a), abs=1e-9)


def test_the_box_relaxes_under_pressure(potential):
    result = minimize(
        copper(3.3),
        potential,
        cell="isotropic",
        pressure=5.0,
        max_force=1e-6,
        max_stress=1e-6,
        max_steps=5000,
    )
    assert result.converged
    np.testing.assert_allclose(result.stress, -5.0 * np.eye(3), atol=1e-6)
    # p = -dE/dV, from finite differences of the energy
    a, h = result.system.box.matrix[0, 0] / 2, 1e-4
    volume = lambda x: (2 * x) ** 3  # noqa: E731
    slope = (energy_at(potential, a + h) - energy_at(potential, a - h)) / (
        volume(a + h) - volume(a - h)
    )
    assert -slope * mdapy.units.EV_PER_A3_TO_GPA == pytest.approx(5.0, abs=1e-4)


def test_only_periodic_directions_relax(potential):
    slab = copper(3.4).with_box(mdapy.Box(2 * 3.4, pbc=(True, True, False)))
    result = minimize(slab, potential, cell="full", max_force=1e-4, max_stress=1e-4)
    assert result.converged
    matrix = result.system.box.matrix
    np.testing.assert_array_equal(matrix[2], slab.box.matrix[2])
    np.testing.assert_array_equal(matrix[:, 2], slab.box.matrix[:, 2])
    assert np.abs(result.stress[:2, :2]).max() < 1e-4
    isotropic = minimize(slab, potential, cell="isotropic", max_stress=1e-4)
    m = isotropic.system.box.matrix
    assert m[0, 0] == pytest.approx(m[1, 1]) and m[2, 2] == slab.box.matrix[2, 2]


def test_cell_options(potential):
    with pytest.raises(InvalidParameter, match="cell must be one of"):
        minimize(copper(3.3), potential, cell="shape")
    with pytest.raises(InvalidParameter, match="max_stress must be positive"):
        minimize(copper(3.3), potential, cell="full", max_stress=0.0)

    class NoVirial(Harmonic):
        implemented = frozenset({"energy", "forces"})

    with pytest.raises(
        InvalidParameter, match=r"does not compute \['energy', 'forces', 'virial'\]"
    ):
        minimize(copper(3.3), NoVirial(), cell="isotropic")
    assert minimize(copper(3.3), NoVirial(), max_steps=2).stress is None
