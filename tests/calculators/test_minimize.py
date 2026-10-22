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
    # the lists from a longer search hold the pairs of new searches, in
    # another order, so forces differ in the last bits only
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
    np.testing.assert_allclose(
        result.system.positions, reference.system.positions, rtol=0, atol=1e-10
    )
    assert result.energy == pytest.approx(reference.energy, rel=1e-13)
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


def lammps_fire(system, calculator, steps, dmax=0.1, dt0=0.001):
    """LAMMPS min_style fire with its default settings, transcribed atom by
    atom from min_fire.cpp, for the given number of steps."""
    ftm2v = 1.0 / mdapy.units.AMU_A2_PER_PS2_TO_EV
    mass = mdapy.elements.atomic_masses(system.table["element"].to_physical())
    x = system.positions.copy()
    v = np.zeros_like(x)
    n = len(x)

    def forces_at(positions):
        return calculator.compute(system.with_positions(positions)).forces

    f = forces_at(x)
    dt, alpha, last_negative, flagv0 = dt0, 0.25, 0, True
    for ntimestep in range(1, steps + 1):
        vdotf = sum(v[i] @ f[i] for i in range(n))
        if vdotf > 0.0:
            vdotv = sum(v[i] @ v[i] for i in range(n))
            fdotf = sum(f[i] @ f[i] for i in range(n))
            scale1 = 1.0 - alpha
            scale2 = 0.0 if fdotf <= 1e-20 else alpha * np.sqrt(vdotv / fdotf)
            if ntimestep - last_negative > 20:
                dt = min(dt * 1.1, 10.0 * dt0)
                alpha *= 0.99
        else:
            last_negative = ntimestep
            if not ntimestep < 20:
                alpha = 0.25
                if dt * 0.5 >= 0.02 * dt0:
                    dt *= 0.5
            for i in range(n):
                x[i] -= 0.5 * dt * v[i]
            v[:] = 0.0
            flagv0 = True
        if flagv0:
            f = forces_at(x)
            for i in range(n):
                v[i] = dt * ftm2v / mass[i] * f[i]
        dtv = dt
        for i in range(n):
            vmax = np.abs(v[i]).max()
            if dtv * vmax > dmax:
                dtv = dmax / vmax
        if flagv0:
            v[:] = 0.0
        for i in range(n):
            v[i] += dtv * ftm2v / mass[i] * f[i]
            if vdotf > 0.0:
                v[i] = scale1 * v[i] + scale2 * f[i]
            x[i] += dtv * v[i]
        f = forces_at(x)
        flagv0 = False
    return x


@pytest.mark.parametrize("steps", [1, 2, 5, 40])
def test_steps_of_lammps_fire(potential, steps):
    # rounding apart, the steps are those of LAMMPS, transcribed in NumPy;
    # the jitter makes uphill steps, which reset and shrink the time step
    system = alloy(seed=7, repeat=2, jitter=0.25)
    result = minimize(system, potential, max_force=1e-12, max_steps=steps)
    assert result.steps == steps
    expected = lammps_fire(system, potential, steps)
    np.testing.assert_allclose(result.system.positions, expected, rtol=0, atol=1e-9)


def test_large_steps_are_limited():
    dimer = mdapy.System(
        [[0.0, 0, 0], [2.9, 0, 0]], mdapy.Box(20.0, pbc=False), elements="H"
    )
    moves = []

    def callback(step, current, energy, largest):
        moves.append(current.positions.copy())

    calculator = Harmonic(k=20.0, r0=2.5)
    result = minimize(dimer, calculator, max_step=0.01, max_steps=8, callback=callback)
    assert result.steps == 8
    steps = np.abs(np.diff(np.array([dimer.positions] + moves), axis=0))
    assert 0.0 < steps[0].max() <= 0.01  # from rest, at most max_step along an axis
    np.testing.assert_allclose(
        moves[-1], lammps_fire(dimer, calculator, 8, dmax=0.01), rtol=0, atol=1e-12
    )
    free = minimize(dimer, calculator, max_steps=8).system.positions
    assert np.abs(free - moves[-1]).max() > 0.01  # the limit changed the path


def test_masses_set_the_path_not_the_minimum(potential):
    system = alloy(seed=8, repeat=2, jitter=0.1)
    light = minimize(system, potential, max_force=1e-6, max_steps=5000)
    heavy = minimize(
        system.with_columns(mass=np.full(len(system), 500.0)),
        potential,
        max_force=1e-6,
        max_steps=5000,
    )
    assert light.converged and heavy.converged and light.steps != heavy.steps
    assert light.energy == pytest.approx(heavy.energy, abs=1e-8)
    bare = mdapy.System(system.positions, system.box)
    with pytest.raises(mdapy.errors.MissingProperty, match="mass or an element"):
        minimize(bare, potential)


@pytest.mark.parametrize(
    "options, message",
    [
        ({"max_force": 0.0}, "max_force must be positive"),
        ({"max_steps": -1}, "max_steps must be at least 0"),
        ({"max_step": 0.0}, "max_step and timestep positive"),
        ({"timestep": -1.0}, "max_step and timestep positive"),
    ],
)
def test_invalid_options(potential, options, message):
    with pytest.raises(InvalidParameter, match=message):
        minimize(alloy(), potential, **options)


def test_calculators_without_forces():
    with pytest.raises(InvalidParameter, match="both energies and forces"):
        minimize(alloy(), EnergyOnly())
