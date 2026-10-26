# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

import numpy as np
import pytest

import mdapy
from mdapy.calculators import EAM
from mdapy.errors import InvalidParameter
from mdapy.simulation import minimize

from calculators.eam_files import alloy, write_setfl
from calculators.test_testing import EnergyOnly, Harmonic


@pytest.fixture
def potential(tmp_path):
    return EAM(write_setfl(tmp_path / "CuNi.eam.alloy"))


def test_a_dimer_reaches_its_bond_length():
    dimer = mdapy.System(
        [[0.0, 0, 0], [3.0, 0.4, 0]], mdapy.Box(20.0, pbc=False), elements="H"
    )
    result = minimize(dimer, Harmonic(k=2.0, r0=2.5), max_force=1e-8)
    assert result.converged and 0 < result.steps < 1000
    distance = np.linalg.norm(
        result.system["positions"][1] - result.system["positions"][0]
    )
    assert distance == pytest.approx(2.5, abs=1e-8)
    assert result.energy == pytest.approx(0.0, abs=1e-15)
    assert result.max_force < 1e-8


def test_a_perturbed_lattice_relaxes_back(potential):
    # near the minimum of E(a), where fcc is stable
    perfect = alloy(seed=1, repeat=2, jitter=0.0, a=3.3).with_columns(
        element=["Cu"] * 32
    )
    rng = np.random.default_rng(2)
    rattled = perfect.with_columns(
        positions=perfect["positions"] + rng.normal(scale=0.05, size=(32, 3))
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
    np.testing.assert_array_equal(
        runs[0].system["positions"], runs[1].system["positions"]
    )


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
        result.system["positions"], reference.system["positions"], rtol=0, atol=1e-10
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
    assert minimize(empty, potential, cell="full").converged


def lammps_fire(system, calculator, steps, dmax=0.1, dt0=0.001):
    """LAMMPS min_style fire with its default settings, transcribed atom by
    atom from min_fire.cpp, for the given number of steps."""
    ftm2v = 1.0 / mdapy.units.AMU_A2_PER_PS2_TO_EV
    mass = mdapy.elements.atomic_masses(system.table["element"].to_physical())
    x = system["positions"].copy()
    v = np.zeros_like(x)
    n = len(x)

    def forces_at(positions):
        return calculator.compute(system.with_columns(positions=positions)).forces

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
    np.testing.assert_allclose(result.system["positions"], expected, rtol=0, atol=1e-9)


def test_large_steps_are_limited():
    dimer = mdapy.System(
        [[0.0, 0, 0], [2.9, 0, 0]], mdapy.Box(20.0, pbc=False), elements="H"
    )
    moves = []

    def callback(step, current, energy, largest):
        moves.append(current["positions"].copy())

    calculator = Harmonic(k=20.0, r0=2.5)
    result = minimize(dimer, calculator, max_step=0.01, max_steps=8, callback=callback)
    assert result.steps == 8
    steps = np.abs(np.diff(np.array([dimer["positions"]] + moves), axis=0))
    assert 0.0 < steps[0].max() <= 0.01  # from rest, at most max_step along an axis
    np.testing.assert_allclose(
        moves[-1], lammps_fire(dimer, calculator, 8, dmax=0.01), rtol=0, atol=1e-12
    )
    free = minimize(dimer, calculator, max_steps=8).system["positions"]
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
    bare = mdapy.System(system["positions"], system.box)
    with pytest.raises(mdapy.errors.MissingProperty, match="mass or an element"):
        minimize(bare, potential)


@pytest.mark.parametrize("every", [1, 7, 50])
def test_frames_every_few_steps(potential, tmp_path, every):
    system = alloy(seed=9, repeat=2, jitter=0.15)
    plain = minimize(system, potential, max_force=1e-4, max_steps=40)
    result = minimize(system, potential, max_force=1e-4, max_steps=40, save_every=every)
    assert plain.frames is None and result.steps == plain.steps == 40
    steps = [frame.info["timestep"] for frame in result.frames]
    assert steps == sorted(set(list(range(0, 41, every)) + [40]))
    first, last = result.frames[0], result.frames[-1]
    np.testing.assert_array_equal(first["positions"], system["positions"])
    np.testing.assert_array_equal(last["positions"], plain.system["positions"])
    start = potential.compute(system, ("energy", "virial"))
    assert first.info["energy"] == start.energy
    np.testing.assert_array_equal(first.info["virial"], start.virial)
    np.testing.assert_array_equal(first.info["stress"], start.stress(system))
    assert last.info["energy"] == plain.energy
    np.testing.assert_array_equal(last.info["stress"], plain.stress)
    np.testing.assert_array_equal(plain.system.info["stress"], plain.stress)
    np.testing.assert_array_equal(
        last.table["forces"].to_numpy(), plain.system.table["forces"].to_numpy()
    )
    # the frames as the callback sees them
    seen = {}
    minimize(system, potential, max_force=1e-4, max_steps=40,
             callback=lambda step, current, energy, largest: seen.update({step: current}))  # fmt: skip
    for frame in result.frames[1:]:
        np.testing.assert_array_equal(
            frame["positions"], seen[frame.info["timestep"]]["positions"]
        )
    # written as a trajectory
    mdapy.write(tmp_path / "relax.xyz", result.frames)
    back = mdapy.read_trajectory(tmp_path / "relax.xyz")
    assert len(back) == len(result.frames)
    np.testing.assert_allclose(back[-1].info["stress"], plain.stress, atol=1e-12)


def test_frames_of_a_converged_start(potential):
    perfect = alloy(seed=1, repeat=2, jitter=0.0, a=3.3).with_columns(
        element=["Cu"] * 32
    )
    result = minimize(perfect, potential, max_force=1e-3, save_every=10)
    assert result.converged and result.steps == 0
    assert [f.info["timestep"] for f in result.frames] == [0]


@pytest.mark.parametrize(
    "options, message",
    [
        ({"max_force": 0.0}, "max_force must be positive"),
        ({"max_steps": -1}, "max_steps must be at least 0"),
        ({"max_step": 0.0}, "max_step and timestep positive"),
        ({"timestep": -1.0}, "max_step and timestep positive"),
        ({"save_every": 0}, "save_every must be a positive integer"),
        ({"save_every": 2.5}, "save_every must be a positive integer"),
        ({"save_every": True}, "save_every must be a positive integer"),
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
    # strained without rotating: a symmetric deformation of the start
    deformation = np.linalg.solve(start.box.matrix, matrix)
    np.testing.assert_allclose(deformation, deformation.T, rtol=0, atol=1e-14)
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


def test_the_box_relaxes_alike_at_any_size(potential):
    # a strained perfect crystal, whose atoms feel no forces: its box takes
    # the same path whatever the number of unit cells
    strain = np.array([[1.02, 0.01, 0.0], [0.0, 0.99, 0.01], [0.0, 0.0, 1.01]])
    runs = []
    for repeat in (2, 4):
        start = copper(3.3, repeat)
        start = start.with_box(
            mdapy.Box(start.box.matrix @ strain.T), scale_positions=True
        )
        runs.append(minimize(start, potential, cell="full", max_stress=1e-6))
    assert runs[0].converged and runs[1].converged
    assert runs[0].steps == runs[1].steps > 10
    np.testing.assert_allclose(
        runs[1].system.box.matrix, 2 * runs[0].system.box.matrix, rtol=1e-9
    )


def test_frames_of_a_relaxing_box(potential):
    start = copper(3.4)
    result = minimize(start, potential, cell="isotropic", save_every=5)
    assert result.converged
    volumes = [frame.box.volume for frame in result.frames]
    assert volumes[0] == start.box.volume and volumes[-1] < volumes[0]
    for frame in result.frames:
        values = potential.compute(frame, ("energy", "virial"))
        assert frame.info["energy"] == pytest.approx(values.energy, abs=1e-12)
        np.testing.assert_allclose(
            frame.info["stress"], values.stress(frame), atol=1e-10
        )
    assert result.frames[0].info["stress"][0, 0] > 1.0  # stretched: tension
    np.testing.assert_array_equal(result.frames[-1].info["stress"], result.stress)


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
    plain = minimize(copper(3.3), NoVirial(), max_steps=2, save_every=1)
    assert plain.stress is None
    assert all("stress" not in frame.info for frame in plain.frames)
