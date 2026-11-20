# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The time step set by the fastest atom: in MDAPY and in LAMMPS (fix
dt/reset) the same runs step by step, with the constants of LAMMPS; the time
step of each step as the formula gives it; the errors."""

import numpy as np
import pytest

from mdapy.build import build_crystal
from mdapy.calculators import EAM, _gpu
from mdapy.errors import InvalidParameter
from mdapy.simulation import MolecularDynamics
from mdapy.simulation._integrator import _FTM2V, _step_limit
from mdapy.simulation.controls import AdaptiveTimestep, Deform, Hold, Langevin, Move
from simulation.test_controls import (  # noqa: F401
    gap,
    lammps_constants,
    potentials,
    slab,
)

from calculators.gpu_stand_in import stand_in


def knocked(seed=0, speed=150.0):
    """The slab with one atom of the middle sent off fast, as in a cascade."""
    system, bottom, top, middle = slab(seed=seed)
    velocities = system["velocities"].copy()
    atom = int(np.flatnonzero(middle)[0])
    velocities[atom] = speed * np.array([1.0, 3.0, 5.0]) / np.sqrt(35.0)
    return system.with_columns(velocities=velocities), bottom


CASES = [
    (AdaptiveTimestep(0.05), {"ensemble": "nve"}, False),
    (
        AdaptiveTimestep(0.03, min_timestep=1e-4, max_timestep=0.0015, every=3),
        {"ensemble": "nve"},
        False,
    ),
    (AdaptiveTimestep(0.05), {"ensemble": "nvt", "temperature": 300}, False),
    (AdaptiveTimestep(0.05), {"ensemble": "nve"}, True),
]


@pytest.mark.parametrize("adaptive, options, hold", CASES)
def test_the_same_runs_as_lammps(potentials, lammps_constants, adaptive, options, hold):  # noqa: F811
    eam, lmp = potentials
    if lmp is None:
        pytest.skip("LAMMPS is not installed")
    system, bottom = knocked()
    controls = [adaptive] + ([Hold(bottom)] if hold else [])
    out = []
    for potential in (lmp, eam):
        with MolecularDynamics(system, potential, timestep=0.002) as md:
            md.run(300, thermo=20, controls=controls, thermostat_damping=0.2, **options)
            md.run(20, thermo=10, ensemble="nve")  # the time step of md again
            out.append((md.system, md.thermo))
    (a, ta), (b, tb) = out
    dx, dv = gap(a, b)
    assert dx < 1e-9 and dv < 1e-8
    np.testing.assert_allclose(ta["time"], tb["time"], rtol=1e-12, atol=1e-15)
    for column in ("temperature", "potential_energy", "total_energy"):
        np.testing.assert_allclose(ta[column], tb[column], rtol=1e-9)
    # the steps were shorter while the atom was fast, and the last run took the
    # time step of md
    time = tb["time"].to_numpy()
    assert time[tb["step"].to_numpy() == 300][0] < 0.6
    np.testing.assert_allclose(np.diff(time[-3:]), 10 * 0.002, rtol=1e-12)


def test_the_limit_of_the_formula():
    """By the velocity alone, by the force alone, and both with the move of
    v dt + f dt^2 / 2m scaled to the distance."""
    ftm2v = _FTM2V
    m = np.array([2.0])
    v = np.array([[3.0, 0.0, 4.0]])
    zero = np.zeros((1, 3))
    assert _step_limit(v, zero, m, 0.1) == pytest.approx(0.1 / 5.0)
    f = np.array([[0.0, 2.0, 0.0]])
    assert _step_limit(zero, f, m, 0.1) == pytest.approx(
        np.sqrt(2 * 0.1 / (ftm2v * 2.0 / 2.0))
    )
    # the smaller of the two, scaled once by the length of its move
    first = min(0.1 / 5.0, np.sqrt(2 * 0.1 / (ftm2v * 2.0 / 2.0)))
    move = first * v + 0.5 * first**2 * f / m * ftm2v
    dt = _step_limit(v, f, m, 0.1)
    assert dt == pytest.approx(first * 0.1 / np.linalg.norm(move), rel=1e-14)
    assert _step_limit(zero, zero, m, 0.1) == 1.0e20


def test_the_bounds_and_the_time(potentials):  # noqa: F811
    eam, _ = potentials
    system, _ = knocked(speed=400.0)
    with MolecularDynamics(system, eam, timestep=0.002) as md:
        md.run(50, thermo=1, controls=AdaptiveTimestep(0.05, min_timestep=4e-4))
        time = md.thermo["time"].to_numpy()
    steps = np.diff(time)
    # the fast atom would want 1.25e-4 ps: the lower bound holds
    assert steps.min() == pytest.approx(4e-4, rel=1e-12)
    assert steps.max() <= 0.002 * (1 + 1e-12)


def test_on_the_gpu(monkeypatch):
    """The time steps on the GPU, its module stood in for on the processor:
    the runs of the processor."""
    monkeypatch.setattr(_gpu, "cuda", stand_in)
    system = build_crystal("fcc", 3.6, elements="Cu", repeat=(5, 5, 5))
    velocities = np.zeros((len(system), 3))
    velocities[0] = (200.0, 100.0, 50.0)
    system = system.with_columns(velocities=velocities)
    out = []
    for device in ("cuda", "cpu"):
        with MolecularDynamics(
            system, EAM.zhou(["Cu"], device=device), timestep=0.002
        ) as md:
            md.run(60, thermo=10, controls=AdaptiveTimestep(0.04))
            out.append((md.system, md.thermo))
    (a, ta), (b, tb) = out
    np.testing.assert_allclose(ta["time"], tb["time"], rtol=1e-12)
    np.testing.assert_allclose(a["positions"], b["positions"], atol=1e-9)
    np.testing.assert_allclose(a["velocities"], b["velocities"], atol=1e-8)


def test_errors(potentials):  # noqa: F811
    eam, lmp = potentials
    system, bottom = knocked()
    for wrong, match in (
        ({"max_distance": 0.0}, "max_distance"),
        ({"max_distance": 0.05, "min_timestep": -1.0}, "min_timestep"),
        ({"max_distance": 0.05, "max_timestep": np.nan}, "max_timestep"),
        ({"max_distance": 0.05, "min_timestep": 0.002, "max_timestep": 0.001}, "less"),
        ({"max_distance": 0.05, "every": 0}, "every"),
        ({"max_distance": True}, "max_distance"),
    ):
        with pytest.raises(InvalidParameter, match=match):
            AdaptiveTimestep(**wrong)
    assert repr(AdaptiveTimestep(0.05)).startswith("AdaptiveTimestep(max_distance=0.05")
    adaptive = AdaptiveTimestep(0.05)
    for potential in (eam, lmp):
        if potential is None:
            continue
        with MolecularDynamics(system, potential) as md:
            with pytest.raises(InvalidParameter, match="one AdaptiveTimestep"):
                md.run(1, controls=[adaptive, AdaptiveTimestep(0.1)])
            with pytest.raises(InvalidParameter, match="with a Deform"):
                md.run(1, controls=[adaptive, Deform(xx=0.1)])
            with pytest.raises(InvalidParameter, match="with moved atoms"):
                md.run(1, controls=[adaptive, Move(bottom, (0.1, 0, 0))])
            # with a bath it runs
            md.run(5, controls=[adaptive, Langevin(bottom, 300.0, seed=2)])
