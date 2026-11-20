# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The free energy in MDAPY's own dynamics, through Switch, Springs and the
Uhlenbeck-Ford model: a cold crystal is harmonic, a dilute liquid a gas with
its second virial coefficient, and the solid and the liquid of LAMMPS where
it is installed."""

import numpy as np
import pytest

from mdapy import units
from mdapy.build import build_crystal
from mdapy.calculators import LAMMPS, LennardJones, _gpu
from mdapy.core.threads import CancelToken
from mdapy.errors import Cancelled
from mdapy.properties.phonons import phonons, primitive_cell
from mdapy.properties.thermodynamics import free_energy
from mdapy.simulation import _free_energy as module
from mdapy.simulation import minimize

from calculators.gpu_stand_in import stand_in

EPSILON, SIGMA, CUT = 0.0104, 3.4, 8.5
A = 5.26
FAST = dict(timestep=0.002, equilibration=2.0, switching=4.0, scan=8.0, seed=3)


def lennard_jones():
    return LennardJones({"Ar": (EPSILON, SIGMA)}, CUT, shift=True)


@pytest.fixture(scope="module")
def argon():
    start = build_crystal("fcc", A, elements="Ar", repeat=(3, 3, 3))
    result = minimize(
        start,
        lennard_jones(),
        cell="full",
        max_stress=1e-4,
        max_force=1e-5,
        max_steps=5000,
    )
    assert result.converged
    return result.system


def test_a_cold_crystal_is_harmonic(argon):
    potential = lennard_jones()
    temperature = 10.0
    cell = primitive_cell(
        build_crystal("fcc", argon.box.matrix[0, 0] / 3, elements="Ar")
    )
    n = len(cell)
    result = phonons(cell, potential, supercell=4, symmetry=False)
    grid = (np.arange(16) + 0.5) / 16
    q = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), -1).reshape(-1, 3)
    kt = units.BOLTZMANN * temperature
    vibrations = 3 * kt * np.log(units.PLANCK * result.frequencies(q).ravel() / kt)
    rest = potential.compute(cell).energy / n
    found = free_energy(argon, potential, temperature, pressure=None, **FAST)
    assert found.free_energy[0] == pytest.approx(rest + vibrations.mean(), abs=3e-4)
    assert abs(found.dissipation[0]) < 1e-3
    assert found.enthalpy[0] == pytest.approx(rest + 3 * kt, abs=3e-4)


def second_virial(potential_over_kt, cutoff, points=200001):
    """1/2 of the integral of 1 - exp(-U / kT) over space, in Å³."""
    r = np.linspace(1e-6, cutoff, points)
    return 0.5 * np.trapezoid((1 - np.exp(-potential_over_kt(r))) * 4 * np.pi * r**2, r)


def test_a_dilute_liquid_is_a_gas():
    """The whole path of a liquid, through the Uhlenbeck-Ford model and the
    ideal gas, with the second virial coefficient."""
    temperature, density = 300.0, 1.2e-3
    system = build_crystal("sc", density ** (-1 / 3), elements="Ar", repeat=[5] * 3)
    result = free_energy(
        system,
        lennard_jones(),
        temperature,
        phase="liquid",
        pressure=None,
        melt_temperature=temperature,
        timestep=0.002,
        equilibration=4.0,
        switching=8.0,
        seed=4,
    )
    kt = units.BOLTZMANN * temperature

    def interaction(r):
        s6, c6 = (SIGMA / r) ** 6, (SIGMA / CUT) ** 6
        shifted = (s6 * s6 - s6) - (c6 * c6 - c6)  # as pair_modify shift yes
        return np.where(r < CUT, 4 * EPSILON * shifted / kt, 0.0)

    virial = second_virial(interaction, CUT)
    expected = module._ideal_gas(system, temperature) + virial * density * kt
    assert result.free_energy[0] == pytest.approx(expected, abs=1e-3)


def test_the_solid_of_lammps(argon):
    """The same crystal at 40 K and zero pressure in either engine, within
    the noise of short switchings."""
    pytest.importorskip("lammps")
    lammps = LAMMPS(
        f"pair_style lj/cut {CUT}\npair_coeff * * {EPSILON} {SIGMA}\n"
        "pair_modify shift yes",
        ["Ar"],
    )
    ours = free_energy(argon, lennard_jones(), 40.0, **FAST)
    theirs = free_energy(argon, lammps, 40.0, **FAST)
    assert ours.free_energy[0] == pytest.approx(theirs.free_energy[0], abs=1.5e-3)
    assert ours.volume[0] == pytest.approx(theirs.volume[0], rel=5e-3)


def test_progress_cancel_and_errors(argon, monkeypatch):
    seen = []
    options = dict(FAST, equilibration=0.1, switching=0.2)
    free_energy(
        argon,
        lennard_jones(),
        20.0,
        pressure=None,
        progress=lambda done, total: seen.append(done),
        **options,
    )
    assert len(seen) > 4 and seen == sorted(seen)
    token = CancelToken()
    with pytest.raises(Cancelled):
        free_energy(
            argon,
            lennard_jones(),
            20.0,
            pressure=None,
            cancel=token,
            progress=lambda done, total: token.cancel(),
            **options,
        )


def test_on_the_gpu(monkeypatch):
    """The whole path of a liquid and of a solid on the GPU, its module stood
    in for on the processor: the dilute liquid its gas, the solid near that
    of the processor."""
    monkeypatch.setattr(_gpu, "cuda", stand_in)
    gpu = LennardJones({"Ar": (EPSILON, SIGMA)}, CUT, shift=True, device="cuda")
    temperature, density = 300.0, 1.2e-3
    dilute = build_crystal("sc", density ** (-1 / 3), elements="Ar", repeat=[5] * 3)
    liquid = free_energy(
        dilute,
        gpu,
        temperature,
        phase="liquid",
        pressure=None,
        melt_temperature=temperature,
        timestep=0.002,
        equilibration=4.0,
        switching=8.0,
        seed=4,
    )
    kt = units.BOLTZMANN * temperature

    def interaction(r):
        s6, c6 = (SIGMA / r) ** 6, (SIGMA / CUT) ** 6
        return np.where(
            r < CUT, 4 * EPSILON * ((s6 * s6 - s6) - (c6 * c6 - c6)) / kt, 0.0
        )

    expected = module._ideal_gas(dilute, temperature) + (
        second_virial(interaction, CUT) * density * kt
    )
    assert liquid.free_energy[0] == pytest.approx(expected, abs=1e-3)
    crystal = build_crystal("fcc", A, elements="Ar", repeat=(4, 4, 4))
    cpu = free_energy(crystal, lennard_jones(), 30.0, pressure=None, **FAST)
    on = free_energy(crystal, gpu, 30.0, pressure=None, **FAST)
    assert on.free_energy[0] == pytest.approx(cpu.free_energy[0], abs=2e-3)
