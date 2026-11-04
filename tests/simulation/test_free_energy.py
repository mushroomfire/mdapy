# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Free energies against closed forms: the classical harmonic crystal, the mass
of the atoms, the second virial coefficient of dilute gases. Skipped where LAMMPS
or its packages EXTRA-FIX and EXTRA-PAIR are missing."""

import math

import numpy as np
import pytest

import mdapy
from mdapy import units
from mdapy.build import build_crystal
from mdapy.calculators import LAMMPS
from mdapy.core.threads import CancelToken
from mdapy.elements import atomic_masses
from mdapy.errors import Cancelled, InvalidParameter, MissingProperty
from mdapy.lattice_dynamics import phonons, primitive_cell
from mdapy.simulation import FreeEnergyResult, free_energy, minimize
from mdapy.simulation import _free_energy as module

pytest.importorskip("lammps")

EPSILON, SIGMA, CUT = 0.0104, 3.4, 8.5
A = 5.26
FAST = dict(timestep=0.002, equilibration=2.0, switching=4.0, scan=8.0, seed=3)


def lennard_jones(*elements):
    return LAMMPS(
        f"pair_style lj/cut {CUT}\npair_coeff * * {EPSILON} {SIGMA}\npair_modify shift yes",
        elements or ["Ar"],
    )


@pytest.fixture(scope="module")
def argon():
    potential = lennard_jones()
    start = build_crystal("fcc", A, elements="Ar", repeat=(3, 3, 3))
    result = minimize(
        start, potential, cell="full", max_stress=1e-4, max_force=1e-5, max_steps=5000
    )
    assert result.converged
    return result.system


def classical_harmonic(cell, potential, temperature):
    """The energy of the crystal at rest and the free energy of its classical
    harmonic vibrations, per atom, from its phonons."""
    n = len(cell)
    result = phonons(cell, potential, supercell=4, symmetry=False)
    grid = (np.arange(16) + 0.5) / 16
    q = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), -1).reshape(-1, 3)
    frequencies = result.frequencies(q).ravel()
    kt = units.BOLTZMANN * temperature
    vibrations = 3 * kt * np.log(units.PLANCK * frequencies / kt).mean()
    return potential.compute(cell).energy / n, vibrations


def test_two_atoms_against_the_integral_over_their_positions():
    """Two atoms need no assumption about a crystal: their partition function is an
    integral over their separation, with every term of finite size in it. The center
    of mass with 1/V instead of N/V would be off by kT/N ln N, here 1.2 meV."""
    epsilon, sigma, cut, temperature = 0.0104, 3.4, 8.5, 40.0
    potential = LAMMPS(
        f"pair_style lj/smooth/linear {cut}\npair_coeff * * {epsilon} {sigma}",
        ["Ar"],
        cutoff=cut,
    )
    cell = minimize(
        build_crystal("bcc", 4.2, elements="Ar"),
        potential,
        cell="full",
        max_stress=1e-6,
        max_force=1e-8,
        max_steps=3000,
    ).system
    a = cell.box.matrix[0, 0]

    def pair(r):
        def lj(x):
            return 4 * epsilon * ((sigma / x) ** 12 - (sigma / x) ** 6)

        slope = 4 * epsilon * (-12 * sigma**12 / cut**13 + 6 * sigma**6 / cut**7)
        return np.where(r < cut, lj(r) - lj(cut) - (r - cut) * slope, 0.0)

    reach = np.arange(-3, 4)
    images = np.stack(np.meshgrid(reach, reach, reach), -1).reshape(-1, 3) * a
    own = pair(np.linalg.norm(images[np.abs(images).sum(axis=1) > 0], axis=1)).sum()
    rest = potential.compute(cell).energy
    kt = units.BOLTZMANN * temperature
    grid = (np.arange(96) + 0.5) / 96 * a
    y, z = np.meshgrid(grid, grid, indexing="ij")
    total = 0.0
    for x in grid:
        separation = np.stack([np.full_like(y, x), y, z], -1)[:, :, None] + images
        energy = pair(np.linalg.norm(separation, axis=-1)).sum(axis=-1) + own
        total += np.exp(-(energy - rest) / kt).sum()
    integral = total * (a / 96) ** 3
    squared = units.PLANCK**2 / (
        2 * np.pi * atomic_masses(["Ar"])[0] * units.AMU_A2_PER_PS2_TO_EV * kt
    )
    expected = rest / 2 - kt / 2 * np.log(a**3 * integral / (2 * squared**3))
    counter = module._Progress(10**12, None, None)
    springs = module._spring_constants(
        cell, potential, temperature, 0.002, 5000, 3, counter
    )
    values = [
        module._frenkel_ladd(
            cell, potential, temperature, springs, 0.002, 2000, 15000, 11 + k, counter
        )[0]
        for k in range(6)
    ]
    assert np.mean(values) == pytest.approx(expected, abs=1.5e-4)


def test_a_cold_crystal_is_harmonic(argon):
    potential = lennard_jones()
    temperature = 10.0
    cell = primitive_cell(
        build_crystal("fcc", argon.box.matrix[0, 0] / 3, elements="Ar")
    )
    rest, vibrations = classical_harmonic(cell, potential, temperature)
    result = free_energy(argon, potential, temperature, pressure=None, **FAST)
    assert result.free_energy[0] == pytest.approx(rest + vibrations, abs=3e-4)
    assert result.pressure is None
    assert abs(result.dissipation[0]) < 1e-3
    # energy: the crystal at rest and 3 kT of vibration
    kt = units.BOLTZMANN * temperature
    assert result.enthalpy[0] == pytest.approx(rest + 3 * kt, abs=3e-4)
    assert not result.free_energy.flags.writeable


def test_the_free_energy_of_a_solid_at_a_pressure_holds_the_volume(argon):
    result = free_energy(argon, lennard_jones(), 40.0, **FAST)
    assert result.pressure == 0.0
    assert result.volume[0] > argon.box.volume / len(argon)
    high = free_energy(argon, lennard_jones(), 40.0, pressure=0.5, **FAST)
    # G grows by about the integral of the volume over the pressure
    gain = 0.5 * units.GPA_TO_EV_PER_A3 * 0.5 * (result.volume[0] + high.volume[0])
    assert high.free_energy[0] - result.free_energy[0] == pytest.approx(gain, rel=0.2)
    assert high.system.box.volume < result.system.box.volume


def test_the_mass_of_the_atoms(argon):
    """Classical atoms: the mass changes the free energy only by its logarithm."""
    temperature = 30.0
    light = free_energy(argon, lennard_jones("Ar"), temperature, pressure=None, **FAST)
    heavy = free_energy(
        argon.with_columns(element=["Kr"] * len(argon)),
        lennard_jones("Kr"),
        temperature,
        pressure=None,
        **FAST,
    )
    masses = mdapy.elements.atomic_masses(["Ar", "Kr"])
    expected = -1.5 * units.BOLTZMANN * temperature * math.log(masses[1] / masses[0])
    assert heavy.free_energy[0] - light.free_energy[0] == pytest.approx(
        expected, abs=4e-4
    )


def test_following_the_temperature_agrees_with_starting_there(argon):
    potential = lennard_jones()
    options = dict(FAST, switching=8.0, scan=40.0)
    scan = free_energy(argon, potential, [20.0, 40.0, 60.0], **options)
    direct = free_energy(argon, potential, 60.0, **options)
    assert scan.free_energy[-1] == pytest.approx(direct.free_energy[0], abs=1e-3)
    assert (np.diff(scan.free_energy) < 0).all()
    assert (np.diff(scan.entropy) > 0).all()
    np.testing.assert_allclose(
        scan.free_energy,
        scan.enthalpy - scan.temperatures * scan.entropy,
        atol=1e-12,
    )
    # the slope of the free energy is minus the entropy
    slope = (scan.free_energy[2] - scan.free_energy[0]) / 40.0
    assert slope == pytest.approx(-scan.entropy[1], rel=0.1)
    assert scan.volume[0] < scan.volume[-1]
    assert scan.temperatures.tolist() == [20.0, 40.0, 60.0]


def test_where_a_scan_changes_phase():
    grid = np.linspace(1000.0, 2000.0, 2001)
    kt = units.BOLTZMANN * 1000.0
    there = 1e-4 * grid  # nearly level: only the step leaves the line
    # nothing happens, whatever the noise
    rng = np.random.default_rng(1)
    noisy = there + rng.normal(0, 0.05 * kt, len(grid))
    assert module._phase_change(grid, noisy, there) is None
    # it melts at 1500 K, and stays melted on the way back
    melting = there + np.where(grid > 1500, 0.2, 0.0)
    assert module._phase_change(grid, melting, there + 0.2) == pytest.approx(
        1500, abs=25
    )
    # a difference of the two ways, but no step on the way there
    assert module._phase_change(grid, there, there + 0.2) == grid[-1]


def test_a_scan_past_melting_leaves_the_rest_out(argon):
    """A crystal heated far above its melting point, in the scan."""
    potential = lennard_jones()
    with pytest.warns(UserWarning, match="left out"):
        result = free_energy(
            argon, potential, [40.0, 60.0, 400.0], **dict(FAST, scan=20.0)
        )
    assert result.temperatures.tolist() == [40.0, 60.0]
    assert len(result.free_energy) == 2 and (result.dissipation < 0.01).all()


def test_mixing_adds_the_ideal_entropy(argon):
    kinds = np.array(["Ar", "Kr"])[np.arange(len(argon)) % 2]
    alloy = argon.with_columns(element=kinds)
    potential = lennard_jones("Ar", "Kr")
    plain = free_energy(alloy, potential, 30.0, pressure=None, **FAST)
    mixed = free_energy(alloy, potential, 30.0, pressure=None, mixing=True, **FAST)
    assert mixed.free_energy[0] - plain.free_energy[0] == pytest.approx(
        -units.BOLTZMANN * 30.0 * math.log(2), rel=1e-9
    )


def test_repeats_give_an_error(argon):
    one = free_energy(argon, lennard_jones(), 30.0, pressure=None, **FAST)
    three = free_energy(argon, lennard_jones(), 30.0, pressure=None, repeats=3, **FAST)
    assert one.error is None
    assert 0 < three.error < 1e-3


def second_virial(potential_over_kt, cutoff, points=200001):
    """1/2 of the integral of 1 - exp(-U / kT) over space, in Å³."""
    r = np.linspace(1e-6, cutoff, points)
    return 0.5 * np.trapezoid((1 - np.exp(-potential_over_kt(r))) * 4 * np.pi * r**2, r)


def gas(element, density, repeats=5):
    """Atoms on a simple cubic lattice of this density, from where they fly."""
    return build_crystal(
        "sc", density ** (-1 / 3), elements=element, repeat=[repeats] * 3
    )


def test_the_uhlenbeck_ford_model_as_a_dilute_gas():
    temperature = 1000.0
    density = 5e-4  # atoms per Å³
    system = gas("Ar", density)
    counter = module._Progress(1, None, None)
    excess, _ = module._model_excess(
        system, lennard_jones(), temperature, 0.001, 2000, 20000, 1, counter
    )

    def model(r):
        return -module._UF_STRENGTH * np.log1p(-np.exp(-((r / module._UF_LENGTH) ** 2)))

    virial = second_virial(model, module._UF_CUTOFF)
    expected = virial * density * units.BOLTZMANN * temperature
    assert excess == pytest.approx(expected, abs=4e-4)


def test_a_dilute_liquid_is_a_gas():
    """The whole path of a liquid, with the second virial coefficient."""
    temperature, density = 300.0, 1.2e-3
    system = gas("Ar", density)
    potential = lennard_jones()
    result = free_energy(
        system,
        potential,
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
    assert result.phase == "liquid"


def test_the_ideal_gas():
    system = gas("Ar", 0.01, repeats=4)
    kt = units.BOLTZMANN * 500.0
    mass = mdapy.elements.atomic_masses(["Ar"])[0]
    wavelength = units.PLANCK / math.sqrt(
        2 * math.pi * mass * units.AMU_A2_PER_PS2_TO_EV * kt
    )
    expected = kt * (
        3 * math.log(wavelength) - math.log(system.box.volume) + math.lgamma(65) / 64
    )
    assert module._ideal_gas(system, 500.0) == pytest.approx(expected)


def test_the_springs_in_closed_form(argon):
    n = len(argon)
    springs = np.full(n, 2.0)
    kt = units.BOLTZMANN * 100.0
    mass = mdapy.elements.atomic_masses(["Ar"])[0]
    omega = math.sqrt(2.0 / (mass * units.AMU_A2_PER_PS2_TO_EV))
    einstein, fixed = module._einstein(argon, 100.0, springs)
    assert einstein == pytest.approx(
        3 * kt * math.log(units.PLANCK / (2 * math.pi) * omega / kt)
    )
    assert fixed == pytest.approx(
        kt / n * math.log(n / argon.box.volume * (2 * math.pi * kt / (n * 2.0)) ** 1.5)
    )


def test_the_scaled_pair_commands():
    potential = LAMMPS(
        "pair_style lj/cut 8.0\npair_coeff * * 0.01 3.4\npair_modify shift yes", ["Ar"]
    )
    commands, style = module._scaled_pair(potential, 6.0, 2.5)
    assert style == "lj/cut"
    assert commands.splitlines() == [
        "pair_style hybrid/scaled v_mdapy_a lj/cut 8.0 v_mdapy_b ufm 6.0",
        "pair_coeff * * lj/cut 0.01 3.4",
        "pair_modify shift yes",
        f"pair_coeff * * ufm 2.5 {module._UF_LENGTH}",
    ]


def test_a_solid_is_no_liquid(argon):
    with pytest.raises(InvalidParameter, match="do not move as in a liquid"):
        free_energy(
            argon,
            lennard_jones(),
            10.0,
            phase="liquid",
            melt_temperature=10.0,
            timestep=0.002,
            equilibration=1.0,
            switching=1.0,
        )


def result_with(temperatures, free, enthalpy=None, volume=None, pressure=0.0):
    temperatures, free = np.array(temperatures, float), np.array(free, float)
    enthalpy = free if enthalpy is None else np.array(enthalpy, float)
    return FreeEnergyResult(
        temperatures=temperatures,
        pressure=pressure,
        free_energy=free,
        enthalpy=enthalpy,
        entropy=(enthalpy - free) / temperatures,
        volume=np.ones_like(free) if volume is None else np.array(volume, float),
        phase="solid",
        error=None,
        dissipation=np.zeros_like(free),
        system=None,
    )


def test_the_crossing_of_two_phases():
    solid = result_with([500, 1000, 1500], [-1.0, -1.1, -1.2], volume=[10, 10, 10])
    liquid = result_with([1500, 1000], [-1.3, -1.0], volume=[12, 12])
    found = solid.crossing(liquid)
    # the liquid: -1.0 - 0.3 (T - 1000) / 500; the solid: -1.1 - 0.1 (T - 1000) / 500
    assert found.temperature == pytest.approx(1250.0, rel=1e-6)
    assert found.volume == pytest.approx(2.0)
    assert found.entropy == pytest.approx(found.latent_heat / found.temperature)
    with pytest.raises(InvalidParameter, match="same pressure"):
        solid.crossing(result_with([500, 1500], [-1, -1], pressure=1.0))
    with pytest.raises(InvalidParameter, match="overlap"):
        solid.crossing(result_with([2000, 3000], [-1, -1]))
    with pytest.raises(InvalidParameter, match="do not cross"):
        solid.crossing(result_with([500, 1500], [0, 0]))


def test_progress_and_cancel(argon):
    potential = lennard_jones()
    seen = []
    options = dict(FAST, equilibration=0.1, switching=0.1, scan=0.1)
    free_energy(
        argon,
        potential,
        [20.0, 30.0],
        progress=lambda d, t: seen.append((d, t)),
        **options,
    )
    assert seen[-1][0] == seen[-1][1]
    assert all(a[0] <= b[0] for a, b in zip(seen, seen[1:]))
    token = CancelToken()
    token.cancel()
    with pytest.raises(Cancelled):
        free_energy(argon, potential, 20.0, cancel=token, **options)


def test_invalid_input(argon, tmp_path):
    potential = lennard_jones()
    for wrong, match in (
        (dict(phase="gas"), "phase"),
        (dict(timestep=0.0), "positive"),
        (dict(switching=-1.0), "positive"),
        (dict(equilibration=0.0), "positive"),
        (dict(scan=0.0), "positive"),
        (dict(repeats=0), "positive"),
        (dict(seed=0), "positive"),
        (dict(phase="liquid", mixing=True), "mixing"),
        (dict(melt_temperature=0.0), "positive"),
    ):
        with pytest.raises(InvalidParameter, match=match):
            free_energy(argon, potential, 20.0, **wrong)
    for temperatures in (0.0, -1.0, [20.0, 40.0, 30.0], [[20.0]]):
        with pytest.raises(InvalidParameter, match="temperatures"):
            free_energy(argon, potential, temperatures)
    with pytest.raises(InvalidParameter, match="LAMMPS"):
        free_energy(argon, object(), 20.0)
    with pytest.raises(MissingProperty, match="element"):
        free_energy(argon.drop("element"), potential, 20.0)
    with pytest.raises(InvalidParameter, match="periodic"):
        free_energy(
            argon.with_box(mdapy.Box(argon.box.matrix, pbc=(True, True, False))),
            potential,
            20.0,
        )
    hybrid = LAMMPS(
        "pair_style hybrid/overlay lj/cut 8.0 zero 8.0\n"
        "pair_coeff * * lj/cut 0.01 3.4\npair_coeff * * zero",
        ["Ar"],
    )
    with pytest.raises(InvalidParameter, match="hybrid"):
        free_energy(argon, hybrid, 20.0, phase="liquid")
