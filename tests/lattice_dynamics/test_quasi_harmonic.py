# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import dataclasses

import numpy as np
import pytest

from mdapy import units
from mdapy.build import build_crystal
from mdapy.calculators import EAM
from mdapy.errors import InvalidParameter
from mdapy.lattice_dynamics import elastic_constants, primitive_cell, quasi_harmonic
from mdapy.lattice_dynamics._quasi_harmonic import _Fit
from mdapy.simulation import minimize

from calculators.eam_files import write_setfl
from calculators.test_testing import Harmonic

OPTIONS = {"supercell": 4, "mesh": 10}


@pytest.fixture(scope="module")
def copper(tmp_path_factory):
    """fcc Cu of the synthetic potential at zero stress: its potential, its
    conventional cell and its primitive cell."""
    potential = EAM(write_setfl(tmp_path_factory.mktemp("eam") / "CuNi.eam.alloy"))
    relaxed = minimize(
        build_crystal("fcc", 3.29, elements="Cu"),
        potential,
        cell="full",
        max_force=1e-10,
        max_stress=1e-8,
        max_steps=10000,
    ).system
    return potential, relaxed, primitive_cell(relaxed)


@pytest.fixture(scope="module")
def heating(copper):
    potential, _, cell = copper
    temperatures = np.array([0.0, 99.0, 100.0, 101.0, 299.0, 300.0, 301.0, 600.0])
    return quasi_harmonic(cell, potential, temperatures, **OPTIONS)


def test_the_crystal_expands_and_softens(heating):
    assert heating.temperatures.shape == (8,) and heating.pressure == 0.0
    assert np.all(np.diff(heating.volume) > 0)
    assert np.all(np.diff(heating.bulk_modulus) < 0)
    assert np.all(np.diff(heating.gibbs_energy) < 0)
    assert heating.thermal_expansion[0] == 0 and heating.entropy[0] == 0
    assert np.all(heating.thermal_expansion[1:] > 0)
    assert heating.volumes.shape == (9,) and heating.free_energies.shape == (9, 8)
    assert not heating.volume.flags.writeable
    # the zero-point vibrations expand it a little beyond the static minimum
    static = heating.volumes[np.argmin(heating.energies)]
    assert heating.volume[0] > static * 0.99


@pytest.mark.parametrize("t", [100.0, 300.0])
def test_thermodynamic_identities(heating, t):
    k = list(heating.temperatures).index(t)
    g, v = heating.gibbs_energy, heating.volume
    # S = -dG/dT, alpha = dV/dT / V, C_P = -T d2G/dT2, by central differences
    assert -(g[k + 1] - g[k - 1]) / 2 == pytest.approx(heating.entropy[k], rel=1e-4)
    assert (v[k + 1] - v[k - 1]) / 2 / v[k] == pytest.approx(
        heating.thermal_expansion[k], rel=2e-2
    )
    assert -t * (g[k + 1] - 2 * g[k] + g[k - 1]) == pytest.approx(
        heating.heat_capacity[k], rel=2e-2
    )


def test_the_static_bulk_modulus(copper, heating):
    # the curvature of the static energy is the bulk modulus of the elastic
    # constants, and the zero-point vibrations soften the crystal
    potential, conventional, _ = copper
    c = elastic_constants(conventional, potential).constants
    expected = (c[0, 0] + 2 * c[0, 1]) / 3
    volume, curvature = _Fit(heating.volumes, heating.energies).minimum()
    static = volume * curvature * units.EV_PER_A3_TO_GPA
    assert static == pytest.approx(expected, rel=5e-3)
    assert heating.bulk_modulus[0] < static


def test_pressure(copper, heating):
    # dG/dP = V
    potential, _, cell = copper
    h = 0.5
    pressed = [
        quasi_harmonic(cell, potential, [300.0], pressure=p, **OPTIONS) for p in (-h, h)
    ]
    assert pressed[0].pressure == -h
    assert pressed[1].volume[0] < pressed[0].volume[0]
    k = list(heating.temperatures).index(300.0)
    slope = (pressed[1].gibbs_energy[0] - pressed[0].gibbs_energy[0]) / (
        2 * h * units.GPA_TO_EV_PER_A3
    )
    assert slope == pytest.approx(heating.volume[k], rel=1e-3)


def test_invalid_arguments(copper):
    potential, conventional, cell = copper
    with pytest.raises(InvalidParameter, match="not be negative"):
        quasi_harmonic(cell, potential, [-1.0])
    with pytest.raises(InvalidParameter, match="five or more"):
        quasi_harmonic(cell, potential, [300.0], volumes=[0.99, 1.0, 1.01])
    with pytest.raises(InvalidParameter, match="outside the volumes"):
        quasi_harmonic(
            cell, potential, [0.0], volumes=np.linspace(1.04, 1.08, 5), **OPTIONS
        )
    shaken = conventional.with_columns(
        positions=conventional["positions"]
        + np.random.default_rng(0).normal(scale=0.05, size=(4, 3))
    )
    with pytest.raises(InvalidParameter, match="did not relax"):
        quasi_harmonic(shaken, potential, [300.0], max_steps=1, **OPTIONS)


def test_unstable_volumes():
    # springs under compression at every volume: imaginary frequencies
    cell = build_crystal("sc", 2.5, elements="Ar")
    with pytest.raises(InvalidParameter, match="at volume ratio .*imaginary"):
        quasi_harmonic(cell, Harmonic(k=1.0, r0=3.0), [300.0], supercell=3, mesh=4)


def test_fit_without_a_minimum():
    volumes = np.linspace(10.0, 12.0, 7)
    t = volumes ** (-2 / 3)
    assert _Fit(volumes, t + t**3).minimum() is None  # no real stationary point
    assert _Fit(volumes, -((t - t.mean()) ** 2)).minimum() is None  # a maximum


def test_pressures_and_volumes(copper, heating):
    potential, _, cell = copper
    # at each volume of the grid the shape relaxes: a hydrostatic stress
    np.testing.assert_allclose(
        heating.volumes / heating.volumes[0],
        np.linspace(0.97, 1.09, 9) / 0.97,
        rtol=1e-6,
    )
    for grid_cell in heating.cells:
        stress = potential.compute(grid_cell, ("virial",)).stress(grid_cell)
        np.testing.assert_allclose(stress, stress[0, 0] * np.eye(3), atol=1e-3)
    # the cells of given pressures are relaxed under them
    pressures = np.linspace(20.0, -40.0, 6)
    given = quasi_harmonic(cell, potential, [300.0], pressures=pressures, **OPTIONS)
    assert len(given.cells) == 6 and np.all(np.diff(given.volumes) > 0)
    for p, grid_cell in zip(sorted(pressures, reverse=True), given.cells):
        stress = potential.compute(grid_cell, ("virial",)).stress(grid_cell)
        np.testing.assert_allclose(stress, -p * np.eye(3), atol=1e-3)
    with pytest.raises(InvalidParameter, match="not both"):
        quasi_harmonic(
            cell,
            potential,
            [300.0],
            pressures=pressures,
            volumes=[1, 1.01, 1.02, 1.03, 1.04],
        )
    with pytest.raises(InvalidParameter, match="five or more numbers"):
        quasi_harmonic(cell, potential, [300.0], pressures=[0.0, 1.0])


def test_the_shape_relaxes_with_the_volume(copper):
    # hcp: c/a changes with the volume, the stress stays hydrostatic
    potential, _, _ = copper
    hcp = minimize(
        build_crystal("hcp", 2.33, elements="Cu"),
        potential,
        cell="full",
        max_force=1e-10,
        max_stress=1e-8,
        max_steps=20000,
    ).system
    result = quasi_harmonic(hcp, potential, [100.0], **OPTIONS)
    ratios = [
        c.box.matrix[2, 2] / np.linalg.norm(c.box.matrix[0]) for c in result.cells
    ]
    assert np.ptp(ratios) > 1e-3
    for grid_cell in result.cells:
        stress = potential.compute(grid_cell, ("virial",)).stress(grid_cell)
        np.testing.assert_allclose(stress, stress[0, 0] * np.eye(3), atol=1e-3)


def test_relaxation_that_does_not_converge(copper):
    potential, conventional, _ = copper
    shaken = conventional.with_columns(
        positions=conventional["positions"]
        + np.random.default_rng(1).normal(scale=0.05, size=(4, 3))
    )
    with pytest.raises(InvalidParameter, match="at volume ratio .* did not relax"):
        quasi_harmonic(shaken, potential, [300.0], max_steps=1, **OPTIONS)


def test_a_volume_that_is_not_kept(copper, monkeypatch):
    from mdapy.lattice_dynamics import _quasi_harmonic as module

    potential, _, cell = copper
    relax = module._relaxed

    def drifting(*args):  # a relaxation that always ends at another volume
        result = relax(*args)
        return dataclasses.replace(result, system=module._scaled(result.system, 1.001))

    monkeypatch.setattr(module, "_relaxed", drifting)
    with pytest.raises(InvalidParameter, match="did not keep its volume"):
        quasi_harmonic(cell, potential, [100.0], **OPTIONS)
