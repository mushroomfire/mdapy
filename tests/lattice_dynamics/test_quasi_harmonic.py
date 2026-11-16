# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import dataclasses

import numpy as np
import pytest

from mdapy import Box, System, units
from mdapy.build import build_crystal
from mdapy.calculators import EAM
from mdapy.errors import InvalidParameter
from mdapy.lattice_dynamics import elastic_constants, primitive_cell, quasi_harmonic
from mdapy.lattice_dynamics._phonons import _crossing, _cutoff_jump
from mdapy.lattice_dynamics._quasi_harmonic import _Fit, _neighbouring
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


@pytest.fixture(scope="module")
def truncated(tmp_path_factory):
    """fcc Cu of the synthetic potential cut off without smoothing, as many
    tabulated potentials are, at zero stress: its potential and its
    primitive cell."""
    path = tmp_path_factory.mktemp("eam") / "cut.eam.alloy"
    potential = EAM(write_setfl(path, abrupt=True))
    relaxed = minimize(
        build_crystal("fcc", 3.63, elements="Cu"),
        potential,
        cell="full",
        max_force=1e-10,
        max_stress=1e-8,
        max_steps=10000,
    ).system
    return potential, primitive_cell(relaxed)


def test_a_pair_that_crosses_the_cutoff(truncated):
    # the fourth shell, 5.14 Å at zero stress, comes inside the cutoff of 5 Å
    # below 0.92 of the volume, and the free energy jumps by most of its change
    # between neighbouring volumes; raised before the phonons of the grid
    potential, cell = truncated
    with pytest.raises(
        InvalidParameter,
        match=r"a pair of atoms crosses the cutoff of the calculator, 5 Å, between "
        r"the volumes: it is 4\.86\d\d Å apart at volume ratio 0\.85 and 5\.13\d\d Å "
        r"at volume ratio 1\. The free energy at 300 K jumps there by \d\.\d+ eV per "
        r"atom, \d\d% of its change between neighbouring volumes, more than the 2% "
        r"that the results tolerate: the potential is not smooth at its cutoff\. Use "
        r"volumes or pressures all on one side of the crossing, or a potential "
        r"smooth at its cutoff",
    ):
        quasi_harmonic(cell, potential, [300.0], volumes=np.linspace(0.85, 1.0, 5))


def test_a_potential_smooth_at_its_cutoff_crosses_it(copper):
    # the same crossing of a potential whose functions and their slopes vanish
    # at the cutoff: a jump of the free energy within the 2 %, computed
    potential, _, cell = copper
    middle = cell.with_box(cell.box.matrix * 0.925 ** (1 / 3), scale_positions=True)
    stress = potential.compute(middle, ["energy", "virial"]).stress
    result = quasi_harmonic(
        cell,
        potential,
        [0.0, 300.0],
        volumes=np.linspace(0.85, 1.0, 7),
        pressure=-np.trace(stress) / 3,
        **OPTIONS,
    )
    ratio = result.volume / cell.box.volume
    assert np.all((ratio > 0.87) & (ratio < 0.98))
    assert np.isfinite(result.thermal_expansion).all()


def test_zhou_potentials_cross_their_cutoff():
    # their cutoff is the fifth shell of their fcc crystal, which every range of
    # volumes about it crosses; their functions nearly vanish there: a jump of
    # about 1 % of the change between neighbouring volumes, computed
    potential = EAM.zhou(["Cu"])
    cell = primitive_cell(build_crystal("fcc", 3.615, elements="Cu"))
    volumes = np.linspace(0.97, 1.07, 7)
    result = quasi_harmonic(cell, potential, [0.0, 100.0], volumes=volumes)
    assert np.all(np.diff(result.volume) > 0)
    fifth = (3.615 * 1.5, 3.615 * 0.5, 0.0)  # a fifth neighbour, on the cutoff
    static, vibration, change = _cutoff_jump(
        cell, fifth, potential, None, 0.01, None, 100.0, 10, 1.015 ** (1 / 3) - 1
    )
    assert np.linalg.norm(fifth) == pytest.approx(potential.cutoff, rel=1e-4)
    assert 0 < static + vibration < 0.02 * change


def test_the_jump_at_the_cutoff_of_springs():
    # fcc of 3.2 Å with springs at rest between first neighbours, cut off at
    # 3.2 Å: the six second neighbours, three springs per atom, on the cutoff;
    # each holds k (rc - r0)^2 / 2 there, which the static energy loses
    rest = 3.2 / np.sqrt(2)
    springs = Harmonic(k=1.0, r0=rest)
    cell = primitive_cell(build_crystal("fcc", 3.2, elements="Ar"))
    static, vibration, change = _cutoff_jump(
        cell, (3.2, 0.0, 0.0), springs, 4, 0.01, None, 300.0, 10, 0.005
    )
    assert static == pytest.approx(3 * 0.5 * (3.2 - rest) ** 2, rel=1e-3)
    assert vibration > 0 and static + vibration > 0.02 * change


def test_the_jump_at_the_cutoff_of_a_sheet():
    # a triangular sheet with its six second neighbours on the cutoff of the
    # springs, scaled in its plane; the springs of first neighbours a little
    # stretched, so that the sheet is stable across its plane
    a = 3.2 / np.sqrt(3)
    springs = Harmonic(k=1.0, r0=0.98 * a)
    box = Box([[a, 0, 0], [a / 2, a * np.sqrt(3) / 2, 0], [0, 0, 20.0]], pbc=[1, 1, 0])
    sheet = System([[0.0, 0.0, 10.0]], box, elements="Ar")
    second = (1.5 * a, a * np.sqrt(3) / 2, 0.0)
    static, vibration, change = _cutoff_jump(
        sheet, second, springs, 4, 0.01, None, 100.0, 8, 0.005
    )
    assert static == pytest.approx(3 * 0.5 * (3.2 - 0.98 * a) ** 2, rel=1e-3)
    assert vibration > 0 and change > 0


def test_neighbouring_volumes():
    sizes = np.array([1.1, 1.0, 1.21])  # in any order
    assert _neighbouring(sizes, 0, 3) == pytest.approx((1.21 ** (1 / 3) - 1) / 2)
    assert _neighbouring(sizes, 1, 3) == pytest.approx(1.1 ** (1 / 3) - 1)
    assert _neighbouring(sizes, 2, 2) == pytest.approx(1.1**0.5 - 1, rel=1e-6)


def test_crossing_follows_the_atoms_across_the_box():
    # fcc of 4 Å: the second shell at 4 Å, the third at 4.90 Å
    crystal = build_crystal("fcc", 4.0, elements="Cu", repeat=(2, 2, 2))
    cells = [
        crystal.with_box(crystal.box.matrix * s, scale_positions=True)
        for s in (0.99, 1.0, 1.01)
    ]
    found = _crossing(cells, 4.02, 0.0)
    assert (found.near, found.far) == (0, 2)
    assert (found.short, found.long) == pytest.approx((3.96, 4.04), rel=1e-12)
    # the cell where it is nearest the cutoff, and the pair there
    length = np.linalg.norm(found.vector)
    assert length == pytest.approx([3.96, 4.0, 4.04][found.closest], rel=1e-12)
    assert abs(length - 4.02) == pytest.approx(0.02, abs=1e-9)
    assert _crossing(cells, 4.5, 0.0) is None
    assert _crossing(cells, 3.9, 0.0) is None
    # give or take a displacement: the shell within it of the cutoff counts
    assert _crossing(cells, 4.045, 0.0) is None
    assert _crossing(cells, 4.045, 0.01)[:4] == found[:4]
    assert _crossing(cells, 3.955, 0.01)[:4] == found[:4]
    assert _crossing(cells, 3.945, 0.01) is None
    # an atom wrapped to the other side of the box in one cell
    positions = cells[1]["positions"].copy()
    positions[0] += cells[1].box.matrix[0]
    cells[1] = cells[1].with_columns(positions=positions)
    wrapped = _crossing(cells, 4.02, 0.0)
    assert wrapped[:5] == pytest.approx(found[:5], rel=1e-12)
    assert wrapped.vector == pytest.approx(found.vector, rel=1e-12)
    assert _crossing(cells, 4.5, 0.0) is None


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
        stress = potential.compute(grid_cell, ("virial",)).stress
        np.testing.assert_allclose(stress, stress[0, 0] * np.eye(3), atol=1e-3)
    # the cells of given pressures are relaxed under them
    pressures = np.linspace(20.0, -40.0, 6)
    given = quasi_harmonic(cell, potential, [300.0], pressures=pressures, **OPTIONS)
    assert len(given.cells) == 6 and np.all(np.diff(given.volumes) > 0)
    for p, grid_cell in zip(sorted(pressures, reverse=True), given.cells):
        stress = potential.compute(grid_cell, ("virial",)).stress
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
        stress = potential.compute(grid_cell, ("virial",)).stress
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
