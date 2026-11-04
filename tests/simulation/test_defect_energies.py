# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import pytest

import mdapy
from mdapy.build import build_crystal
from mdapy.calculators import EAM
from mdapy.errors import InvalidParameter
from mdapy.simulation import (
    minimize,
    stacking_fault_energy,
    surface_energy,
    vacancy_formation_energy,
)

A = 3.615
FCC111 = [(1, 1, -2), (-1, 1, 0), (1, 1, 1)]


@pytest.fixture(scope="module")
def copper():
    return EAM.zhou(["Cu"])


def test_movable_atoms_stay(copper):
    crystal = build_crystal("fcc", A, elements="Cu", repeat=(3, 3, 3))
    moved = crystal.with_columns(
        positions=crystal["positions"]
        + np.random.default_rng(1).normal(scale=0.05, size=(108, 3))
    )
    movable = np.ones((108, 3), bool)
    movable[:10] = False
    movable[10:20, :2] = False
    result = minimize(moved.with_columns(movable=movable), copper, max_force=1e-4)
    assert result.converged
    np.testing.assert_array_equal(
        result.system["positions"][:10], moved["positions"][:10]
    )
    np.testing.assert_array_equal(
        result.system["positions"][10:20, :2], moved["positions"][10:20, :2]
    )
    assert (
        np.abs(result.system["positions"][20:] - moved["positions"][20:]).max() > 1e-3
    )


def test_vacancy(copper):
    crystal = build_crystal("fcc", A, elements="Cu", repeat=(4, 4, 4))
    result = vacancy_formation_energy(crystal, copper)
    assert 1.0 < result.energy < 1.5
    assert len(result.system) == len(crystal) - 1
    np.testing.assert_allclose(
        result.energy, result.defect - 255 / 256 * result.perfect
    )
    # every site of a perfect crystal alike
    other = vacancy_formation_energy(crystal, copper, atom=77)
    assert other.energy == pytest.approx(result.energy, abs=1e-6)
    relaxed_box = vacancy_formation_energy(
        crystal, copper, cell="isotropic", max_force=1e-3
    )
    assert relaxed_box.energy <= result.energy + 1e-4
    with pytest.raises(InvalidParameter, match="two atoms or more"):
        vacancy_formation_energy(crystal.gather([0]), copper)
    with pytest.raises(InvalidParameter, match="did not converge"):
        vacancy_formation_energy(crystal, copper, max_steps=1)


def test_surfaces(copper):
    cube = surface_energy(
        build_crystal("fcc", A, elements="Cu", repeat=(3, 3, 6)), copper
    )
    dense = surface_energy(
        build_crystal("fcc", A, elements="Cu", orientation=FCC111, repeat=(2, 3, 4)),
        copper,
    )
    assert (
        0.5 < dense.energy < cube.energy < 2.5
    )  # the close-packed surface costs least
    assert cube.area == pytest.approx(9 * A * A)
    assert cube.system.box.matrix[2, 2] == pytest.approx(6 * A + 15.0)
    side = surface_energy(
        build_crystal("fcc", A, elements="Cu", repeat=(6, 3, 3)), copper, axis="x"
    )
    assert side.energy == pytest.approx(cube.energy, rel=1e-6)
    open_box = cube.system.with_box(
        mdapy.Box(cube.system.box.matrix, pbc=(True, True, False))
    )
    with pytest.raises(InvalidParameter, match="periodic in every direction"):
        surface_energy(open_box, copper)
    with pytest.raises(InvalidParameter, match="vacuum must be positive"):
        surface_energy(build_crystal("fcc", A, elements="Cu"), copper, vacuum=0.0)


def test_stacking_faults(copper):
    layers = build_crystal(
        "fcc", A, elements="Cu", orientation=FCC111, repeat=(2, 3, 4)
    )
    # along [11-2]: the intrinsic fault at one third, a minimum
    curve = stacking_fault_energy(layers, copper, [A * np.sqrt(6) / 2, 0, 0], points=13)
    assert curve.energy[0] == 0 and abs(curve.energy[-1]) < 1e-6
    intrinsic = curve.energy[4]
    assert 0.0 < intrinsic < curve.energy[2] and intrinsic < curve.energy[6]
    np.testing.assert_allclose(curve.displacement[4], [A * np.sqrt(6) / 6, 0, 0])
    assert curve.area == pytest.approx(2 * A * np.sqrt(6) / 2 * 3 * A / np.sqrt(2))
    # along [-110], symmetric about the half
    straight = stacking_fault_energy(layers, copper, [0, A / np.sqrt(2), 0], points=9)
    np.testing.assert_allclose(straight.energy, straight.energy[::-1], atol=1e-6)
    rigid = stacking_fault_energy(
        layers, copper, [0, A / np.sqrt(2), 0], points=9, relax=False
    )
    assert (rigid.energy >= straight.energy - 1e-9).all()
    with pytest.raises(InvalidParameter, match="points must be at least 2"):
        stacking_fault_energy(layers, copper, [0, 1.0, 0], points=1)
    hexagonal = build_crystal("hcp", 2.5, elements="Cu", repeat=(3, 3, 3))
    with pytest.raises(InvalidParameter, match="normal to the Cartesian axis"):
        stacking_fault_energy(hexagonal, copper, [0, 0, 1.0], axis="x")


def _slab(structure, a, element, orientation):
    crystal = build_crystal(
        structure, a, elements=element, orientation=orientation, repeat=(1, 1, 4)
    )
    return crystal.with_box(mdapy.Box(crystal.box.matrix, pbc=(True, True, False)))


def test_the_gamma_surface_has_the_symmetry_of_its_plane():
    def energy(crystal, potential, x, y):
        lx, ly = crystal.box.lengths[:2]
        slipped = stacking_fault_energy(
            crystal, potential, [x * lx, y * ly, 0.0], points=2
        )
        return slipped.energy[1]

    copper = _slab("fcc", 3.615, "Cu", [(1, 1, -2), (-1, 1, 0), (1, 1, 1)])
    potential = EAM.zhou(["Cu"])
    # lattice vectors of {111} bring the crystal back onto itself
    for x, y in ((1.0, 0.0), (0.0, 1.0), (0.5, 0.5)):
        assert energy(copper, potential, x, y) == pytest.approx(0.0, abs=1e-9)
    # the two sites of the intrinsic fault, and a mirror across the slip direction
    fault = energy(copper, potential, 1 / 3, 0.0)
    assert fault > 0.0
    assert energy(copper, potential, 5 / 6, 0.5) == pytest.approx(fault, abs=1e-9)
    assert energy(copper, potential, 0.2, 0.3) == pytest.approx(
        energy(copper, potential, 0.2, 0.7), abs=1e-9
    )
    # the fault is a minimum among its neighbors along the direction of the partial
    assert energy(copper, potential, 1 / 6, 0.0) > fault
    assert energy(copper, potential, 1 / 2, 0.0) > fault

    tungsten = _slab("bcc", 3.165, "W", [(1, -1, 0), (0, 0, 1), (1, 1, 0)])
    potential = EAM.zhou(["W"])
    assert energy(tungsten, potential, 0.5, 0.5) == pytest.approx(0.0, abs=1e-9)
    # no stable fault along a/2 <111> of bcc: the energy rises up to its middle
    along = [energy(tungsten, potential, f / 2, f / 2) for f in np.linspace(0, 0.5, 7)]
    assert (np.diff(along) > 0).all()
