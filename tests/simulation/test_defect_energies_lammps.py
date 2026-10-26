# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Defect energies against LAMMPS with the same EAM table. Skipped where LAMMPS
is not installed."""

import numpy as np
import pytest

import mdapy
from mdapy import units
from mdapy.build import add_vacancies, add_vacuum, build_crystal, slip
from mdapy.calculators import EAM
from mdapy.simulation import (
    stacking_fault_energy,
    surface_energy,
    vacancy_formation_energy,
)

lammps = pytest.importorskip("lammps")

A = 3.615


def lammps_energy(
    tmp_path, system, potential, *, relax=True, fixed_xy=False, name="in.data"
):
    """The energy that LAMMPS minimizes the system to with FIRE, the
    algorithm of mdapy, to a tight tolerance (the conjugate gradient may
    reach another minimum of a slab)."""
    data = tmp_path / name
    mdapy.write(data, system)
    boundary = " ".join("p" if p else "f" for p in system.box.pbc)
    lmp = lammps.lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
    try:
        lmp.commands_string(
            f"""
units metal
atom_style atomic
boundary {boundary}
read_data {data}
pair_style eam/alloy
pair_coeff * * {potential} Cu
thermo_style custom step pe
"""
        )
        if fixed_xy:
            lmp.command("fix plane all setforce 0.0 0.0 NULL")
        if relax:
            lmp.command("min_style fire")
            lmp.command("minimize 0.0 1.0e-8 100000 1000000")
        else:
            lmp.command("run 0")
        return lmp.get_thermo("pe")
    finally:
        lmp.close()


@pytest.fixture(scope="module")
def copper(tmp_path_factory):
    calculator = EAM.zhou(["Cu"])
    path = tmp_path_factory.mktemp("eam") / "Cu.eam.alloy"
    calculator.write(path)
    return calculator, path


def test_vacancy_as_lammps(tmp_path, copper):
    calculator, potential = copper
    crystal = build_crystal("fcc", A, elements="Cu", repeat=(4, 4, 4))
    ours = vacancy_formation_energy(crystal, calculator, max_force=1e-6)
    perfect = lammps_energy(tmp_path, crystal, potential, relax=False)
    defect = lammps_energy(
        tmp_path, add_vacancies(crystal, atoms=[0]), potential, name="v.data"
    )
    assert ours.energy == pytest.approx(defect - 255 / 256 * perfect, abs=1e-6)


def test_surface_as_lammps(tmp_path, copper):
    calculator, potential = copper
    crystal = build_crystal(
        "fcc",
        A,
        elements="Cu",
        orientation=[(1, 1, -2), (-1, 1, 0), (1, 1, 1)],
        repeat=(2, 3, 4),
    )
    ours = surface_energy(crystal, calculator, max_force=1e-6)
    bulk = lammps_energy(tmp_path, crystal, potential, relax=False)
    slab = lammps_energy(
        tmp_path, add_vacuum(crystal, 15.0), potential, name="slab.data"
    )
    expected = (slab - bulk) / (2 * ours.area) * units.EV_PER_A2_TO_J_PER_M2
    assert ours.energy == pytest.approx(expected, abs=1e-6)


def test_stacking_fault_as_lammps(tmp_path, copper):
    """Relaxed along the normal only, as with fix setforce 0 0 NULL."""
    calculator, potential = copper
    crystal = build_crystal(
        "fcc",
        A,
        elements="Cu",
        orientation=[(1, 1, -2), (-1, 1, 0), (1, 1, 1)],
        repeat=(2, 3, 4),
    )
    vector = np.array([A * np.sqrt(6) / 2, 0, 0])
    ours = stacking_fault_energy(crystal, calculator, vector, points=7, max_force=1e-6)
    energies = [
        lammps_energy(
            tmp_path,
            slip(crystal, f * vector),
            potential,
            fixed_xy=True,
            name=f"{k}.data",
        )
        for k, f in enumerate(ours.fraction)
    ]
    expected = (
        (np.array(energies) - energies[0]) / ours.area * units.EV_PER_A2_TO_J_PER_M2
    )
    np.testing.assert_allclose(ours.energy, expected, atol=1e-6)
