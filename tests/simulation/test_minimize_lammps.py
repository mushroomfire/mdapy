# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""minimize against LAMMPS min_style fire and fix box/relax, with EAM.
Skipped where LAMMPS is not installed."""

import numpy as np
import pytest

import mdapy
from mdapy.calculators import EAM
from mdapy.simulation import minimize

from calculators.eam_files import alloy, write_setfl

lammps = pytest.importorskip("lammps")


def test_relaxation_as_lammps(tmp_path):
    """FIRE in LAMMPS and in mdapy reach the same minimum from the same start:
    a vacancy in a slightly perturbed alloy near the potential's lattice
    constant."""
    potential_file = write_setfl(tmp_path / "CuNi.eam.alloy")
    full = alloy(seed=12, repeat=3, jitter=0.05, a=3.35)
    system = full.gather(np.arange(1, len(full))).with_columns(
        id=np.arange(1, len(full))
    )
    data, out = tmp_path / "in.data", tmp_path / "out.data"
    mdapy.write(data, system, elements=["Cu", "Ni"])
    lmp = lammps.lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
    try:
        lmp.commands_string(
            f"""
units metal
atom_style atomic
boundary p p p
read_data {data}
pair_style eam/alloy
pair_coeff * * {potential_file} Cu Ni
min_style fire
min_modify norm max
minimize 0.0 1.0e-8 100000 1000000
write_data {out}
"""
        )
        energy = lmp.get_thermo("pe")
    finally:
        lmp.close()
    relaxed = minimize(system, EAM(potential_file), max_force=1e-8, max_steps=100000)
    assert relaxed.converged
    assert relaxed.energy == pytest.approx(energy, abs=1e-8)
    back = mdapy.read(out)
    order = np.argsort(back.table["id"].to_numpy())
    step = system.box.minimum_image(relaxed.system.positions - back.positions[order])
    # a rigid translation costs nothing in a periodic box; each run drifts
    np.testing.assert_allclose(step - step.mean(axis=0), 0.0, atol=1e-6)
    assert np.abs(relaxed.system.positions - system.positions).max() > 0.01


@pytest.mark.parametrize("steps", [1, 3, 30, 120])
def test_steps_as_lammps(tmp_path, steps):
    """The positions after each number of steps are those of LAMMPS
    min_style fire with its default settings, with the masses of the
    potential file, which pair_coeff sets in LAMMPS; only the conversion
    constant of LAMMPS, rounded to 8 digits, differs."""
    potential_file = write_setfl(tmp_path / "CuNi.eam.alloy")
    potential = EAM(potential_file)
    start = alloy(seed=13, repeat=2, jitter=0.25)
    system = start.with_columns(id=np.arange(1, len(start) + 1))
    data, out = tmp_path / "in.data", tmp_path / "out.data"
    mdapy.write(data, system, elements=["Cu", "Ni"])
    lmp = lammps.lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
    try:
        lmp.commands_string(
            f"""
units metal
atom_style atomic
boundary p p p
read_data {data}
pair_style eam/alloy
pair_coeff * * {potential_file} Cu Ni
min_style fire
minimize 0.0 0.0 {steps} 1000000
write_data {out}
"""
        )
    finally:
        lmp.close()
    masses = dict(zip(potential.elements, potential.masses))
    weighted = system.with_columns(
        mass=[masses[e] for e in system.table["element"].to_list()]
    )
    result = minimize(weighted, potential, max_force=1e-12, max_steps=steps)
    assert result.steps == steps
    back = mdapy.read(out)
    order = np.argsort(back.table["id"].to_numpy())
    step = system.box.minimum_image(result.system.positions - back.positions[order])
    np.testing.assert_allclose(step, 0.0, atol=1e-7)
    assert np.abs(result.system.positions - system.positions).max() > 1e-5


def test_cell_relaxation_as_lammps(tmp_path):
    """A strained alloy relaxed with its box, by minimize and by LAMMPS fix
    box/relax tri: the same energy, volume and shape of the box (its metric,
    which does not change when LAMMPS rotates the box)."""
    potential_file = write_setfl(tmp_path / "CuNi.eam.alloy")
    system = alloy(seed=16, repeat=3, jitter=0.03, a=3.45)
    data = tmp_path / "in.data"
    mdapy.write(data, system, elements=["Cu", "Ni"])
    lmp = lammps.lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
    try:
        lmp.commands_string(
            f"""
units metal
atom_style atomic
boundary p p p
read_data {data}
change_box all triclinic
pair_style eam/alloy
pair_coeff * * {potential_file} Cu Ni
fix relax all box/relax tri 0.0 vmax 0.001
min_style cg
min_modify norm max
minimize 0.0 1.0e-10 100000 1000000
minimize 0.0 1.0e-10 100000 1000000
"""
        )
        energy = lmp.get_thermo("pe")
        lo, hi, xy, yz, xz, _, _ = lmp.extract_box()
    finally:
        lmp.close()
    lengths = np.array(hi) - np.array(lo)
    matrix = np.array([[lengths[0], 0, 0], [xy, lengths[1], 0], [xz, yz, lengths[2]]])
    relaxed = minimize(
        system,
        EAM(potential_file),
        cell="full",
        max_force=1e-9,
        max_stress=1e-6,
        max_steps=50000,
    )
    assert relaxed.converged
    assert relaxed.energy == pytest.approx(energy, abs=1e-10)
    ours = relaxed.system.box.matrix
    np.testing.assert_allclose(ours @ ours.T, matrix @ matrix.T, rtol=1e-6, atol=1e-6)
    assert relaxed.system.box.volume < system.box.volume * 0.9  # it shrank
