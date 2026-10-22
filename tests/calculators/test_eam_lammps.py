# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""EAM against LAMMPS pair_style eam/alloy. Skipped where LAMMPS is not installed."""

import numpy as np
import pytest

import mdapy
from mdapy.calculators import EAM, minimize

from .eam_files import alloy, write_setfl

lammps = pytest.importorskip("lammps")

# LAMMPS writes the virial as pressure (bar) and per-atom stress (bar Å^3),
# converted with its own constant for metal units, nktv2p = 1.6021765e6 bar
# per eV/Å^3 (the exact value is 1.602176634e6).
BAR_A3_TO_EV = 1 / 1.6021765e6


def by_id(lmp, values):
    n = lmp.extract_global("nlocal")
    ids = np.array(lmp.numpy.extract_atom("id")[:n], copy=True)
    return np.array(values[:n], copy=True)[np.argsort(ids)]


@pytest.mark.parametrize("seed, repeat", [(10, 2), (11, 3)])
def test_against_lammps(tmp_path, seed, repeat):
    potential_file = write_setfl(tmp_path / "CuNi.eam.alloy")
    system = alloy(seed=seed, repeat=repeat, jitter=0.2)
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
pair_style eam/alloy
pair_coeff * * {potential_file} Cu Ni
compute pe all pe/atom
compute w all stress/atom NULL virial
compute p all pressure NULL virial
thermo_style custom step pe c_p[1]
run 0
"""
        )
        energy = lmp.get_thermo("pe")
        atomic = by_id(lmp, lmp.numpy.extract_compute("pe", 1, 1))
        forces = by_id(lmp, lmp.numpy.extract_atom("f"))
        stress = by_id(lmp, lmp.numpy.extract_compute("w", 1, 2))
        # the global vector of the pressure compute: xx yy zz xy xz yz
        pressure = np.array(lmp.numpy.extract_compute("p", 0, 1), copy=True)
    finally:
        lmp.close()
    result = EAM(potential_file).compute(
        system, ("energy", "forces", "virial", "atomic_virials")
    )
    np.testing.assert_allclose(result.energy, energy, rtol=1e-12)
    np.testing.assert_allclose(result.atomic_energies, atomic, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(result.forces, forces, rtol=1e-9, atol=1e-10)
    xx, yy, zz, xy, xz, yz = pressure
    virial = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]) * system.box.volume
    np.testing.assert_allclose(
        result.virial, virial * BAR_A3_TO_EV, rtol=1e-9, atol=1e-9
    )
    s = stress  # xx yy zz xy xz yz, the negative of the per-atom virial
    atomic_virial = (
        -np.stack([s[:, [0, 3, 4]], s[:, [3, 1, 5]], s[:, [4, 5, 2]]], axis=1)
        * BAR_A3_TO_EV
    )
    np.testing.assert_allclose(
        result.atomic_virials, atomic_virial, rtol=1e-9, atol=1e-9
    )


def test_relaxation_as_lammps(tmp_path):
    """FIRE in LAMMPS and in mdapy reach the same minimum from the same start:
    a vacancy in a slightly perturbed alloy near the potential's lattice
    constant."""
    potential_file = write_setfl(tmp_path / "CuNi.eam.alloy")
    full = alloy(seed=12, repeat=3, jitter=0.05, a=3.35)
    system = full.select(np.arange(1, len(full))).with_columns(
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
