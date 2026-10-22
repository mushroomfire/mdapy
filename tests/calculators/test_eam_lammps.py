# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""EAM against LAMMPS pair_style eam/alloy. Skipped where LAMMPS is not installed."""

import numpy as np
import pytest

import mdapy
from mdapy.calculators import EAM

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
