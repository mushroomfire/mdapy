# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Elastic constants against the stress of strained and relaxed boxes in
LAMMPS. Skipped where LAMMPS is not installed."""

import numpy as np
import pytest

import mdapy
from mdapy import units
from mdapy.calculators import EAM
from mdapy.lattice_dynamics import elastic_constants
from mdapy.simulation import minimize

from calculators.eam_files import alloy, write_setfl

lammps = pytest.importorskip("lammps")

# LAMMPS converts the virial to bar with its own constant for metal units
BAR_TO_EV_PER_A3 = 1 / 1.6021765e6
# change_box arguments of the strains xx, yy, zz, yz, xz, xy: a normal strain
# also scales the tilt factors along it, as a uniform strain does; a shear
# strain tilts the box, and x += g y also moves the third vector by g yz
CHANGES = (
    "x scale {s} xy final {xy_s} xz final {xz_s}",
    "y scale {s} yz final {yz_s}",
    "z scale {s}",
    "yz delta {t}",
    "xz delta {t}",
    "xy delta {t} xz delta {t_yz}",
)


def lammps_stress(data, potential_file, change):
    """The stress (Voigt, eV/Å^3) of the system in data after change_box and a
    relaxation of the positions."""
    lmp = lammps.lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
    try:
        lmp.commands_string(
            f"""
units metal
atom_style atomic
boundary p p p
read_data {data}
change_box all triclinic
change_box all {change} remap units box
pair_style eam/alloy
pair_coeff * * {potential_file} Cu Ni
compute p all pressure NULL virial
thermo_style custom step pe c_p[1]
min_style fire
min_modify norm max
minimize 0.0 1.0e-10 100000 1000000
"""
        )
        xx, yy, zz, xy, xz, yz = np.array(
            lmp.numpy.extract_compute("p", 0, 1), copy=True
        )
    finally:
        lmp.close()
    return -np.array([xx, yy, zz, yz, xz, xy]) * BAR_TO_EV_PER_A3


def test_constants_as_lammps(tmp_path):
    """mdapy strains symmetrically and LAMMPS shears by tilting the box; at
    zero stress the two differ only in the second order of the strain."""
    potential_file = write_setfl(tmp_path / "CuNi.eam.alloy")
    eam = EAM(potential_file)
    start = alloy(seed=15, repeat=3, jitter=0.0, a=3.35)
    relaxed = minimize(
        start, eam, cell="full", max_force=1e-9, max_stress=1e-7, max_steps=20000
    ).system
    # LAMMPS rotates a general triclinic box to its restricted form, and the
    # constants of an alloy change with the orientation: strain both in that
    # frame
    general, data = tmp_path / "general.data", tmp_path / "in.data"
    mdapy.write(general, relaxed, elements=["Cu", "Ni"])
    lmp = lammps.lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
    try:
        lmp.commands_string(
            f"units metal\natom_style atomic\nread_data {general}\nwrite_data {data}"
        )
    finally:
        lmp.close()
    rotated = mdapy.read(data).with_columns(
        element=np.where(mdapy.read(data).table["type"].to_numpy() == 1, "Cu", "Ni")
    )
    strain = 1e-3
    lengths = np.diag(rotated.box.matrix)
    xy, xz, yz = (
        rotated.box.matrix[1, 0],
        rotated.box.matrix[2, 0],
        rotated.box.matrix[2, 1],
    )
    expected = np.empty((6, 6))
    for k, change in enumerate(CHANGES):
        stresses = []
        for sign in (1, -1):
            # an engineering shear strain g tilts by g times the length along it
            t = sign * strain * (lengths[2] if k in (3, 4) else lengths[1])
            scale = 1 + sign * strain
            command = change.format(
                s=scale,
                t=t,
                t_yz=sign * strain * yz,
                xy_s=xy * scale,
                xz_s=xz * scale,
                yz_s=yz * scale,
            )
            stresses.append(lammps_stress(data, potential_file, command))
        expected[:, k] = (stresses[0] - stresses[1]) / (2 * strain)
    expected *= units.EV_PER_A3_TO_GPA
    result = elastic_constants(rotated, eam, strain=strain, max_force=1e-9)
    np.testing.assert_allclose(result.constants, expected, rtol=1e-4, atol=1e-3)
    assert expected[0, 0] > 50  # GPa, a stiff metal
