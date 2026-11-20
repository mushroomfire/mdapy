# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""neb against LAMMPS's neb, which runs each image in a partition of its own
of an MPI LAMMPS with the REPLICA package: skipped unless
MDAPY_LAMMPS_NEB gives the command that starts it, with {n} for the number
of processes, such as ``mpirun -np {n} /path/to/lmp``."""

import os
import shlex
import subprocess

import numpy as np
import pytest

import mdapy
from mdapy.build import build_crystal
from mdapy.calculators import EAM
from mdapy.simulation import minimize, neb

COMMAND = os.environ.get("MDAPY_LAMMPS_NEB")

pytestmark = pytest.mark.skipif(
    not COMMAND, reason="MDAPY_LAMMPS_NEB names no MPI LAMMPS with REPLICA"
)


def lammps_neb(tmp_path, initial, final, potential, images, more=""):
    """The energies and the positions of LAMMPS's band, with the climbing
    image in its second stage."""
    mdapy.io.write(tmp_path / "initial.data", initial, elements=["Cu"])
    x = final["positions"]
    lines = [str(len(final))] + [
        f"{k + 1} {x[k, 0]:.17g} {x[k, 1]:.17g} {x[k, 2]:.17g}"
        for k in range(len(final))
    ]
    (tmp_path / "final.txt").write_text("\n".join(lines) + "\n")
    replicas = images + 2
    (tmp_path / "in.neb").write_text(
        f"""units metal
atom_style atomic
atom_modify map array sort 0 0.0
boundary p p p
read_data initial.data
pair_style eam/alloy
pair_coeff * * {potential} Cu
fix band all neb 1.0 {more}
min_style fire
timestep 0.001
thermo 100
neb 0.0 1.0e-7 50000 50000 100 final final.txt
variable u uloop {replicas}
write_data out.$u.data
print "ENERGY $(pe:%.12f)"
"""
    )
    command = shlex.split(COMMAND.format(n=replicas)) + [
        "-partition", f"{replicas}x1", "-in", "in.neb", "-nocite",
    ]  # fmt: skip
    subprocess.run(command, cwd=tmp_path, check=True, capture_output=True, timeout=600)
    energies, positions = [], []
    for u in range(1, replicas + 1):
        log = (tmp_path / f"log.lammps.{u - 1}").read_text().splitlines()
        energies.append(
            float(next(s for s in log if s.startswith("ENERGY")).split()[1])
        )
        back = mdapy.io.read(tmp_path / f"out.{u}.data")
        order = np.argsort(back.table["id"].to_numpy())
        positions.append(back["positions"][order])
    return np.array(energies), np.stack(positions)


def test_a_vacancy_hop_as_lammps(tmp_path):
    eam = EAM.zhou(["Cu"])
    potential = tmp_path / "Cu.eam.alloy"
    eam.write(potential)
    eam = EAM(potential)
    a = 3.615
    crystal = build_crystal("fcc", a, elements="Cu", repeat=(3, 3, 3))
    x = crystal["positions"]
    hop = int(np.argmin(np.linalg.norm(x - [a / 2, a / 2, 0.0], axis=1)))
    keep = np.arange(len(crystal)) > 0
    moved = x.copy()
    moved[hop] = x[0]
    initial = minimize(crystal.filter(keep), eam, max_force=1e-8, max_steps=10000)
    final = minimize(
        crystal.with_columns(positions=moved).filter(keep), eam, max_force=1e-8,
        max_steps=10000,
    )  # fmt: skip
    ids = np.arange(1, len(initial.system) + 1)
    initial, final = (r.system.with_columns(id=ids) for r in (initial, final))
    energies, positions = lammps_neb(tmp_path, initial, final, potential, 3)
    ours = neb(
        initial, final, eam, images=3, spring=1.0, max_force=1e-7, max_steps=50000
    )
    assert ours.converged
    # the saddle and the states to rounding; the other images sit along the
    # band within what the force left by the springs, 1e-7 eV/Å, moves them
    np.testing.assert_allclose(ours.energies, energies, rtol=0, atol=1e-7)
    top = ours.highest
    assert ours.energies[[0, top, -1]] == pytest.approx(
        energies[[0, top, -1]], abs=1e-9
    )
    assert ours.barrier == pytest.approx(energies.max() - energies[0], abs=1e-9)
    for image, theirs in zip(ours.images, positions):
        step = initial.box.minimum_image(image["positions"] - theirs)
        np.testing.assert_allclose(step, 0.0, atol=1e-5)
