# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Path integrals against LAMMPS's fix pimd/langevin, which needs LAMMPS
with MPI and the REPLICA package, one partition a bead: neon of
Lennard-Jones at 20 K, its quantum estimators averaged, the same within
their statistical errors. MDAPY_LAMMPS_NEB gives the command that starts
it, with {n} for the number of processes, such as
``mpirun -np {n} /path/to/lmp``."""

import glob
import os
import subprocess

import numpy as np
import pytest

from mdapy.build import build_crystal
from mdapy.calculators import LennardJones
from mdapy.simulation import MolecularDynamics

COMMAND = os.environ.get("MDAPY_LAMMPS_NEB")
pytestmark = pytest.mark.skipif(
    not COMMAND, reason="MDAPY_LAMMPS_NEB names no MPI LAMMPS with REPLICA"
)

BEADS, STEPS, SKIPPED, TEMPERATURE = 8, 12000, 2000, 20.0
INPUT = """units metal
atom_style atomic
atom_modify map array
lattice fcc 4.46
region box block 0 4 0 4 0 4
create_box 1 box
create_atoms 1 box
mass 1 20.1797
pair_style lj/cut 8.0
pair_coeff 1 1 0.0031 2.74
timestep 0.002
velocity all create 20.0 4711 dist gaussian mom yes
fix 1 all pimd/langevin method nmpimd integrator baoab ensemble nvt temp 20.0 &
    thermostat PILE_L 4711 tau 0.1 fixcom no
thermo_style custom step f_1[3] f_1[5] f_1[7] f_1[10]
thermo_modify norm no
thermo 10
run {steps}
"""


def mean_and_error(x):
    """The mean and its error from 20 blocks."""
    blocks = x[: len(x) // 20 * 20].reshape(20, -1).mean(axis=1)
    return x.mean(), blocks.std(ddof=1) / np.sqrt(20)


def lammps(tmp_path):
    (tmp_path / "in.pimd").write_text(INPUT.format(steps=STEPS))
    command = COMMAND.format(n=BEADS).split() + [
        "-partition", f"{BEADS}x1", "-in", "in.pimd", "-screen", "none",
    ]  # fmt: skip
    subprocess.run(command, cwd=tmp_path, check=True, capture_output=True, timeout=1800)
    rows = []
    for path in sorted(glob.glob(str(tmp_path / "log.lammps.*"))):
        lines = open(path).read().splitlines()
        start = next(i for i, line in enumerate(lines) if line.split()[:1] == ["Step"])
        table = []
        for line in lines[start + 1 :]:
            parts = line.split()
            if len(parts) != 5:
                break
            table.append([float(p) for p in parts])
        rows.append(np.array(table))
    keep = rows[0][:, 0] > SKIPPED
    return {
        "potential_energy": np.mean([r[keep, 1] for r in rows], axis=0),
        "primitive_kinetic_energy": rows[0][keep, 2],
        "kinetic_energy": rows[0][keep, 3],
        "pressure": rows[0][keep, 4] * 1e-4,  # bar to GPa
    }


def test_neon_as_lammps(tmp_path):
    expected = lammps(tmp_path)
    neon = build_crystal("fcc", 4.46, elements="Ne", repeat=(4, 4, 4))
    lj = LennardJones({"Ne": (0.0031, 2.74)}, 8.0)
    with MolecularDynamics(neon, lj, timestep=0.002, beads=BEADS) as md:
        md.velocities(TEMPERATURE, seed=4711)
        md.run(STEPS, ensemble="nvt", thermostat="langevin", temperature=TEMPERATURE,
               thermostat_damping=0.1, seed=4711, thermo=10)  # fmt: skip
        thermo = md.thermo.filter(md.thermo["step"] > SKIPPED)
    classical = 1.5 * len(neon) * 8.617333e-5 * TEMPERATURE
    for name, values in expected.items():
        mine, mine_error = mean_and_error(thermo[name].to_numpy())
        theirs, their_error = mean_and_error(values)
        assert abs(mine - theirs) < 4 * np.hypot(mine_error, their_error), name
    # quantum: the kinetic energy half again the classical
    assert thermo["kinetic_energy"].mean() > 1.4 * classical
