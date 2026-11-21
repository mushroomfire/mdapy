# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Run under mpirun: path-integral dynamics with its beads shared among the
processes, each computing the forces of its own, gives the run of one
process. The first process runs it again alone and compares. Prints "ok"
on the first process, or raises.

    MDAPY_NUM_THREADS=1 mpirun -np 3 python tests/mpi/check_pimd.py
"""

import numpy as np
from mpi4py import MPI

from mdapy.build import build_crystal
from mdapy.calculators import EAM
from mdapy.simulation import MolecularDynamics, _domain

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world = _domain.world

crystal = build_crystal("fcc", 3.615, elements="Cu", repeat=(3, 3, 3))
eam = EAM.zhou(["Cu"])


def run():
    with MolecularDynamics(crystal, eam, timestep=0.002, beads=6) as md:
        md.velocities(50, seed=1)
        md.run(40, ensemble="nvt", thermostat="langevin", temperature=50, seed=2,
               thermo=10)  # fmt: skip
        return md.system, md.thermo, [b["positions"] for b in md.beads]


shared = run()
if rank == 0:
    _domain.world = lambda: None
    try:
        alone = run()
    finally:
        _domain.world = world
    np.testing.assert_allclose(
        shared[0]["velocities"], alone[0]["velocities"], atol=1e-12
    )
    for a, b in zip(shared[2], alone[2]):
        np.testing.assert_allclose(a, b, atol=1e-12)
    for column in ("potential_energy", "kinetic_energy", "pressure"):
        np.testing.assert_allclose(shared[1][column], alone[1][column], rtol=1e-12)
    print("ok", comm.Get_size(), "processes")
comm.Barrier()
