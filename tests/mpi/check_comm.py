# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Run under mpirun: with comm=MPI.COMM_SELF each process runs a dynamics, a
minimization and a band of its own, the same as alone; the processes of one
dynamics that hold different temperatures all refuse it; two groups of
processes from COMM_WORLD.Split each split a dynamics of their own as they
would alone. With a LAMMPS of MPI (MDAPY_LAMMPS_NEB is not needed), a LAMMPS
calculator on COMM_SELF computes on each process alone. Prints "ok" on the
first process, or raises.

    MDAPY_NUM_THREADS=1 mpirun -np 4 python tests/mpi/check_comm.py
"""

import numpy as np
from mpi4py import MPI

from mdapy.build import build_crystal
from mdapy.calculators import EAM, LennardJones
from mdapy.errors import InvalidParameter
from mdapy.simulation import MolecularDynamics, minimize

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()
crystal = build_crystal("fcc", 3.615, elements="Cu", repeat=(6, 6, 6))
eam = EAM.zhou(["Cu"])


def nvt(temperature, comm):
    with MolecularDynamics(crystal, eam, timestep=0.002, comm=comm) as md:
        md.velocities(temperature, seed=1)
        md.run(20, ensemble="nvt", temperature=temperature, seed=2)
        return md.system["positions"], md.thermo["temperature"].to_numpy()


# a dynamics of its own on each process, at its own temperature
mine = nvt(200.0 + 100 * rank, MPI.COMM_SELF)
results = comm.gather(mine)
if rank == 0:
    for r, (x, t) in enumerate(results):
        ex, et = nvt(200.0 + 100 * r, MPI.COMM_SELF)
        np.testing.assert_array_equal(x, ex)
        np.testing.assert_array_equal(t, et)

# a minimization of its own on each process
rng = np.random.default_rng(rank)
shaken = crystal.with_columns(
    positions=crystal["positions"] + rng.normal(0, 0.05, (len(crystal), 3))
)
lj = LennardJones({"Cu": (0.4, 2.3)}, 5.0)
energy = minimize(shaken, lj, max_force=1e-4, comm=MPI.COMM_SELF).energy
energies = comm.gather((rank, energy))
if rank == 0:
    for r, e in energies:
        rng = np.random.default_rng(r)
        alone = crystal.with_columns(
            positions=crystal["positions"] + rng.normal(0, 0.05, (len(crystal), 3))
        )
        assert minimize(alone, lj, max_force=1e-4, comm=MPI.COMM_SELF).energy == e

# the processes of one dynamics refuse a temperature of each
if size > 1:
    try:
        nvt(200.0 + 100 * rank, None)
    except InvalidParameter as error:
        assert "differ in the velocities" in str(error), error
    else:
        raise AssertionError("different temperatures on the processes of one dynamics")

# two groups, each splitting a dynamics of its own
if size >= 2:
    group = comm.Split(rank % 2, rank)
    split = nvt(300.0 + 100 * (rank % 2), group)
    alone = nvt(300.0 + 100 * (rank % 2), MPI.COMM_SELF)
    np.testing.assert_allclose(split[0], alone[0], atol=1e-9)
    group.Free()

# a LAMMPS calculator of its own on each process
try:
    from mdapy.calculators import LAMMPS

    lammps_lj = LAMMPS("pair_style lj/cut 5.0\npair_coeff * * 0.4 2.3", ["Cu"],
                       comm=MPI.COMM_SELF)  # fmt: skip
except ImportError:
    lammps_lj = None
if lammps_lj is not None:
    computed = comm.gather((rank, lammps_lj.compute(shaken).energy))
    if rank == 0:
        for r, e in computed:
            rng = np.random.default_rng(r)
            alone = crystal.with_columns(
                positions=crystal["positions"] + rng.normal(0, 0.05, (len(crystal), 3))
            )
            assert abs(lammps_lj.compute(alone).energy - e) <= 1e-10 * abs(e), (r, e)
comm.Barrier()
if rank == 0:
    print("ok", size, "processes")
