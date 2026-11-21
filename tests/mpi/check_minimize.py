# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Run under mpirun: a minimization whose forces are shared among the
processes gives that of one process, the atoms alone and with the box. The
first process runs each again alone and compares. Prints "ok" on the first
process, or raises.

    MDAPY_NUM_THREADS=1 mpirun -np 4 python tests/mpi/check_minimize.py
"""

import numpy as np
from mpi4py import MPI

from mdapy.build import build_crystal
from mdapy.calculators import EAM
from mdapy.simulation import _domain, minimize

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world = _domain.world

crystal = build_crystal("fcc", 3.615, elements="Cu", repeat=(8, 8, 8))
rng = np.random.default_rng(0)
shaken = crystal.with_columns(
    positions=crystal["positions"] + rng.normal(0, 0.08, (len(crystal), 3))
)
eam = EAM.zhou(["Cu"])
for options in ({}, {"cell": "full", "pressure": 1.0}):
    shared = minimize(shaken, eam, max_force=1e-4, max_steps=400, **options)
    if rank == 0:
        _domain.world = lambda: None
        try:
            alone = minimize(shaken, eam, max_force=1e-4, max_steps=400, **options)
        finally:
            _domain.world = world
        assert shared.steps == alone.steps and shared.converged == alone.converged
        np.testing.assert_allclose(shared.system["positions"], alone.system["positions"],
                                   atol=1e-7)  # fmt: skip
        np.testing.assert_allclose(shared.energy, alone.energy, rtol=1e-11)
    comm.Barrier()
if rank == 0:
    print("ok", comm.Get_size(), "processes")
