# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Run under mpirun: neb shares its images among the processes and gives
the path of one process, to the last bit; the first process finds it again
alone and compares. Prints "ok" on the first process, or raises.

    mpirun -np 3 python tests/mpi/check_neb.py
"""

import numpy as np
from mpi4py import MPI

from mdapy.build import build_crystal
from mdapy.calculators import EAM
from mdapy.simulation import _domain, neb

comm = MPI.COMM_WORLD
crystal = build_crystal("fcc", 3.615, elements="Cu", repeat=(3, 3, 3))
x = crystal["positions"]
# a vacancy and its neighbor hopping into it
gone = 0
near = int(np.argsort(np.linalg.norm(x - x[gone], axis=1))[1])
initial = crystal.filter(np.arange(len(crystal)) != gone)
moved = x.copy()
moved[near] = x[gone]
final = crystal.with_columns(positions=moved).filter(np.arange(len(crystal)) != gone)
eam = EAM.zhou(["Cu"])
split = neb(initial, final, eam, images=5, max_force=1e-3)
if comm.Get_rank() == 0:
    world = _domain.world
    _domain.world = lambda: None
    alone = neb(initial, final, eam, images=5, max_force=1e-3)
    _domain.world = world
    assert split.steps == alone.steps and split.converged
    np.testing.assert_array_equal(split.energies, alone.energies)
    for a, b in zip(split.images, alone.images):
        np.testing.assert_array_equal(a["positions"], b["positions"])
    print("ok", comm.Get_size(), "processes")
