# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Molecular dynamics on the GPUs of one node, one process of MPI each, for
the cases of benchmarks/gpu.py: MDAPY's own dynamics on many processes, or
the input of LAMMPS with KOKKOS for the same run. Each run builds the
crystal, gives it the velocities of the temperature, runs 50 steps and times
the next 300; it prints the milliseconds of a step on the first process.

    srun -n 4 --gpus-per-task=1 --gpu-bind=none --mpi=pmix \\
        python benchmarks/gpus.py mdapy eam 80 mixed FOLDER
    python benchmarks/gpus.py input eam 80 > in.eam
    srun -n 4 --gpus-per-task=1 --gpu-bind=none --mpi=pmix \\
        lmp -k on g 4 -sf kk -pk kokkos neigh full newton off -in in.eam

FOLDER holds the files of the potentials that read one (see gpu.py). MDAPY
gives the memory of the GPUs to MPI itself when Open MPI moves it (built with
CUDA), as KOKKOS does by default; MDAPY_GPU_AWARE_MPI=0 sends it through the
host.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import gpu  # noqa: E402

mode, name, repeat = sys.argv[1], sys.argv[2], int(sys.argv[3])
case = gpu.CASES[name]
if mode == "input":
    print(gpu.lammps_input(case, repeat))
    raise SystemExit(0)
precision, folder = sys.argv[4], Path(sys.argv[5])
from mpi4py import MPI  # noqa: E402

from mdapy.build import build_crystal  # noqa: E402
from mdapy.simulation import MolecularDynamics  # noqa: E402

comm = MPI.COMM_WORLD
structure, a, element, _ = case["crystal"]
system = build_crystal(structure, a, elements=element, repeat=(repeat,) * 3)
potential = gpu.potential(name, folder, precision)
with MolecularDynamics(system, potential, timestep=case["timestep"]) as md:
    md.velocities(case["temperature"], seed=1)
    md.run(gpu.WARM, thermo=gpu.WARM)
    comm.Barrier()
    start = time.perf_counter()
    md.run(gpu.STEPS, thermo=gpu.STEPS)
    comm.Barrier()
    took = (time.perf_counter() - start) / gpu.STEPS
if comm.Get_rank() == 0:
    print(f"{name} {len(system)} atoms, {comm.Get_size()} GPUs, {precision}: "
          f"{took * 1e3:.3f} ms/step", flush=True)  # fmt: skip
