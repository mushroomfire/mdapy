# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Molecular dynamics on many processes of one thread, for the cases of
benchmarks/gpu.py on the processor: MDAPY's own dynamics, or the dynamics in
LAMMPS through MDAPY's LAMMPS calculator. Each run builds the crystal, gives
it the velocities of the temperature, runs 20 steps and times the next 100; it
prints the milliseconds of a step on the first process.

    MDAPY_NUM_THREADS=1 srun -n 40 --mpi=pmix python benchmarks/processes.py eam mdapy 80 FOLDER
    MDAPY_NUM_THREADS=1 srun -n 40 --mpi=pmix python benchmarks/processes.py eam lammps 80 FOLDER

FOLDER holds Cu_zhou.eam.alloy, written by EAM.zhou(["Cu"]).write(path), and
the Si.sw and Si.tersoff of the potentials folder of LAMMPS.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import gpu  # noqa: E402

name, engine, repeat, folder = (
    sys.argv[1],
    sys.argv[2],
    int(sys.argv[3]),
    Path(sys.argv[4]),
)
steps = int(sys.argv[5]) if len(sys.argv) > 5 else 100
from mpi4py import MPI  # noqa: E402

from mdapy.build import build_crystal  # noqa: E402
from mdapy.calculators import (  # noqa: E402
    EAM,
    LAMMPS,
    LennardJones,
    Morse,
    StillingerWeber,
    Tersoff,
)
from mdapy.simulation import MolecularDynamics  # noqa: E402

case = gpu.CASES[name]
comm = MPI.COMM_WORLD
structure, a, element, _ = case["crystal"]
system = build_crystal(structure, a, elements=element, repeat=(repeat,) * 3)
if engine == "lammps":
    text = case["lammps"]
    for file in ("Si.sw", "Si.tersoff", "Cu_zhou.eam.alloy"):
        text = text.replace(file, str(folder / file))
    potential = LAMMPS(text, [element])
else:
    potential = {
        "eam": lambda: EAM(folder / "Cu_zhou.eam.alloy"),
        "lj": lambda: LennardJones((0.0104, 3.40), 8.5),
        "morse": lambda: Morse({"Cu": (0.3429, 1.3588, 2.866)}, 6.5),
        "sw": lambda: StillingerWeber(folder / "Si.sw"),
        "tersoff": lambda: Tersoff(folder / "Si.tersoff"),
    }[name]()
with MolecularDynamics(system, potential, timestep=case["timestep"]) as md:
    md.velocities(case["temperature"], seed=1)
    md.run(20, thermo=20)
    comm.Barrier()
    start = time.perf_counter()
    md.run(steps, thermo=steps)
    comm.Barrier()
    took = (time.perf_counter() - start) / steps
if comm.Get_rank() == 0:
    print(f"{name} {engine} {len(system)} atoms, {comm.Get_size()} processes: "
          f"{took * 1e3:.2f} ms/step", flush=True)  # fmt: skip
