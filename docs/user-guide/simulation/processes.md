# On many processes

<p class="runs-on" markdown>**Runs on** the processor, on many processes with MPI · not the [GPU](gpu.md) yet · [LAMMPS](in-lammps.md) splits its own runs: see [LAMMPS on many processes](../../examples/mpi.md)</p>

A script of molecular dynamics started with `mpirun` runs on all its processes
together: each process owns the atoms in a brick of the box and keeps copies of
those near it, so that a system too large or too slow for one process is shared
by many, on one machine or on many. The script stays as it is, and so does the
run: that of one process, to the rounding of the forces, which sum in another
order.

<!-- skip-test -->
```python
from mdapy.build import build_crystal
from mdapy.calculators import EAM
from mdapy.simulation import MolecularDynamics

crystal = build_crystal(
    "fcc", 3.615, elements="Cu", repeat=(50, 50, 50)
)  # 500 000 atoms
potential = EAM.zhou(["Cu"])

with MolecularDynamics(crystal, potential, timestep=0.002, log="md.log") as md:
    md.velocities(600, seed=1)
    md.run(
        10_000,
        ensemble="npt",
        temperature=300,
        pressure=0.0,
        thermo=500,
        restart="md.restart",
        restart_every=5000,
    )
    thermo = md.thermo
```

Saved as `heat.py`, it runs on 16 processes of one thread each with

```bash
MDAPY_NUM_THREADS=1 mpirun -np 16 python heat.py
```

On a cluster with Slurm, `srun` starts the processes, with the plugin of MPI
that the cluster names, often PMIx:

```bash
MDAPY_NUM_THREADS=1 srun -n 64 --mpi=pmix python heat.py
```

## What it needs

[mpi4py](https://mpi4py.readthedocs.io), built with the MPI that starts the
processes: `pip install mpi4py`. MDAPY splits its dynamics when `mpirun` or
`mpiexec` started more than one process, or `srun` with the plugin of PMIx or
PMI-2 (`--mpi=pmix`, `--mpi=pmi2`); with another launcher, the script imports
`mpi4py` before it makes the `MolecularDynamics`.

## Every process runs the script

Each process runs the whole script: what the script builds, each process builds,
and each makes every call of `MolecularDynamics`, in the same order, as one call
of all. Each gets the whole result: `system`, `thermo` and the frames of
`save_every` and `iterate` are the same on every process. The main process, of
rank 0, writes the log and the restart files. The files the script writes
itself, the main process writes alone, while every process goes through the
frames:

<!-- skip-test -->
```python
import mdapy
from mpi4py import MPI

main = MPI.COMM_WORLD.rank == 0

with MolecularDynamics(crystal, potential, timestep=0.002) as md:
    md.velocities(600, seed=1)
    for frame in md.iterate(10_000, save_every=1000, ensemble="nvt", temperature=900):
        if main:
            mdapy.io.write(f"frame.{frame.info['timestep']}.xyz", frame)
```

A restart file holds the atoms in their own order, whatever the number of
processes: a run stopped on 16 processes goes on from its file on 32, or on one.

## How the box is split

The box is cut into a grid of bricks, one for each process, with the least
surface between them, as LAMMPS's `processors * * *` cuts it: 16 processes cut
a cube 2 × 2 × 4. Each process computes the forces on the atoms of its brick from
them and from the ghosts, copies of the atoms within a reach of the brick: the
cutoff of the potential and a skin of 1 Å for a pair potential, twice the cutoff
and the skin for the others, whose forces reach to the neighbors of the
neighbors. Each brick must be at least as thick as that reach: 12.4 Å for copper
with `EAM.zhou`, whose cutoff is 5.7 Å, so that the box of 181 Å above takes at
most 14 bricks along each axis. The atoms go to the processes of their bricks,
and the ghosts are found again, when the box changes or an atom has moved half
the skin; in between, the ghosts move with their atoms.

## What it runs

Every ensemble and every control that runs on the processor (see
[What it can do](index.md#what-it-can-do)), with MDAPY's own potentials:
[EAM](../potentials/eam.md), [ADP](../potentials/adp.md),
[MEAM](../potentials/meam.md), [NEP](../potentials/nep.md),
[Tersoff](../potentials/tersoff.md),
[Stillinger–Weber](../potentials/stillinger-weber.md) and the
[pair potentials](../potentials/pair-potentials.md). Other calculators, as those
of ASE or [VASP](../potentials/vasp.md), run on one process; a
[LAMMPS](../potentials/lammps.md) potential runs the dynamics in LAMMPS, which
splits it itself. [`neb`](neb.md#on-many-processes) shares its images among the
processes, with any calculator.

## Errors

`MolecularDynamics` under MPI raises `InvalidParameter` for a calculator other
than MDAPY's own potentials, a potential on the GPU, a box whose bricks are
thinner than the reach of the ghosts, and when mpi4py is not installed. A run
raises it when the box shrinks until its bricks are thinner than the reach, when
a `Switch` turns to a potential of a longer reach than the bricks hold, and when
an atom moves farther than a brick between two builds of the neighbor lists, a
sign of a time step far too long.
