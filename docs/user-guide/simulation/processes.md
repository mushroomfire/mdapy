# On many processes

<p class="runs-on" markdown>**Runs on** the processor, on many processes with MPI · a [GPU](gpu.md) for each process · [LAMMPS](in-lammps.md) splits its own runs: see [LAMMPS on many processes](../../examples/mpi.md)</p>

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

The processes of one dynamics must hold the same system and take the same runs.
Each checks it, at `MolecularDynamics`, `velocities` and each run, and when one
differs every process raises `InvalidParameter`. Without this check, a
temperature of its own on each process would give one system made of the atoms
of each process, each part at its own temperature.

## A simulation on each process

For many independent simulations, such as a temperature or a seed on each
process, `comm=MPI.COMM_SELF` gives each process a simulation of its own, as if
it were alone:

<!-- skip-test -->
```python
from mpi4py import MPI

world = MPI.COMM_WORLD
temperature = 300 + 100 * world.rank

with MolecularDynamics(
    crystal, potential, timestep=0.002, log=f"md.{world.rank}.log", comm=MPI.COMM_SELF
) as md:
    md.velocities(temperature, seed=1)
    md.run(10_000, ensemble="nvt", temperature=temperature)
```

A communicator of a group of processes splits a simulation among the group:
`MPI.COMM_WORLD.Split(world.rank // 4)` gives each four processes a simulation
of their own. `minimize` and `neb` take `comm` too, and so does the
[LAMMPS calculator](../potentials/lammps.md), whose LAMMPS computes on the
processes given: `MPI.COMM_SELF` for a LAMMPS of its own on each process. `neb`
shares its images among its processes, each computing its own, so it refuses a
LAMMPS calculator on several processes unless `neb` runs with
`comm=MPI.COMM_SELF`, all of them then computing each image together.

## How the box is split

The box is cut into a grid of bricks, one for each process, with the least
surface between them, as LAMMPS's `processors * * *` cuts it: 16 processes cut
a cube 2 × 2 × 4. Each process computes the atoms of its brick alone, from them
and from the ghosts, copies of the atoms within the cutoff of the potential and
a skin of 1 Å of the brick, which come from the processes around it, periodic
images among them; along a box vector of one brick the box stays periodic and
has no ghosts. What an atom of a many-body potential has of its neighbors goes
to its ghosts within each step, as the slope of the embedding function of EAM,
or the forces of a process's atoms on the ghosts go back to the atoms, as for
Tersoff and NEP; MEAM, whose screening reaches the neighbors of the neighbors,
keeps ghosts within twice its cutoff. Bricks thinner than the cutoff pass the
ghosts on, brick to brick, so that a box may be cut into as many bricks as it
has room for, as long as it is thicker than the cutoff and the skin along each
vector it is cut along. The atoms go to the processes of their bricks, and the
ghosts are found again, when the box changes or an atom has moved half the
skin; in between, the ghosts move with their atoms.

A process of one thread computes each pair of atoms of EAM, ADP and the pair
potentials once among all the processes, as LAMMPS with `newton on`: a pair of two of its atoms once, a pair of
one of its atoms and a ghost on the process of the atom of the lower id, and the
forces on the ghost go back to their atom. A process of several threads, set by
`MDAPY_NUM_THREADS`, has each atom gather its pairs, as one process does, so that
the threads write no atom but their own. One thread a process is the faster: run
as many processes as cores, with `MDAPY_NUM_THREADS=1`; [Performance](../../performance/mpi.md)
compares the times with those of LAMMPS. The forces of a run on
many processes sum in another order than on one, and so differ in their last
digits with the number of processes and of threads.

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

[Energy minimization](minimize.md) shares the forces in the same way: every process
keeps every atom and takes the same steps, and computes the forces of the atoms of its
brick with the ghosts it finds among them.

## With a GPU each

With a potential of `device="cuda"`, each process keeps its atoms and their ghosts
on a GPU, and the steps of the dynamics run there, as on [one GPU](gpu.md). At each
step only the positions of the atoms that other processes keep as ghosts go out,
and the few sums that the thermostats and the barostat need. Between two builds of
the neighbor list nothing else leaves the GPUs. A build, when an atom of any process
has moved half the skin or the box has changed, moves the atoms to the processes of
their bricks and finds the ghosts again.

The pair potentials, EAM, Stillinger-Weber, Tersoff and NEP take ghosts within their
cutoff and the skin, as KOKKOS does. EAM sends the slopes of the embedding functions
of the atoms to their ghosts at each step, between its densities and its forces.
The terms of Stillinger-Weber and Tersoff, and the energies of NEP, are those of the
atoms of the process, and the forces they leave on the ghosts go back to their
atoms, with their virials under a heat field, and with NEP whenever virials are
computed. ADP and MEAM take ghosts within twice their cutoff and the skin, so that
each GPU computes the atoms of its process from all their neighbors.

What the processes send each other goes to MPI in the memory of the GPU itself
when MPI moves that memory (an MPI built with CUDA, often called GPU-aware or
CUDA-aware), over NVLink or PCIe between the GPUs of a machine; otherwise through
the memory of the host, which takes longer. MDAPY asks Open MPI whether it moves
the memory of the GPU (`MPIX_Query_cuda_support`). For another MPI, or to decide,
set `MDAPY_GPU_AWARE_MPI` to `1` or `0` on every process:

```bash
MDAPY_GPU_AWARE_MPI=1 srun -n 4 --gpus-per-task=1 --mpi=pmix python heat.py
```

With `1` and an MPI that does not move the memory of the GPU, the processes crash.

The processes on a machine share its GPUs in turn: the first process of the
machine takes the first GPU, the second the second, and so on, starting again from
the first when there are more processes than GPUs. The launcher numbers the
processes (`OMPI_COMM_WORLD_LOCAL_RANK` of Open MPI, `MPI_LOCALRANKID` of MPICH,
`SLURM_LOCALID` of Slurm). On a cluster of nodes with 4 GPUs each:

```bash
srun -N 2 --ntasks-per-node=4 --gpus-per-node=4 --mpi=pmix python heat.py
```

One process on its GPU, with the whole dynamics there, is the faster on one
machine with one GPU: see [On the GPU](gpu.md).

## Errors

`MolecularDynamics`, `minimize` and `neb` raise `InvalidParameter` for a `comm` that
is not a communicator of mpi4py, and on every process when the processes of one
simulation hold different systems or options. `MolecularDynamics`
under MPI raises `InvalidParameter` for a calculator other
than MDAPY's own potentials, a box thinner than the
reach of the ghosts along a vector it is cut along, and when mpi4py is not
installed. A run raises it when the box shrinks until it is that thin, when a
`Switch` turns to a potential of a longer reach than the box holds, and when an
atom moves farther than a brick between two builds of the neighbor lists, a sign
of a time step far too long.
