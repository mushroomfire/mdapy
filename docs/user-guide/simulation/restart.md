# Restart files

<p class="runs-on" markdown>**Runs on** the processor · the [GPU](gpu.md) · not [LAMMPS](in-lammps.md), which writes its own with `write_restart`</p>

A long run that a crash, a time limit or a power cut stops goes on from its
last restart file: `run(..., restart="md.restart", restart_every=N)` writes the
whole state of the dynamics every N steps and at the end of the run, and
`MolecularDynamics.load` reads it back, with the potential given again.
`resume` then finishes the run that stopped, with its settings, its controls and
its ramps, and the script goes on with what came after.

<!-- skip-test -->
```python
from mdapy.simulation import MolecularDynamics

with MolecularDynamics(crystal, potential, timestep=0.002, log="md.log") as md:
    md.velocities(300, seed=1)
    md.run(500_000, ensemble="npt", temperature=(300, 1500), pressure=0.0,
           restart="md.restart", restart_every=10_000)
```

After a crash:

<!-- skip-test -->
```python
md = MolecularDynamics.load("md.restart", potential, log="md.log")
md.step     # where the file was written
md.resume() # the rest of the run, as if it had not stopped
md.close()
```

The run finished from the file is the run that did not stop, to the rounding of
the forces: their neighbor lists are built again, and sum in another order.
Between runs, `md.save("md.restart")` writes the state; a run's own file at its
end is such a file too, from which the next run of the script starts.

## Parameters

| Name | Type | Default | Meaning |
|---|---|---|---|
| `restart` | str or path | `None` | of `run` and `iterate`: the file, replaced each time once the new one is whole, so that a crash while writing leaves the last |
| `restart_every` | int32 | 1000 | steps between files, with `restart` |

`MolecularDynamics.load(path, potential, *, others=(), log=None)` reads a file:
`potential` is that of the dynamics, of the same kind; `others` the other
potentials its runs use, those of a [`Switch`](switching.md), in the order they
first came; `log` a log to go on with, added to. `resume(*, progress=None,
cancel=None)` finishes the run the file stopped in.

## What a file holds

The atoms with every column, the box, the time step, the step and the time; the
thermodynamic state recorded so far; the variables of the thermostats and
barostats, of [MSST](msst.md) and the [Hugoniostat](hugoniostat.md); the state of
every [control](controls.md), the deformation, walls, indenters, baths and the
adaptive time step; the settings of the run it stopped in and the steps it ran.
The random forces of Langevin dynamics go on as they would have: each atom draws
its noise from the seed, its index and the step, on the processor and the GPU
alike.

It holds no potential, which `load` takes again, and no frames: those that
`save_every` keeps stay in memory; [`iterate`](records.md#frames-in-files) writes
them to a file as the run goes, which a crash leaves as far as it got. A file is
read only by the version of MDAPY that wrote it, as LAMMPS's own; `load` reads
MDAPY's objects, NumPy arrays and polars tables only, and refuses a file that
holds anything else.

## Errors

`run` and `iterate` raise `InvalidParameter` for `restart_every` without
`restart`, or not a positive integer, a `restart` that is not a path, a file
that cannot be written, and a LAMMPS calculator. `load` raises it for a file
that is not a restart file of this version of MDAPY, a potential of another
kind than the file's, or too few `others`; `resume` when no run stopped in the
file.
