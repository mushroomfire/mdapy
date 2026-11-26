# Records and frames

<p class="runs-on" markdown>**Runs on** the processor · the [GPU](gpu.md) · [LAMMPS](in-lammps.md)</p>

A run of [molecular dynamics](running.md) gives back, as it goes, the thermodynamic
state in `thermo`, a log that shows its progress, and the atoms: in `frames`, or one at
a time from `iterate`, to a file or straight into an analysis.

## Columns of `thermo`

`thermo` is a polars `DataFrame` of all the runs so far, one row every `thermo` steps
of each run, and at its start and end.

| Column of `thermo` | Type | Unit | Meaning |
|---|---|---|---|
| `step` | int64 | | time step number |
| `time` | float64 | ps | time |
| `temperature` | float64 | K | temperature, from the kinetic energy of 3N − 3 directions, or of those the `movable` column frees |
| `potential_energy`, `kinetic_energy`, `total_energy` | float64 | eV | energies of all the atoms |
| `pressure` | float64 | GPa | pressure; NaN in MDAPY with a calculator without virials |
| `volume` | float64 | Å³ | volume of the box |
| `stress` | (6,) float64 | GPa | stress in Voigt order xx, yy, zz, yz, xz, xy, positive under tension, in the axes of `system` |
| `hugoniot` | float64 | K | with `"nphug"`: how far the state is from the Hugoniot, the energy it lacks over the kinetic energy of 1 K: see [Hugoniostat](hugoniostat.md) |
| `shock_velocity`, `particle_velocity` | float64 | Å/ps | with `"nphug"`: the velocities of a steady shock to this state and of the matter behind it; with `"msst"`, the second |
| `rayleigh`, `lagrangian_position`, `msst_energy` | float64 | GPa, Å, eV | with `"msst"`: see [Multiscale shock technique](msst.md#parameters) |
| `<name>_force` | (3,) float64 | eV/Å | with a named [`Move`](controls.md), the force of the potential on its atoms; with a named [`Indenter`](indentation.md#parameters), the force of the atoms on it |
| `<name>_energy` | float64 | eV | with a named [`Langevin`](controls.md), the heat it gave its group since the run began, negative when it took heat away; with a named `Indenter`, its energy |
| `primitive_kinetic_energy`, `spring_energy`, `ring_polymer_energy` | float64 | eV | with `beads`: see [Path integrals](pimd.md#results), whose energies, temperature and pressure are those of the quantum estimators |
| `electronic_loss` | float64 | eV | with an [`ElectronicStopping`](electronic-stopping.md), the energy the atoms gave the electrons since the run began |
| `<name>_lambda` | float64 | | with a named [`Switch`](switching.md), λ |
| `<name>_difference` | float64 | eV | with a named `Switch`, the energy of the target less that of the potential |

## Log

With `log`, the dynamics writes what it does as it goes, so that a long run
shows its progress, `tail -f md.log` in a terminal:

```python
from mdapy.build import build_crystal
from mdapy.calculators import EAM
from mdapy.simulation import MolecularDynamics

crystal = build_crystal("fcc", 3.615, elements="Cu", repeat=(5, 5, 5))
copper = EAM.zhou(["Cu"])
with MolecularDynamics(crystal, copper, timestep=0.002, log="md.log") as md:
    md.velocities(600, seed=1)
    md.run(200, thermo=50)
```

```text
# MDAPY 2.2.0, molecular dynamics, 2026-11-19 13:29:02
# 500 atoms; box vectors 18.0750 18.0750 18.0750 Å long, periodic along a b c
# engine: MDAPY on the processor, 10 threads
# machine: Apple M5, 10 logical CPUs, 10 usable, 24.0 GB of memory
# potential: EAM(elements=('Cu',), cutoff=5.715751993501819)
# time step: 0.002 ps
#
# run 1: 200 steps, nve; thermo every 50
#     step        time   temperature    potential_energy    kinetic_energy        total_energy      pressure            volume      wall
         0      0.0000       600.000        -1769.991294         38.700444        -1731.290850      0.697585         5905.2042      0.00
        50      0.1000       256.824        -1747.829401         16.565321        -1731.264079      3.565728         5905.2042      0.02
       100      0.2000       324.699        -1752.217267         20.943354        -1731.273913      3.078099         5905.2042      0.03
       150      0.3000       307.193        -1751.084251         19.814151        -1731.270099      3.109218         5905.2042      0.04
       200      0.4000       301.119        -1750.691224         19.422382        -1731.268842      3.167338         5905.2042      0.06
# run 1: 200 steps in 0.06 s; 3340.2 steps/s, 1.67 million atom-steps/s, 577.2 ns/day
```

Each run writes its settings, a row every `thermo` steps, the row
`thermo` records, with `wall`, the seconds since the run began, and its
speed; a run that stops early, cancelled or failed, says where it stopped.
The engine is MDAPY on the processor with its threads, MDAPY on a GPU with
its memory, compute capability, multiprocessors and precision, or LAMMPS with
its processes and command-line arguments; the machine is the processor, its
logical CPUs, those this process may use, and the memory, as
`mdapy.show_versions()` names them, of the main process under MPI. The
[minimization](minimize.md) and the [band](neb.md) write the machine, and the
GPU of a potential there, in their heading too.
Lines that are not rows of numbers start with `#`, so that
`numpy.loadtxt("md.log")` reads the rows of every run as one table. Under
MPI the main process alone writes. Every run and `iterate` of the
`MolecularDynamics` writes to the same log, which `close` closes. A calculator
with a log of its own, as the [DFT](../potentials/dft.md#in-molecular-dynamics), writes
it to another file: the same file for both raises `InvalidParameter`, their rows being
different.

## Frames in files

The frames that `save_every` keeps stay in memory as long as the
`MolecularDynamics` or its `frames` do, about 60 bytes per atom each: a
hundred frames of a million atoms take 6 GB. `iterate` runs as `run` does
but gives each frame as the run reaches it and keeps none, so that
[`mdapy.io.write`](../../basics/io.md) writes them to a file one at a
time while the run goes on, and
[`read_trajectory`](../../basics/trajectory.md) reads them back one at a time:

<!-- skip-test -->
```python
import mdapy
from mdapy.identify import CommonNeighborAnalysis
from mdapy.io import read_trajectory

with MolecularDynamics(crystal, potential, timestep=0.002) as md:
    md.velocities(600, seed=1)
    frames = md.iterate(5000, save_every=500, ensemble="nvt", temperature=300)
    mdapy.io.write("frames.xyz", frames)  # one frame in memory at a time

cna = CommonNeighborAnalysis()
with read_trajectory("frames.xyz") as trajectory:
    for frame in trajectory:
        print(frame.info["time"], cna.compute(frame).counts()["fcc"])
```

or analyzes them as they come, without a file:

<!-- skip-test -->
```python
with MolecularDynamics(crystal, potential, timestep=0.002) as md:
    md.velocities(600, seed=1)
    for frame in md.iterate(5000, save_every=500, ensemble="nvt", temperature=300):
        print(frame.info["time"], cna.compute(frame).counts()["fcc"])
```

Each frame is a `System` as `system` gives it: the atoms in the order of
`system`, with their velocities, images and other columns, in full
precision, the box, and the step and the time in `info`, in LAMMPS and in
MDAPY alike. The run goes on as the frames are read, from the first, and
stops where their reading stops, such as at a `break` out of the loop, with
the steps run so far kept; while they are being read, the dynamics takes no
other `run`, `iterate`, `velocities` or `command`. As `save_every` does, it
gives the atoms at the start unless the run before gave them there, so that
the frames of runs one after the other follow each other once each.

[In LAMMPS](in-lammps.md#frames-from-a-dump), a `dump` of LAMMPS writes the frames
too.

## Errors

`log` raises `InvalidParameter` when it is neither a path nor a stream, or cannot be
written; `iterate` raises what `run` raises, for its options at once and for the rest as
its frames are read, and `InvalidParameter` without `save_every`; `run`, `iterate`,
`velocities` and `command` raise `InvalidParameter` while the frames of an `iterate` are
being read. See [Running](running.md#errors).
