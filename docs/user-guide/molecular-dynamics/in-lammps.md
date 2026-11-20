# In LAMMPS

<p class="runs-on" markdown>**Runs on** [LAMMPS](https://www.lammps.org), stable_30Sep2026</p>

With a [LAMMPS calculator](../potentials/lammps.md) the same calls run in LAMMPS: its fixes `nve`,
`nvt`, `npt`, `nph`, `nphug` and `msst`, and the controls as its fixes, with any of its
potentials and packages, on several processes with MPI or on GPUs:

<!-- skip-test -->
```python
from mdapy.calculators import LAMMPS

potential = LAMMPS(
    "pair_style eam/alloy\npair_coeff * * Cu_mishin1.eam.alloy Cu", elements=["Cu"]
)
crystal = build_crystal("fcc", 3.615, elements="Cu", repeat=(8, 8, 8))
with MolecularDynamics(crystal, potential, timestep=0.002) as md:
    md.velocities(600, seed=1)
    md.run(10000, ensemble="npt", temperature=300, pressure=0.0, thermo=100)
    md.command("compute peratom all pe/atom")  # any LAMMPS command between runs
```

[Running LAMMPS from MDAPY](../../examples/molecular-dynamics.md) says what it
is not made for, and how to run it on several threads or a GPU;
[LAMMPS on many processes with MPI](../../examples/mpi.md) runs it with MPI.

## What runs as which fix

| In MDAPY | In LAMMPS |
|---|---|
| `ensemble="nve"`, `"nvt"`, `"npt"`, `"nph"` | `fix nve`, `nvt`, `npt`, `nph` |
| `thermostat="langevin"` | `fix langevin` with `fix nve` or `fix nph` |
| `"nphug"`, `"msst"` | `fix nphug`, `fix msst` |
| [`Hold`](controls.md) | `fix setforce` |
| [`Move`](controls.md) | `fix move linear` |
| [`AddForce`](controls.md) | `fix addforce` |
| [`Langevin`](controls.md) | `fix langevin ... tally yes` on the group |
| [`Deform`](deformation.md) | `fix deform` |
| [`AdaptiveTimestep`](adaptive-timestep.md) | `fix dt/reset` |
| [`Wall`](walls.md) | `fix wall/reflect`, or `fix wall/piston` for a moving one, from below along z only |
| [`Indenter`](indentation.md) | `fix indent` |
| [`Switch`](switching.md) | none: MDAPY only |

## The box and the units

In LAMMPS, the dynamics runs in metal units, those of MDAPY, through its
Python module, installed as the
[LAMMPS documentation](https://docs.lammps.org/Python_install.html) describes;
`potential` gives the commands of the interaction, the element of each atom
type and the command-line arguments of LAMMPS, such as those of Kokkos. The
box may have any orientation and need not be periodic: LAMMPS gets it rotated
into its own form, its non-periodic directions shrink-wrapped but never
smaller than the box of `system`, and everything comes back in the
orientation of `system`. MDAPY's own dynamics works in the same form of the
box.

## LAMMPS commands

`command(commands)` runs any LAMMPS commands between runs, one or several lines, and
`lammps` is the LAMMPS instance itself, for what MDAPY does not wrap. Fixes and computes
that a command defines stay until a command removes them; the fixes MDAPY defines for a
run are removed at its end.

## Frames from a dump

With LAMMPS, a `dump` writes the frames too, from LAMMPS itself:

<!-- skip-test -->
```python
with MolecularDynamics(crystal, potential, timestep=0.002) as md:
    md.velocities(600, seed=1)
    md.command(
        "dump frames all custom 500 frames.*.lammpstrj id element x y z vx vy vz\n"
        "dump_modify frames element Cu time yes"
    )
    md.run(5000, ensemble="nvt", temperature=300)
    md.command("undump frames")
```

LAMMPS writes the atoms in the order it holds them, which changes as it runs,
each with its `id`, numbered from 1 in the order of `system`.
[`AtomicDisplacement`](../analysis/displacement.md) and
[`MeanSquaredDisplacement`](../analysis/msd.md) match the atoms of frames by
it, and `frame.sort("id")` puts them in the order of `system`. Their positions
are those of LAMMPS, which are those of `system` for a box at the origin with
its first vector along x and its second in the xy plane, as `build_crystal`
makes them, to the six digits a dump writes by default (`format float %.10g`
writes more). `iterate` is the better choice: it writes any format of
several frames, in the order and the orientation of `system`, in LAMMPS and
in MDAPY. A dump suits a run over several processes with
[MPI](../../examples/mpi.md), where each process would receive every atom
of every frame of `iterate`.

## Errors

`MolecularDynamics` with a LAMMPS calculator raises `InvalidParameter` for elements
without an atom type, a `movable` column that holds atoms and commands that LAMMPS
rejects, and `ImportError` without the LAMMPS Python module; `command` raises
`InvalidParameter` in MDAPY and for commands LAMMPS rejects, and `lammps` in MDAPY and
after `close`. See [Running](running.md#errors).
