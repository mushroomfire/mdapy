# Simulation

MDAPY simulates atoms itself: molecular dynamics, the subject of this part,
[energy minimization](minimize.md), which relaxes the atoms and the box to the nearest
minimum of the energy, and [minimum energy paths](neb.md), the nudged elastic band
between two minima and the barrier on its saddle point. The functions that run them live in `mdapy.simulation`, the
controls of a run in `mdapy.simulation.controls`; the properties computed from them,
such as conductivities and free energies, in `mdapy.properties`.

MDAPY has its own molecular dynamics engine. It runs on the processor and, for a
potential on an NVIDIA GPU, entirely on the [GPU](gpu.md), so that a study goes from
the first structure through the dynamics to the analysis and the last picture in one
Python script, with no other program. The same calls run [in LAMMPS](in-lammps.md)
with a [LAMMPS calculator](../potentials/lammps.md), for potentials that MDAPY does not
have or for runs over many processes.

## What it is for

The engine is made for materials: metals, alloys, semiconductors and other crystals,
and glasses and liquids of atoms, with the many-body potentials of MDAPY, such as
[EAM](../potentials/eam.md), [MEAM](../potentials/meam.md),
[Tersoff](../potentials/tersoff.md) and
[Stillinger–Weber](../potentials/stillinger-weber.md), and machine-learned potentials,
such as [NEP](../potentials/nep.md) or any model with an
[ASE calculator](../interop/ase-calculators.md). Molecular force fields, with bonds,
angles and dihedrals, are not its aim.

Its features serve mechanical and thermal simulations: the loading of a crystal by
deformation, indentation and shock, and the transport of heat, free energies and
melting.

## How a simulation goes

A `MolecularDynamics` holds the atoms, the potential and the time step. Each `run`
integrates a number of steps in an [ensemble](ensembles.md), which keeps the energy,
the temperature or the pressure; [controls](controls.md) given to a run act on groups
of atoms, the box, the time step or the potential while it lasts. What a run gives back
is [recorded](records.md): the thermodynamic state in `thermo`, a log, and the atoms
along the way and at the end.

A copper crystal brought to 300 K at zero pressure, then stretched along x:

```python
from mdapy.build import build_crystal
from mdapy.calculators import EAM
from mdapy.simulation import MolecularDynamics
from mdapy.simulation.controls import Deform

crystal = build_crystal("fcc", 3.615, elements="Cu", repeat=(6, 6, 6))
copper = EAM.zhou(["Cu"])  # EAM.zhou(["Cu"], device="cuda") runs on the GPU

with MolecularDynamics(crystal, copper, timestep=0.002) as md:
    md.velocities(300, seed=1)
    md.run(2500, ensemble="npt", temperature=300, pressure=0.0)  # a run, an ensemble
    md.run(2500, ensemble="nvt", temperature=300, controls=[Deform(xx=0.02)])
    thermo = md.thermo  # one row per record of both runs
    stretched = md.system  # the atoms at the end
```

[Running molecular dynamics](running.md) explains these calls, and the pages below
each part of them.

## What it can do

✓ runs there; — does not.

| To | Use | Processor | GPU | LAMMPS |
|---|---|:-:|:-:|:-:|
| keep the energy, the temperature or the pressure | [`ensemble="nve"`, `"nvt"`, `"npt"`, `"nph"`](ensembles.md) | ✓ | ✓ | ✓ |
| hold the temperature with friction and noise | [`thermostat="langevin"`](ensembles.md#thermostats) | ✓ | ✓ | ✓ |
| let the box change in every direction, along each axis, in its plane or along one axis | [`barostat`](ensembles.md#barostats) | ✓ | ✓ | ✓ |
| start at a temperature, or give part of the atoms one | [`velocities`, `thermal_velocities`](running.md#starting-velocities) | ✓ | ✓ | ✓ |
| record the state, write a log, keep or stream the atoms | [`thermo`, `log`, `save_every`, `iterate`](records.md) | ✓ | ✓ | ✓ |
| go on after a crash from a restart file | [`restart`, `load`, `resume`](restart.md) | ✓ | ✓ | — |
| hold, move or push a group of atoms, and record the force on it | [`Hold`, `Move`, `AddForce`](controls.md) | ✓ | ✓ | ✓ |
| keep a group at a temperature, a heat bath | [`Langevin`](controls.md) | ✓ | ✓ | ✓ |
| take smaller steps when atoms move fast | [`AdaptiveTimestep`](adaptive-timestep.md) | ✓ | ✓ | ✓ |
| turn one potential into another | [`Switch`](switching.md) | ✓ | ✓ | — |
| stretch, compress or shear the box | [`Deform`](deformation.md) | ✓ | ✓ | ✓ |
| press a sphere, a cylinder or a plane into the atoms | [`Indenter`](indentation.md) | ✓ | ✓ | ✓ |
| drive a shock with a piston, or reflect the atoms at a still wall | [`Wall`](walls.md) | ✓ | ✓ | ✓ |
| find states on the Hugoniot of a shock | [`ensemble="nphug"`](hugoniostat.md) | ✓ | ✓ | ✓ |
| follow the matter behind a steady shock | [`ensemble="msst"`](msst.md) | ✓ | ✓ | ✓ |

[Choosing a method](shock.md) compares the three ways of a shock. On the GPU, MDAPY
needs to be built with CUDA: see [On the GPU](gpu.md).

## Properties from molecular dynamics

These calculations run the dynamics themselves, one call each; their pages say where
they run.

- Thermal conductivity: [Green–Kubo](../properties/green-kubo.md),
  [HNEMD](../properties/hnemd.md) and [NEMD](../properties/nemd.md).
- [Free energy](../properties/free-energy.md),
  [melting point](../properties/melting-point.md),
  [phase diagrams of two elements](../properties/phase-diagram.md) and
  [pressure-temperature diagrams](../properties/pressure-temperature.md).
- [Elastic constants from molecular dynamics](../properties/md-elastic.md),
  [phonons at finite temperature](../properties/thermal-phonons.md) and
  [stacking faults at finite temperature](../properties/stacking-fault-temperature.md).

## Where to start

[Running molecular dynamics](running.md), [Ensembles](ensembles.md) and
[Records and frames](records.md) cover what every simulation needs;
[On the GPU](gpu.md) takes it to millions of atoms. The other pages each describe one
feature. The examples run whole studies in MDAPY:
[a copper cluster melting](../../examples/cluster-melting.md),
[a collision cascade in tungsten](../../examples/radiation-cascade.md) and
[shock and spall in copper](../../examples/shock.md).
