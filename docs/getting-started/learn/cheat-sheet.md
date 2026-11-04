# Cheat sheet

The lessons on one page. Everything here assumes:

<!-- skip-test -->
```python
import numpy as np
import mdapy as mp
```

and the imports in [the table at the end](#where-each-thing-is-imported-from).

## Get a system

| I want | Write |
|---|---|
| a file | `system = read("file.data")` |
| a crystal | `build_crystal("fcc", 3.615, elements="Cu", repeat=(8, 8, 8))` |
| a polycrystal | `build_polycrystal(crystal, mp.Box(100.0), 8, seed=1)` |
| atoms of my own | `mp.System(positions, mp.Box(20.0), elements="Cu")` |
| to know what I have | `print(system)` |

## Read it

| I want | Write |
|---|---|
| a column | `system["positions"]`, `system["element"]` |
| a component | `system["positions.z"]` |
| an expression | `system["positions.x**2 + positions.y**2"]` |
| the number of atoms | `len(system)` |
| the box | `system.box.matrix`, `system.box.volume`, `system.box.pbc` |
| global values | `system.info["timestep"]` |
| anything else | `system.table`, a polars DataFrame |

## Change it

A system never changes: each of these returns a new one.

| I want | Write |
|---|---|
| a new or changed column | `system.with_columns(charge=values)` |
| the columns of an analysis | `system.with_columns(result)`, named after it, as `cna_structure` |
| the columns of two runs of one analysis | `system.with_columns(near=first, far=second)` |
| to move atoms | `system.with_columns(positions=new_positions)` |
| another box | `system.with_box(matrix, scale_positions=True)` |
| global values | `system.with_info(energy=-3.54)` |
| some atoms, by condition | `system.filter("positions.z > 10", element="Cu")` |
| some atoms, by index | `system.gather(indices)` |
| the atoms in another order | `system.sort("id")` |
| to drop columns | `system.drop("velocities")` |

## Ask it

Every analysis has the same three steps:

<!-- skip-test -->
```python
result = Analysis(parameters).compute(system)
system = system.with_columns(result)  # per-atom values, as columns
```

| I want | Write |
|---|---|
| the structure of each atom | `CommonNeighborAnalysis()`, `PolyhedralTemplateMatching()` |
| their numbers | `result.counts()` |
| the number of neighbors | `CoordinationNumber(cutoff=3.0)` |
| the radial distribution | `RadialDistributionFunction(r_max=8.0)`, then `result.r`, `result.g` |
| one list for several analyses | `nl = mp.NeighborList.build(system, 5.0)`, then `compute(system, neighbors=nl)` |

## Compute energies

| I want | Write |
|---|---|
| a potential | `EAM.zhou(["Cu"])` or `EAM("Cu.eam.alloy")` |
| energy and forces | `potential.compute(system, ("energy", "forces"))` |
| to relax the atoms | `minimize(system, potential)`, then `result.system` |
| to relax the box as well | `minimize(system, potential, cell="full")` |

## Follow a trajectory

| I want | Write |
|---|---|
| to open a file | `with read_trajectory("dump.*.lammpstrj") as trajectory:` |
| a frame, or some frames | `trajectory[0]`, `trajectory[::10]` |
| the frames through a function | `trajectory.map(function)` |
| one number per frame | `TimeSeries(function).compute(trajectory)` |
| to write them | `write("out.xyz", trajectory.map(function))` |

## Show and save it

| I want | Write |
|---|---|
| a file | `write("out.xyz", system)` |
| a picture | `render(system, "picture.png")` |
| colors by a column | `render(system, "picture.png", color="coordination")` |
| colors by structure | `render(system, "picture.png", color=result)` |
| what the colors mean | `render(system, "picture.png", color=result, legend=True)` |
| some atoms only | `render(system, "picture.png", atoms=system["positions.x"] < 20)` |
| on the graphics card | `render(system, "picture.png", device="gpu")` |
| a figure | `with plot.style(): fig, ax = plot.figure("single")` |
| to save a figure | `plot.save(fig, "figure")` |

## Where each thing is imported from

| Module | Holds |
|---|---|
| `mdapy` | `System`, `Box`, `Topology`, `NeighborList`, `set_num_threads` |
| `mdapy.io` | `read`, `write`, `read_trajectory` |
| `mdapy.build` | `build_crystal`, `build_polycrystal`, `random_alloy` and others |
| `mdapy.transform` | `replicate`, `wrap`, `combine`, `slice` and others |
| `mdapy.identify` | the structure analyses |
| `mdapy.analysis` | every other analysis |
| `mdapy.calculators` | `EAM`, `NEP`, `LAMMPS`, `VASP` |
| `mdapy.simulation` | `minimize`, `MolecularDynamics`, defect energies, free energies, melting points |
| `mdapy.lattice_dynamics` | `elastic_constants`, `phonons`, `quasi_harmonic` and others |
| `mdapy.mlip` | descriptors, choosing frames, errors of a model |
| `mdapy.visualization` | `render`, `plot`, `write_movie` |
| `mdapy.interop` | `to_ase`, `from_ase`, `to_ovito`, `from_ovito` and others |
