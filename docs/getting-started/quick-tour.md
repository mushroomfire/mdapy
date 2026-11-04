# Quick tour

This page builds a small copper crystal, writes and reads it, finds its
neighbors, computes its energy with a potential, relaxes it and writes the
result. Every block runs as shown, in order. To learn step by step instead,
start with [the lessons](learn/table.md).

## A system

A `System` holds the atoms, as a table of columns, and the simulation box.

```python
import numpy as np
import mdapy as mp
from mdapy.build import build_crystal

a = 3.615  # Å
system = build_crystal("fcc", a, elements="Cu", repeat=(3, 3, 3))
len(system), system.box.volume  # 108 atoms, Å^3
```

See [Systems and topology](../basics/system.md) and [Simulation box](../basics/box.md).

## Files

```python
from mdapy.io import read, write

write("copper.xyz", system)
back = read("copper.xyz")
np.abs(back["positions"] - system["positions"]).max()  # 0.0
```

See [Reading and writing files](../basics/io.md) and, for a whole trajectory,
[Trajectories](../basics/trajectory.md).

## Neighbors

```python
nl = mp.NeighborList.build(system, 3.0)
nl.counts  # 12 neighbors for every atom of fcc
nl.distances.min()  # a / sqrt(2)
```

See [Neighbor lists](../basics/neighbor.md).

## Energy and forces

```python
from mdapy.calculators import EAM

potential = EAM.zhou(["Cu"])
result = potential.compute(system, ("energy", "forces", "virial"))
result.energy / len(system)  # eV per atom
result.stress  # (3, 3) GPa
```

See [Calculators](../user-guide/simulation/calculators.md) and [EAM](../user-guide/simulation/eam.md).

## Relaxation

The atoms are displaced at random, then the positions and the box relax to
the nearest minimum of the energy.

```python
from mdapy.simulation import minimize

rng = np.random.default_rng(0)
rattled = system.with_columns(
    positions=system["positions"] + rng.normal(scale=0.05, size=(108, 3))
)
relaxed = minimize(rattled, potential, cell="full")
relaxed.converged, relaxed.energy, relaxed.stress
write("relaxed.xyz", relaxed.system)  # with forces, energy and stress
```

See [Energy minimization](../user-guide/simulation/minimize.md).

## Analysis and pictures

A polycrystal, its grain boundaries found by common neighbor analysis and
drawn by the ray tracer:

```python
from mdapy.build import build_polycrystal
from mdapy.identify import CommonNeighborAnalysis
from mdapy.visualization import render

grains = build_polycrystal(
    build_crystal("fcc", a, elements="Cu"), mp.Box(60.0), 6, seed=4
)
structures = CommonNeighborAnalysis().compute(grains)
structures.counts()  # fcc, and other at the boundaries
render(
    grains,
    "images/quick-tour.png",
    atoms=grains["positions"][:, 0] < 30,  # the half at x < 30 Å
    color=structures,
    style="matte",
    width=900,
    height=600,
    legend=True,
)
```

![A polycrystal cut in half](images/quick-tour.png)

*Six grains of copper cut in half: fcc atoms green, grain boundaries white,
in the matte style.*

See [Building structures](../user-guide/build/polycrystals.md), [Common neighbor
analysis](../user-guide/analysis/cna.md) and [Rendering](../user-guide/visualization/rendering.md).
