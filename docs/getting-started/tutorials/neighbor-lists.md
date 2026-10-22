# Sharing neighbor lists

Every analysis finds the neighbors it needs by itself. When several
analyses of one system need the same neighbors, a [neighbor
list](../neighbor.md) built once and given to each with `neighbors=` saves
the searches. This page shows when that pays, and when it does not.

The system is a crystal of 108 000 copper atoms, displaced at random as at
room temperature:

```python
import time

import numpy as np

from mdapy.build import build_crystal
from mdapy.neighbor import NeighborList

crystal = build_crystal("fcc", 3.615, elements="Cu", repeat=(30, 30, 30))
rng = np.random.default_rng(0)
system = crystal.with_positions(
    crystal.positions + rng.normal(0.0, 0.1, (len(crystal), 3))
)


def timed(work):
    """The result of work() and how long it took, in ms."""
    start = time.perf_counter()
    result = work()
    return result, 1000 * (time.perf_counter() - start)
```

## Nearest neighbors

Adaptive common neighbor analysis and Ackland–Jones analysis look at the 14
nearest neighbors of each atom, the centro-symmetry parameter and these
Steinhardt parameters at the 12 nearest. A list of the 14 nearest serves
all four: an analysis that needs fewer takes the nearest of them.

```python
from mdapy.analysis import SteinhardtParameters
from mdapy.identify import (
    AcklandJonesAnalysis,
    CentroSymmetryParameter,
    CommonNeighborAnalysis,
)

nearest = [
    CommonNeighborAnalysis(),
    AcklandJonesAnalysis(),
    CentroSymmetryParameter(12),
    SteinhardtParameters((4, 6), num_neighbors=12),
]
alone, before = timed(lambda: [a.compute(system) for a in nearest])


def shared_nearest():
    nl = NeighborList.build(system, num_neighbors=14)
    return [a.compute(system, neighbors=nl) for a in nearest]


shared, after = timed(shared_nearest)
for one, other in zip(alone, shared):
    for name, value in vars(one).items():
        if isinstance(value, np.ndarray):
            assert np.array_equal(value, vars(other)[name])  # the same results
print(f"{before:.0f} ms alone, {after:.0f} ms with one list")
```

## Voronoi neighbors

A Voronoi tessellation costs far more than a search within a cutoff, so
analyses of Voronoi neighbors gain most from sharing one list. Here three
kinds of Steinhardt parameters over the faces of the cells:

```python
voronoi = [
    SteinhardtParameters((4, 6), voronoi=True),
    SteinhardtParameters((4, 6), voronoi=True, weighted=True),
    SteinhardtParameters((4, 6), voronoi=True, weighted=True, average=True),
]
_, before = timed(lambda: [a.compute(system) for a in voronoi])


def shared_voronoi():
    nl = NeighborList.build(system, voronoi=True)
    return [a.compute(system, neighbors=nl) for a in voronoi]


_, after = timed(shared_voronoi)
print(f"{before:.0f} ms alone, {after:.0f} ms with one list")
```

## Neighbors within a cutoff

Analyses with a cutoff, such as the coordination number, common neighbor
analysis with a cutoff, the bond angle distribution or the radial
distribution function, take a list with their cutoff or a longer one and
use the pairs closer than theirs. Without a list they count or classify the
neighbors of each atom while they find them, without storing the pairs,
which is already fast; a list of just their cutoff still saves a little:

```python
from mdapy.analysis import BondAngleDistribution, CoordinationNumber

within = [
    CoordinationNumber(3.086),
    CommonNeighborAnalysis(3.086),
    BondAngleDistribution(cutoff=3.086, bins=180),
]
_, before = timed(lambda: [a.compute(system) for a in within])


def shared_within():
    nl = NeighborList.build(system, 3.086)
    return [a.compute(system, neighbors=nl) for a in within]


_, after = timed(shared_within)
print(f"{before:.0f} ms alone, {after:.0f} ms with one list")
```

A list longer than the analyses need costs more to build than it saves.

On an Apple M5 with 10 threads, each list built inside the timing:

| Analyses | Each alone | One list for all |
|---|---:|---:|
| four on the nearest neighbors | 115 ms | 72 ms |
| three on Voronoi neighbors | 440 ms | 170 ms |
| three within a cutoff | 42 ms | 34 ms |

A list within a cutoff is also the way to your own sums over pairs, with
NumPy, as in [Neighbor lists](../neighbor.md#methods).

## One list, the same atoms

A list serves every system with the positions and the box it was built for:
the system itself, one with more columns, or one read again from the same
file. Moved atoms or another box need a new list; analyses refuse the old
one. `nl.fits(system)` tells which.

```python
from mdapy.errors import InvalidParameter

nl = NeighborList.build(system, num_neighbors=14)
cna = CommonNeighborAnalysis().compute(system, neighbors=nl)
labelled = system.with_columns(cna)  # the same atoms, one more column
csp = CentroSymmetryParameter(12).compute(labelled, neighbors=nl)
nl.fits(labelled)  # True

shifted = system.with_positions(system.positions + [0.1, 0.0, 0.0])
nl.fits(shifted)  # False
try:
    CommonNeighborAnalysis().compute(shifted, neighbors=nl)
except InvalidParameter as error:
    print(error)  # the neighbor list was built for other positions or another box
```

[Energy minimization](../../simulation/minimize.md) keeps its own list
between steps, with a margin beyond the cutoff of the calculator, and
searches again only when an atom has moved far enough to need it.
