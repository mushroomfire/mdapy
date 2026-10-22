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

Analyses with a cutoff, such as the radial distribution function, the
coordination number, common neighbor analysis with a cutoff or the bond
angle distribution, need no list: without one they count or classify the
neighbors of each atom while they find them, in parallel and without
storing the pairs. Given a list, they read it instead, which is slower
today than searching again:

```python
from mdapy.analysis import CoordinationNumber, RadialDistributionFunction

within = [
    RadialDistributionFunction(5.0, bins=100),
    CoordinationNumber(3.0),
    CommonNeighborAnalysis(3.086),
]
_, without_list = timed(lambda: [a.compute(system) for a in within])
nl = NeighborList.build(system, 5.0)
_, with_list = timed(lambda: [a.compute(system, neighbors=nl) for a in within])
print(f"{without_list:.0f} ms without a list, {with_list:.0f} ms with one")
```

On an Apple M5 with 10 threads, the list built inside the timing except
for the last row:

| Analyses | Each alone | One list for all |
|---|---:|---:|
| four on the nearest neighbors | 110 ms | 75 ms |
| three on Voronoi neighbors | 450 ms | 165 ms |
| three within a cutoff | 35 ms | 160 ms |

A list within a cutoff is still the way to your own sums over pairs, with
NumPy, as in [Neighbor lists](../neighbor.md#methods).

## One list, one system

A list belongs to the system it was built for, the object itself; an
analysis refuses it for any other system, even one made from it with an
extra column. Compute the analyses first and attach their results at the
end:

```python
from mdapy.errors import InvalidParameter

nl = NeighborList.build(system, num_neighbors=14)
cna = CommonNeighborAnalysis().compute(system, neighbors=nl)
csp = CentroSymmetryParameter(12).compute(system, neighbors=nl)
labelled = system.with_columns(cna, csp)  # attach at the end

try:
    CommonNeighborAnalysis().compute(labelled, neighbors=nl)
except InvalidParameter as error:
    print(error)  # the neighbor list was built for another system
```

[Energy minimization](../../simulation/minimize.md) keeps its own list
between steps, with a margin beyond the cutoff of the calculator, and
searches again only when an atom has moved far enough to need it.
