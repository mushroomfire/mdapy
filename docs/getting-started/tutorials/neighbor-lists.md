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

## One list for a metal

The analyses of the structure of a metal look at its neighbors within some
5 Å: the nearest 12 to 18 of each atom, or those within a cutoff. One list
within 5 Å serves them all. An analysis of the nearest neighbors puts the
nearest pairs of each atom first in the list, once, and reads them there;
one within a cutoff reads the pairs closer than its own. Neither copies the
list, and every result is the one the analysis gives alone.

```python
from mdapy.analysis import (
    BondAngleDistribution,
    CoordinationNumber,
    RadialDistributionFunction,
    SteinhardtParameters,
)
from mdapy.identify import (
    AcklandJonesAnalysis,
    CentroSymmetryParameter,
    CommonNeighborAnalysis,
    PolyhedralTemplateMatching,
)

metal = [
    CommonNeighborAnalysis(),  # the 14 nearest
    AcklandJonesAnalysis(),  # the 14 nearest
    PolyhedralTemplateMatching(),  # the 18 nearest
    CentroSymmetryParameter(12),
    SteinhardtParameters((4, 6), num_neighbors=12),
    CommonNeighborAnalysis(3.086),  # within cutoffs
    CoordinationNumber(3.086),
    BondAngleDistribution(cutoff=3.086, bins=180),
    RadialDistributionFunction(5.0, bins=200),
]
alone, before = timed(lambda: [a.compute(system) for a in metal])


def shared_metal():
    nl = NeighborList.build(system, 5.0)
    return [a.compute(system, neighbors=nl) for a in metal]


shared, after = timed(shared_metal)
for one, other in zip(alone, shared):
    for name, value in vars(one).items():
        if isinstance(value, np.ndarray):
            assert np.array_equal(value, vars(other)[name])  # the same results
print(f"{before:.0f} ms alone, {after:.0f} ms with one list")
```

Without a list, each analysis finds the neighbors it needs as it goes,
without storing the pairs; with one, the search is done once and the
analyses only read it. The list must hold the neighbors each analysis
needs: an analysis of the 18 nearest refuses a list in which an atom has
fewer pairs.

Building the list costs about as much as the search of one analysis, as it
stores every pair, and putting the nearest first a third of that. One list
pays for many analyses, as here; for two or three, finding the neighbors as
they go is about as fast.

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

On a node of two Intel Xeon Gold 6248 (40 cores) with 40 threads, each list
built and sorted inside the timing, the best of three:

| Analyses | Atoms | Each alone | One list for all |
|---|---:|---:|---:|
| nine of a metal, one list within 5 Å | 108 000 | 231 ms | 142 ms |
| the same | 2 048 000 | 3.67 s | 2.31 s |
| the same without polyhedral template matching | 4 000 000 | 4.16 s | 2.16 s |
| three on Voronoi neighbors | 108 000 | 501 ms | 182 ms |

At 4 000 000 atoms, building the list takes 0.47 s and putting the 14
nearest of each atom first 0.15 s; then the centro-symmetry parameter takes
0.08 s instead of 0.51 s alone, and the radial distribution function 0.06 s
instead of 0.34 s. Polyhedral template matching spends most of its time
matching templates, which a list does not shorten.

A list within a cutoff is also the way to your own sums over pairs, with
NumPy, as in [Neighbor lists](../neighbor.md#methods).

## One list, the same atoms

A list serves every system with the positions and the box it was built for:
the system itself, one with more columns, or one read again from the same
file. Moved atoms or another box need a new list; analyses refuse the old
one. `nl.fits(system)` tells which.

```python
from mdapy.errors import InvalidParameter

nl = NeighborList.build(system, 5.0)
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
