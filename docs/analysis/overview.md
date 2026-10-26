# Analysis

Every analysis in mdapy is used the same way: a class named after what it
computes takes the parameters, and its `compute` method takes a system and
returns a result.

```python
import numpy as np

import mdapy as mp
from mdapy.analysis import RadialDistributionFunction
from mdapy.identify import CommonNeighborAnalysis

# an fcc crystal of 4 x 4 x 4 cubic cells
a = 3.6
cells = np.stack(np.meshgrid(*[np.arange(4)] * 3, indexing="ij"), -1).reshape(-1, 1, 3)
basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
system = mp.System(((cells + basis) * a).reshape(-1, 3), mp.Box(4 * a), elements="Cu")

result = CommonNeighborAnalysis().compute(system)
result.counts()  # {'other': 0, 'fcc': 256, ...}
system = system.with_columns(result)  # adds the column 'structure'

nl = mp.NeighborList.build(system, 6.0)  # one search for several analyses
rdf = RadialDistributionFunction(r_max=6.0).compute(system, neighbors=nl)
```

Structure identification is in `mdapy.identify`; every other analysis is in
`mdapy.analysis`.

## Parameters

The constructor of an analysis takes its parameters and checks them, so a
wrong value fails before any computation. `compute` takes:

| Name | Type | Default | Meaning |
|---|---|---|---|
| `system` | `System` | | the atoms, with the columns the analysis needs, such as `element` or `velocities` |
| `neighbors` | `NeighborList` | `None` | a neighbor list of this system to use instead of a new search, for analyses that take one |
| `progress`, `cancel` | | `None` | progress reports and cancellation, as described in [Parallelism](../getting-started/parallelism.md#progress-and-cancellation) |

Analyses of a trajectory, the mean squared displacement and the Lindemann
index, take a sequence of systems, `frames`, instead of `system`. The
Voronoi analysis takes no neighbor list: it searches as far as each cell
needs.

A neighbor list passed as `neighbors` must belong to the same `System`
object. An analysis of the neighbors within a cutoff needs a list with at
least that cutoff. An analysis of the k nearest neighbors of each atom takes
either a list of nearest neighbors or a list within a cutoff, with at least
k neighbors for every atom. The results are those of the analysis's own
search: for nearest neighbors exactly, for a cutoff up to the order in which
values are summed.

Without `neighbors`, most analyses within a cutoff count each atom's
neighbors as the search finds them, without storing the pairs, which keeps
the memory small for large systems. Pass a list when several analyses need
the same neighbors.

## Results

A result is an immutable object whose arrays are read-only; all derive from
`mdapy.core.Result`.

| Method | Returns |
|---|---|
| `columns()` | the per-atom values by column name; empty for results of a whole system, such as a radial distribution function |

`system.with_columns(result)` adds these columns to a system, for example
to write them to a file.

Structure identification returns a `StructureResult`:

| Attribute or method | Type | Meaning |
|---|---|---|
| `structure` | (N,) int8 | the type of each atom, an index into `names`; its column is `structure` |
| `names` | tuple of str | the name of each type, in lower case, such as `"fcc"`; type 0 is always `"other"` |
| `colors` | tuple of (r, g, b) | the default color of each type, from 0 to 1 |
| `counts()` | dict of str to int | the number of atoms of each type, by name |

## Errors

The constructors raise
[`InvalidParameter`](../reference/api.md#mdapy.errors.InvalidParameter) for
invalid parameters. `compute` raises `InvalidParameter` for a neighbor list
that does not fit, as above;
[`MissingProperty`](../reference/api.md#mdapy.errors.MissingProperty) when
the system lacks a column the analysis needs; and
[`Cancelled`](../reference/api.md#mdapy.errors.Cancelled) when cancelled.

## Analyses

| Section | Classes |
|---|---|
| Structure identification | [`CommonNeighborAnalysis`](cna.md) · [`AcklandJonesAnalysis`](ackland-jones.md) · [`PolyhedralTemplateMatching`](ptm.md) · [`PlanarFaults`](planar-faults.md) · [`GrainSegmentation`](grain-segmentation.md) · [`DiamondStructure`](diamond.md) · [`ChillPlus`](chill-plus.md) · [`CentroSymmetryParameter`](csp.md) · [`CommonNeighborParameter`](cnp.md) |
| Local order and distributions | [`RadialDistributionFunction`](rdf.md) · [`CoordinationNumber`](coordination.md) · [`StructureFactor`](structure-factor.md) · [`BondAngleDistribution`](bond-angles.md) · [`WarrenCowley`](warren-cowley.md) · [`SteinhardtParameters`](steinhardt.md) · [`EntropyFingerprint`](entropy.md) · [`VoronoiAnalysis`](voronoi.md) |
| Defects and deformation | [`ClusterAnalysis`](clusters.md) · [`WignerSeitz`](wigner-seitz.md) · [`AtomicStrain`](atomic-strain.md) |
| Dynamics | [`MeanSquaredDisplacement`](msd.md) · [`LindemannIndex`](lindemann.md) · [`AtomicTemperature`](temperature.md) |
