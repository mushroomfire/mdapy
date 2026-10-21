# Analyses

Every analysis in mdapy is a class named after what it computes. Its
parameters are given to the constructor, which checks them; its
`compute(system, *, neighbors=None)` method returns a result object.

```python
import numpy as np
import mdapy
from mdapy.analysis import RadialDistributionFunction
from mdapy.identify import CommonNeighborAnalysis

# an fcc crystal of 4 x 4 x 4 cubic cells
a = 3.6
cells = np.stack(np.meshgrid(*[np.arange(4)] * 3, indexing="ij"), -1).reshape(-1, 1, 3)
basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
system = mdapy.System(
    ((cells + basis) * a).reshape(-1, 3), mdapy.Box(4 * a), elements="Cu"
)

cna = CommonNeighborAnalysis()
result = cna.compute(system)
result.counts()  # {'other': 0, 'fcc': 256, ...}
system = system.with_columns(result)  # adds the column 'structure'

nl = mdapy.NeighborList.build(system, 6.0)
rdf = RadialDistributionFunction(r_max=6.0).compute(system, neighbors=nl)
```

## Results

A result is a frozen dataclass whose arrays are read-only. All of them derive
from `mdapy.core.Result`. Per-atom values are listed by its `columns()` method,
and `System.with_columns(result)` adds them to a system under those names.
Results of a whole system, such as a radial distribution function, have no
columns.

## Neighbor lists

An analysis builds the neighbor list it needs. `neighbors=` passes one that
was built before, so that several analyses share a single search; it must
belong to the same `System` object and reach at least as far as the analysis
needs, or [`InvalidParameter`](../reference/api.md#mdapy.errors.InvalidParameter)
is raised.

## Available analyses

| Class | Page |
|---|---|
| `mdapy.identify.CommonNeighborAnalysis` | [Structure identification](structure.md) |
| `mdapy.identify.CentroSymmetryParameter` | [Structure identification](structure.md) |
| `mdapy.analysis.RadialDistributionFunction` | [Radial distribution and coordination](distribution.md) |
| `mdapy.analysis.CoordinationNumber` | [Radial distribution and coordination](distribution.md) |
| `mdapy.analysis.SteinhardtParameters` | [Bond-orientational order](steinhardt.md) |
| `mdapy.analysis.ClusterAnalysis` | [Clusters, bond angles and short-range order](bonds.md) |
| `mdapy.analysis.BondAngleDistribution` | [Clusters, bond angles and short-range order](bonds.md) |
| `mdapy.analysis.WarrenCowley` | [Clusters, bond angles and short-range order](bonds.md) |
