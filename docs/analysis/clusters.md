# Cluster analysis

`ClusterAnalysis` finds the groups of atoms joined by bonds, also across
periodic boundaries.

```python
import numpy as np

import mdapy
from mdapy.analysis import ClusterAnalysis

# two groups of atoms far apart in a 30 Å box
rng = np.random.default_rng(0)
positions = np.vstack([rng.random((20, 3)) * 4, rng.random((10, 3)) * 4 + 15])
system = mdapy.System(positions, mdapy.Box(30.0), elements="Cu")

result = ClusterAnalysis(cutoff=3.0).compute(system)
result.sizes  # atoms in each cluster, largest first
system = system.with_columns(result)  # the column 'cluster'
```

## Parameters

| Name | Type | Default | Unit | Meaning |
|---|---|---|---|---|
| `cutoff` | float | `None` | Å | atoms closer than it are bonded |
| `cutoffs` | dict | `None` | Å | instead of `cutoff`, a bond length for each pair of elements `(a, b)`, in either order; pairs not given are never bonded |

`compute(system, *, neighbors=None, progress=None, cancel=None)` takes the
arguments of [every analysis](overview.md#parameters). A given neighbor list
must reach the largest cutoff; without one, the bonds are found and used at
once, never stored. `cutoffs` need an `element` column.

## Results

A `ClusterResult`:

| Attribute | Type | Meaning |
|---|---|---|
| `cluster` | (N,) int64 | the cluster of each atom, numbered from 0 by decreasing size, equal sizes in the order of their first atom; its column `cluster` |
| `sizes` | (clusters,) int64 | the number of atoms of each cluster |

## Errors

The constructor raises `InvalidParameter` without exactly one of `cutoff`
and `cutoffs`, for a cutoff that is not positive or an unknown element;
`compute` raises it for a neighbor list that does not fit, and
`MissingProperty` for `cutoffs` without an `element` column.

## Definition

Two atoms are bonded when closer than the cutoff of their pair of elements;
a cluster is a largest set of atoms connected by bonds. An atom without
bonds is a cluster of its own.

## See also

- OVITO [Cluster analysis](https://docs.ovito.org/reference/pipelines/modifiers/cluster_analysis.html).
- LAMMPS [`compute cluster/atom`](https://docs.lammps.org/compute_cluster_atom.html).
