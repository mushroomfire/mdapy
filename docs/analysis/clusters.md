# Cluster analysis

`ClusterAnalysis` finds the groups of atoms joined by bonds, also across
periodic boundaries, and, when asked for, the positions that keep each
group unbroken by the boundaries and the centers of mass and gyration.

```python
import numpy as np

import mdapy as mp
from mdapy.analysis import ClusterAnalysis

# two groups of atoms far apart in a 30 Å box
rng = np.random.default_rng(0)
positions = np.vstack([rng.random((20, 3)) * 4, rng.random((10, 3)) * 4 + 15])
system = mp.System(positions, mp.Box(30.0), elements="Cu")

result = ClusterAnalysis(cutoff=3.0).compute(system)
result.sizes  # atoms in each cluster, largest first
system = system.with_columns(result)  # the column 'cluster'

clusters = ClusterAnalysis(cutoff=3.0, unwrap=True, center_of_mass=True, gyration=True)
result = clusters.compute(system)
result.center_of_mass, result.radius_of_gyration  # Å, of each cluster
whole = system.with_columns(
    positions=result.unwrapped
)  # no cluster cut by the boundaries
```

## Parameters

| Name | Type | Default | Unit | Meaning |
|---|---|---|---|---|
| `cutoff` | float | `None` | Å | atoms closer than it are bonded |
| `cutoffs` | dict | `None` | Å | instead of `cutoff`, a bond length for each pair of elements `(a, b)`, in either order; pairs not given are never bonded |
| `unwrap` | bool | `False` | | also give the positions with no cluster cut by the periodic boundaries |
| `center_of_mass` | bool | `False` | | also compute the mass and the center of mass of each cluster |
| `gyration` | bool | `False` | | also compute the gyration tensor and the radius of gyration of each cluster |

`compute(system, *, neighbors=None, progress=None, cancel=None)` takes the
arguments of [every analysis](overview.md#parameters). A given neighbor list
must reach the largest cutoff; without one, the bonds are found and used at
once, never stored. `cutoffs` need an `element` column. The masses are the
`mass` column, else those of the elements, else 1 for every atom.

## Results

A `ClusterResult`:

| Attribute | Type | Meaning |
|---|---|---|
| `cluster` | (N,) int64 | the cluster of each atom, numbered from 0 by decreasing size, equal sizes in the order of their first atom; its column `cluster` |
| `sizes` | (clusters,) int64 | the number of atoms of each cluster |
| `spanning` | (clusters,) bool | whether the cluster joins a periodic image of itself, spanning the box |
| `unwrapped` | (N, 3) float64 | positions in Å with no cluster cut by the periodic boundaries, the first atom of each cluster at its own position; with `unwrap`, else `None` |
| `masses` | (clusters,) float64 | the mass of each cluster in amu, or its number of atoms without masses; with `center_of_mass`, else `None` |
| `center_of_mass` | (clusters, 3) float64 | the center of mass of each cluster in Å; with `center_of_mass`, else `None` |
| `radius_of_gyration` | (clusters,) float64 | the radius of gyration $R_g$ of each cluster in Å; with `gyration`, else `None` |
| `gyration_tensor` | (clusters, 3, 3) float64 | the gyration tensor $G$ of each cluster in Å²; with `gyration`, else `None` |

## Errors

The constructor raises `InvalidParameter` without exactly one of `cutoff`
and `cutoffs`, for a cutoff that is not positive or an unknown element;
`compute` raises it for a neighbor list that does not fit, and
`MissingProperty` for `cutoffs` without an `element` column.

## Definition

Two atoms are bonded when closer than the cutoff of their pair of elements;
a cluster is a largest set of atoms connected by bonds. An atom without
bonds is a cluster of its own.

The unwrapped positions keep the first atom of each cluster where it is and
place every other atom at the periodic image next to the atoms it is bonded
to. With these positions $\mathbf u_i$ and the masses $m_i$ of the atoms of a
cluster of mass $M$,

$$
\mathbf c = \frac{1}{M} \sum_i m_i \mathbf u_i , \qquad
G = \frac{1}{M} \sum_i m_i (\mathbf u_i - \mathbf c)(\mathbf u_i - \mathbf c)^T , \qquad
R_g = \sqrt{\operatorname{tr} G} .
$$

A cluster that joins a periodic image of itself, such as a percolating
network or a layer through the box, has no unbroken position: its unwrapped
positions, center of mass and gyration depend on the order in which its
bonds are taken, and `spanning` marks it.

## See also

- OVITO [Cluster analysis](https://docs.ovito.org/reference/pipelines/modifiers/cluster_analysis.html).
- LAMMPS [`compute cluster/atom`](https://docs.lammps.org/compute_cluster_atom.html).
