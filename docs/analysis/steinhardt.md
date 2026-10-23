# Steinhardt parameters

`SteinhardtParameters` gives the bond-orientational order parameters $q_l$
of each atom, which tell crystals, liquids and glasses apart, plain or
averaged over the neighbors, with neighbors by number, by distance or by
Voronoi cell, optionally weighted by the areas of the Voronoi faces.

```python
import numpy as np

import mdapy as mp
from mdapy.analysis import SteinhardtParameters

# fcc copper of 3 x 3 x 3 cubic cells
a = 3.615
basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
cells = np.stack(np.meshgrid(*[np.arange(3)] * 3, indexing="ij"), -1).reshape(-1, 1, 3)
positions = ((cells + basis) * a).reshape(-1, 3)
system = mp.System(positions, mp.Box(3 * a), elements="Cu")

result = SteinhardtParameters(degrees=(4, 6), num_neighbors=12).compute(system)
result[6]  # q6 of each atom: 0.574524 in fcc
averaged = SteinhardtParameters(cutoff=3.0, average=True).compute(system)
weighted = SteinhardtParameters(voronoi=True, weighted=True).compute(system)
system = system.with_columns(result)  # columns q4 and q6
```

## Parameters

| Name | Type | Default | Unit | Meaning |
|---|---|---|---|---|
| `degrees` | sequence of int | (4, 6) | | the degrees $l$, from 0 to 20 |
| `num_neighbors` | int | `None` | | the number of nearest neighbors; 12 when neither it, `cutoff` nor `voronoi` is given |
| `cutoff` | float | `None` | Å | instead, the neighbors are closer than it |
| `voronoi` | bool | `False` | | instead, the neighbors share a face of the [Voronoi cell](voronoi.md) |
| `weighted` | bool | `False` | | with `voronoi`, weight each neighbor by the area of its face |
| `average` | bool | `False` | | average the $q_{lm}$ over the atom and its neighbors |

`compute(system, *, neighbors=None, progress=None, cancel=None)` takes the
arguments of [every analysis](overview.md#parameters). A given neighbor list
must reach the cutoff, hold the `num_neighbors` nearest neighbors of every
atom, or, with `voronoi`, be a list of
[Voronoi neighbors](../getting-started/neighbor.md#voronoi-neighbors), whose
thresholds then choose the faces; without one, every face of a cell counts.
The walls of a box that is not periodic are not neighbors.

## Results

A `SteinhardtResult`:

| Attribute or method | Type | Meaning |
|---|---|---|
| `degrees` | tuple of int | the degrees |
| `q` | (N, degrees) float64 | $q_l$ of each atom, one column per degree; 0 for an atom without neighbors |
| `average` | bool | whether the parameters are averaged |
| `result[l]` | (N,) float64 | the parameter of degree $l$ |

Its columns are `q4`, `q6`, …, or `q4_average`, … when averaged.

## Errors

The constructor raises `InvalidParameter` for a degree outside 0 to 20, more
than one of `num_neighbors`, `cutoff` and `voronoi`, either number not
positive, or `weighted` without `voronoi`; `compute` raises it for a
neighbor list that does not fit or, with `voronoi`, two atoms at the same
position, and `result[l]` for a degree not computed.

## Definition

For the $N_b(i)$ neighbors $j$ of atom $i$, with $\hat{\mathbf r}_{ij}$ the
direction to $j$ and $Y_{lm}$ the spherical harmonics,

$$
q_{lm}(i) = \frac{1}{N_b(i)} \sum_{j} Y_{lm}(\hat{\mathbf r}_{ij}), \qquad
q_l(i) = \sqrt{\frac{4\pi}{2l + 1} \sum_{m=-l}^{l} \lvert q_{lm}(i) \rvert^2 } .
$$

With `weighted`, each neighbor counts by the area $A_{ij}$ of the face it
shares with atom $i$, so that a small face, which a tiny displacement can
create or remove, adds little:

$$
q_{lm}(i) = \frac{1}{\sum_j A_{ij}} \sum_{j} A_{ij}\, Y_{lm}(\hat{\mathbf r}_{ij}) .
$$

With `average`, the $q_{lm}$ of each atom are first averaged over the atom
and its neighbors, which separates crystal structures more sharply:

$$
\bar q_{lm}(i) = \frac{1}{N_b(i) + 1} \Big( q_{lm}(i) + \sum_{j} q_{lm}(j) \Big),
$$

and $\bar q_l$ is formed from them as $q_l$, the average itself unweighted.
Values for perfect lattices:

| Lattice | Neighbors | $q_4$ | $q_6$ |
|---|---|---|---|
| fcc | 12, or Voronoi | 0.190941 | 0.574524 |
| hcp | 12, or Voronoi | 0.097222 | 0.484762 |
| bcc | 8 | 0.509175 | 0.628539 |
| bcc | 14, or Voronoi | 0.036369 | 0.510688 |
| bcc | Voronoi, weighted | 0.224025 | 0.566940 |
| icosahedron | 12 | 0 | 0.663325 |

## See also

- OVITO [Steinhardt parameters](https://docs.ovito.org/reference/pipelines/modifiers/steinhardt_parameters.html).
- LAMMPS [`compute orientorder/atom`](https://docs.lammps.org/compute_orientorder_atom.html).

!!! quote "Please cite"
    [Steinhardt et al. (1983)](../reference/references.md#steinhardt1983); with `average`, also [Lechner and Dellago (2008)](../reference/references.md#lechner2008); with `weighted`, also [Mickel et al. (2013)](../reference/references.md#mickel2013).
