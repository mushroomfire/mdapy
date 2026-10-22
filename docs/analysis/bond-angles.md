# Bond angle distribution

`BondAngleDistribution` counts the angles between the bonds of each atom, in
total and for each triplet of elements.

```python
import numpy as np

import mdapy
from mdapy.analysis import BondAngleDistribution

# fcc copper of 3 x 3 x 3 cubic cells
a = 3.615
basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
cells = np.stack(np.meshgrid(*[np.arange(3)] * 3, indexing="ij"), -1).reshape(-1, 1, 3)
positions = ((cells + basis) * a).reshape(-1, 3)
system = mdapy.System(positions, mdapy.Box(3 * a), elements="Cu")

angles = BondAngleDistribution(cutoff=3.0, bins=180).compute(system)
# per atom, 24, 12, 24 and 6 angles at 60, 90, 120 and 180 degrees
angles.angle, angles.counts
angles.triplet("Cu", "Cu", "Cu")  # the angles at Cu between two Cu neighbors
```

For water, `BondAngleDistribution(cutoffs={("O", "H"): 1.2})` keeps only
the O–H bonds and `triplet("O", "H", "H")` gives the H–O–H angles.

## Parameters

| Name | Type | Default | Unit | Meaning |
|---|---|---|---|---|
| `cutoff` | float | `None` | Å | bonds are shorter than it |
| `bins` | int | 180 | | number of bins of equal width from 0 to 180 degrees |
| `cutoffs` | dict | `None` | Å | instead of `cutoff`, a bond length for each pair of elements `(a, b)`, in either order; pairs not given are never bonded |

`compute(system, *, neighbors=None, progress=None, cancel=None)` takes the
arguments of [every analysis](overview.md#parameters). A given neighbor list
must reach the largest cutoff; without one, the bonds are found and used at
once, never stored. Partial counts and `cutoffs` need an `element` column.

## Results

A `BondAngleResult`:

| Attribute or method | Type | Meaning |
|---|---|---|
| `angle` | (bins,) float64 | bin centers in degrees |
| `edges` | (bins + 1,) float64 | bin edges in degrees |
| `counts` | (bins,) int64 | number of angles in each bin |
| `density` | (bins,) float64 | the counts normalized so that their integral over degrees is 1 |
| `partial` | dict | counts for each triplet `(a, b, c)`: an atom of element `a` at the vertex and neighbors of elements `b` and `c`, `b` not after `c` by atomic number; they add up to `counts` |
| `triplet(a, b, c)` | (bins,) int64 | the counts of the triplet, with `b` and `c` in either order |

## Errors

The constructor raises `InvalidParameter` without exactly one of `cutoff`
and `cutoffs`, for a cutoff that is not positive, an unknown element or
fewer than one bin; `compute` raises it for a neighbor list that does not
fit, and `MissingProperty` for `cutoffs` without an `element` column;
`triplet` raises `InvalidParameter` for a triplet without counts.

## Definition

Every two bonds $ij$ and $ik$ of an atom $i$ give the angle $\theta_{jik}$
between the vectors to $j$ and $k$, counted in the bin
$\lfloor \theta / \Delta\theta \rfloor$; an atom with $n$ bonds gives
$n(n - 1)/2$ angles. Bonds join atoms closer than the cutoff of their pair
of elements, also across periodic boundaries.

## See also

- OVITO [Bond angle distribution](https://docs.ovito.org/reference/pipelines/modifiers/bond_angle_distribution.html).
- LAMMPS [`compute adf`](https://docs.lammps.org/compute_adf.html).
