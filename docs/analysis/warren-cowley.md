# Warren–Cowley parameters

`WarrenCowley` measures the chemical short-range order of an alloy: whether
the neighbors of each element are richer or poorer in another element than
the alloy as a whole.

```python
import numpy as np

import mdapy
from mdapy.analysis import WarrenCowley

# a CsCl-ordered CuZn crystal of 4 x 4 x 4 cubic cells
a = 2.95
basis = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
cells = np.stack(np.meshgrid(*[np.arange(4)] * 3, indexing="ij"), -1).reshape(-1, 1, 3)
positions = ((cells + basis) * a).reshape(-1, 3)
system = mdapy.System(positions, mdapy.Box(4 * a), elements=["Cu", "Zn"] * 64)

order = WarrenCowley(num_neighbors=8).compute(system)
order["Cu", "Zn"]  # -1: every neighbor of Cu is Zn
order.alpha  # rows and columns by atomic number
```

## Parameters

| Name | Type | Default | Unit | Meaning |
|---|---|---|---|---|
| `cutoff` | float | `None` | Å | neighbors are closer than it |
| `num_neighbors` | int | `None` | | instead of `cutoff`, the number of nearest neighbors |

`compute(system, *, neighbors=None, progress=None, cancel=None)` takes the
arguments of [every analysis](overview.md#parameters); the system needs an
`element` column. A given neighbor list must reach the cutoff or hold the
`num_neighbors` nearest neighbors of every atom.

## Results

A `WarrenCowleyResult`:

| Attribute or method | Type | Meaning |
|---|---|---|
| `elements` | tuple of str | the elements, by atomic number |
| `alpha` | (n, n) float64 | $\alpha_{ab}$, row $a$ and column $b$; NaN for an element whose atoms have no neighbors |
| `result[a, b]` | float | one parameter |

## Errors

The constructor raises `InvalidParameter` without exactly one of `cutoff`
and `num_neighbors`, or for one that is not positive; `compute` raises it
for a neighbor list that does not fit, and `MissingProperty` without an
`element` column; `result[a, b]` raises `InvalidParameter` for an element
not in the system.

## Definition

$$
\alpha_{ab} = 1 - \frac{p(b \mid a)}{x_b},
$$

where $p(b \mid a)$ is the fraction of atoms of element $b$ among the
neighbors of all atoms of element $a$, and $x_b$ the fraction of $b$ in the
system. It is 0 for a random alloy, negative for pairs that attract each
other and positive for pairs that avoid each other; in the CsCl structure
with 8 neighbors, $\alpha_{ab} = -1$ between unlike and 1 between like
elements.

!!! quote "Please cite"
    [Cowley (1950)](../reference/references.md#cowley1950).
