# Atomic strain

`AtomicStrain` measures the local deformation of each atom between a
reference configuration and the current one: the strain tensor, the von
Mises shear and volumetric strains, and the non-affine displacement
$D^2_\text{min}$.

```python
import numpy as np

import mdapy
from mdapy.analysis import AtomicStrain

# fcc copper of 4 x 4 x 4 cubic cells as the reference
a = 3.615
basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
cells = np.stack(np.meshgrid(*[np.arange(4)] * 3, indexing="ij"), -1).reshape(-1, 1, 3)
positions = ((cells + basis) * a).reshape(-1, 3)
reference = mdapy.System(positions, mdapy.Box(4 * a), elements="Cu")

# the crystal sheared by 2 % with its box
shear = np.array([[1, 0.02, 0], [0, 1, 0], [0, 0, 1]])
current = mdapy.System(
    positions @ shear.T, mdapy.Box(reference.box.matrix @ shear.T), elements="Cu"
)

strain = AtomicStrain(reference, cutoff=3.0)
result = strain.compute(current)
result.shear  # von Mises shear strain, the same for every atom here
current = current.with_columns(result)  # shear_strain, volumetric_strain, d2min

# many frames against one reference: find its neighbors once
neighbors = mdapy.NeighborList.build(reference, 3.0)
strain = AtomicStrain(reference, cutoff=3.0, neighbors=neighbors)
```

## Parameters

| Name | Type | Default | Unit | Meaning |
|---|---|---|---|---|
| `reference` | `System` | | | the reference configuration, with the same atoms in the same order as the systems to compare |
| `cutoff` | float | | Å | the neighbors closer than it in the reference define the neighbor vectors |
| `affine` | bool | `False` | | measure the current vectors in the reference box, which removes the homogeneous deformation of the box |
| `neighbors` | `NeighborList` | `None` | | neighbors of the reference within at least `cutoff`, used for every configuration; without, `compute` finds them as it goes each time |

`compute(system, *, progress=None, cancel=None)` takes the current
configuration: the atoms of the reference in the same order, each moved by
less than half the box. Positions may be wrapped back into the box.

## Results

A `StrainResult`:

| Attribute | Type | Unit | Meaning |
|---|---|---|---|
| `strain` | (N, 3, 3) float64 | | the Green–Lagrange strain $\eta$; NaN where the reference neighbors do not span three dimensions |
| `shear` | (N,) float64 | | the von Mises shear strain $\eta_\text{s}$; its column `shear_strain` |
| `volumetric` | (N,) float64 | | the volumetric strain $\operatorname{tr}\eta / 3$; its column `volumetric_strain` |
| `d2min` | (N,) float64 | Å² | the non-affine squared displacement $D^2_\text{min}$; its column `d2min` |

## Errors

The constructor raises `InvalidParameter` for a cutoff that is not positive;
`compute` raises it for a system with another number of atoms or other
periodic directions than the reference, or, when both have an `id` column,
atoms in another order.

## Definition

The neighbors of atom $i$ are the atoms closer than the cutoff in the
reference, periodic images included, at the vectors $\mathbf d^0_j$. Their
current vectors $\mathbf d_j$ follow from each atom's displacement, taken by
the nearest periodic image in fractional coordinates, in the current box or,
with `affine`, in the reference box. The deformation gradient minimizes
$\sum_j |\mathbf d_j - F \mathbf d^0_j|^2$:

$$
F = \Big(\sum_j \mathbf d_j \mathbf d^{0\,T}_j\Big)\Big(\sum_j \mathbf d^0_j \mathbf d^{0\,T}_j\Big)^{-1},
\qquad
\eta = \tfrac{1}{2}(F^T F - I),
$$

$$
\eta_\text{s} = \sqrt{\eta_{xy}^2 + \eta_{xz}^2 + \eta_{yz}^2
+ \tfrac{1}{6}\big[(\eta_{xx}-\eta_{yy})^2 + (\eta_{yy}-\eta_{zz})^2 + (\eta_{xx}-\eta_{zz})^2\big]},
\qquad
D^2_\text{min} = \sum_j |\mathbf d_j - F\mathbf d^0_j|^2 .
$$

## See also

- OVITO [Atomic strain](https://docs.ovito.org/reference/pipelines/modifiers/atomic_strain.html).

!!! quote "Please cite"
    [Shimizu et al. (2007)](../reference/references.md#shimizu2007); for $D^2_\text{min}$, [Falk and Langer (1998)](../reference/references.md#falk1998).
