# Choosing frames

A training set grows best with frames unlike those it has.
`structure_descriptors` gives each frame the mean of the NEP descriptors of
its atoms, `farthest_point_sampling` chooses the frames that cover the space
of descriptors, and `principal_components` projects them onto a plane to
look at.

<!-- skip-test -->
```python
import mdapy
from mdapy.calculators import NEP
from mdapy.mlip import (
    farthest_point_sampling,
    principal_components,
    structure_descriptors,
)
from mdapy.visualization import plot

potential = NEP("nep.txt")
train = mdapy.read_trajectory("train.xyz")
candidates = mdapy.read_trajectory("md.xyz")

old = structure_descriptors(train, potential)  # (F, D)
new = structure_descriptors(candidates, potential)

# 50 candidates far from the training set and from each other
chosen = farthest_point_sampling(new, 50, start=None, selected=old)
mdapy.write("add.xyz", [candidates[i] for i in chosen])

pca = principal_components(old, n=2)
pca.ratio  # fraction of the variance on each component
with plot.style():
    fig, ax = plot.figure()
    ax.scatter(*pca.projection.T, s=2, label="train")
    ax.scatter(*pca.transform(new[chosen]).T, s=2, label="chosen")
    ax.legend()
```

## Parameters

| Function | Parameters |
|---|---|
| `structure_descriptors(frames, calculator)` | the frames, and a calculator with `descriptors(system)` such as [NEP](../simulation/nep.md) |
| `farthest_point_sampling(points, count, *, start=0, selected=None)` | (M, D) candidates, the number chosen, the first one chosen (`None`: the farthest from `selected`), and (K, D) points chosen before |
| `principal_components(points, n=2)` | (M, D) points, two or more, and the number of components kept, at most D |

The points may be descriptors of frames or of atoms, or any other features.

## Results

`structure_descriptors` returns an (F, D) array.
`farthest_point_sampling` returns the int64 indices of the chosen points, in
the order chosen. `principal_components` returns a `PrincipalComponents`:

| Attribute | Meaning |
|---|---|
| `projection` | (M, n) coordinates of the points along the components |
| `components` | (n, D) unit vectors by decreasing variance, each with its largest entry positive |
| `variance` | (n,) variance along each component |
| `ratio` | (n,) fraction of the total variance |
| `mean` | (D,) mean of the points |
| `transform(points)` | coordinates of other points along the same components |

## Errors

`farthest_point_sampling` raises `InvalidParameter` for points that are not
(M, D), a count outside 0 to M, a start outside the points, and
`start=None` without `selected`. `principal_components` raises it for fewer
than two points and an `n` outside 1 to D.

## Definition

Farthest point sampling chooses, after the first, the point whose distance
to the nearest point already chosen (or in `selected`) is largest:

$$
i_{k+1} = \arg\max_i \min_{j \in S_k} \lVert \mathbf{x}_i - \mathbf{x}_j \rVert ,
$$

so that the chosen points spread over the whole set. The principal
components are the eigenvectors of the covariance of the points,
$\frac{1}{M-1} \sum_i (\mathbf{x}_i - \bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})^\mathsf{T}$,
by decreasing eigenvalue, the variance along each.
