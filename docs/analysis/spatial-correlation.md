# Spatial correlation function

`SpatialCorrelation` gives how a property of the atoms at one place goes
with a property, the same or another, at a distance r from it: the
correlation function ⟨P₁(0) P₂(r)⟩, from fast Fourier transforms on a grid
up to half the box, and on request by a direct sum over the pairs of
neighbors at short range, as OVITO's spatial correlation function does.

```python
import numpy as np

import mdapy as mp
from mdapy.analysis import SpatialCorrelation
from mdapy.build import build_crystal
from mdapy.visualization import plot

# nickel with twelve spherical copper precipitates of radius 8 Å
nickel = build_crystal("fcc", 3.55, elements="Ni", repeat=(16, 16, 16))
box = nickel.box
centers = np.random.default_rng(3).random((12, 3))
step = box.to_fractional(nickel["positions"])[:, None, :] - centers
step -= np.round(step)
inside = (np.linalg.norm(step @ box.matrix, axis=-1) < 8).any(axis=1)
alloy = nickel.with_columns(
    element=np.where(inside, "Cu", "Ni"), copper=inside.astype(float)
)

result = SpatialCorrelation("copper", direct=True, cutoff=10.0, bins=40).compute(alloy)
result.r, result.correlation  # the grid, up to half the box
result.neighbor_r, result.neighbor_correlation  # the pairs closer than 10 Å

# normalized as OVITO's plot offers it: by the pair distribution, less the
# square of the mean, over the variance
mean = result.means[0]
variance = result.mean_squares[0] - mean**2
grid = (result.correlation / result.rdf - mean**2) / variance
shells = result.neighbor_rdf > 0  # the bins of the crystal's neighbor shells
pairs = result.neighbor_correlation[shells] / result.neighbor_rdf[shells]
pairs = (pairs - mean**2) / variance
with plot.style():
    fig, ax = plot.figure("single")
    ax.plot(result.r, grid, "o-", label="grid of 3 Å")
    ax.plot(result.neighbor_r[shells], pairs, "s", label="neighbor shells")
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_xlabel("r (Å)")
    ax.set_ylabel("normalized correlation")
    ax.legend()
    plot.save(fig, "images/spatial-correlation", formats=("png",))
```

![Correlation of copper in nickel](images/spatial-correlation.png)

*Copper atoms go with copper atoms within the size of the precipitates:
the correlation of the copper fraction falls from 0.75 at the nearest
neighbors, which the atoms at the surface of a precipitate share with
nickel, to none over about 16 Å, their diameter. The grid, whose cells
are some 3 Å wide, blurs the shortest distances, which the shells of
neighbors of the direct sum resolve.*

## Parameters

`SpatialCorrelation(first, second=None, *, grid_spacing=3.0, window=True, direct=False, cutoff=5.0, bins=50)`

| Name | Type | Default | Unit | Meaning |
|---|---|---|---|---|
| `first` | str | | | a column of one value per atom, a component such as `"velocities.x"`, or an expression over the columns |
| `second` | str | `first` | | the other property; by default the autocorrelation of the first |
| `grid_spacing` | float | 3.0 | Å | the approximate size of the cells of the grid: each box vector is cut into as many cells of at least this length as fit |
| `window` | bool | `True` | | weight the atoms by a Hann window along the box vectors that are not periodic, which takes away the ringing of the edges of the box |
| `direct` | bool | `False` | | also sum the correlation over the pairs of neighbors closer than `cutoff` |
| `cutoff` | float | 5.0 | Å | the reach of the direct sum |
| `bins` | int | 50 | | the number of bins of the direct sum |

`compute(system)` takes the atoms in their box; along the periodic box
vectors atoms outside the box count in the cells they fold back into,
along the others they are left out.

## Results

A `SpatialCorrelationResult`:

| Attribute | Type | Unit | Meaning |
|---|---|---|---|
| `r` | (bins,) float64 | Å | the centers of the distance bins of the grid, each as wide as half the shortest distance between opposite faces of the box over their number |
| `correlation` | (bins,) float64 | | the correlation function from the grid |
| `rdf` | (bins,) float64 | | the radial distribution function of the same grid |
| `q` | (wavevectors,) float64 | 1/Å | the centers of the wavevector bins |
| `reciprocal` | (wavevectors,) float64 | | the correlation function in reciprocal space; of two properties equal to 1, the structure factor |
| `means` | (2,) float64 | | the mean of each property over the atoms |
| `mean_squares` | (2,) float64 | | the mean of the square of each property |
| `mean_product` | float | | the mean over the atoms of the product of the two properties |
| `neighbor_r` | (bins,) float64 or `None` | Å | the centers of the bins of the direct sum, with `direct` |
| `neighbor_correlation` | (bins,) float64 or `None` | | the correlation function summed over the pairs of neighbors, with `direct` |
| `neighbor_rdf` | (bins,) float64 or `None` | | the radial distribution function of the pairs of neighbors, with `direct` |

These are OVITO's tables `correlation-real-space`,
`correlation-real-space-rdf`, `correlation-reciprocal-space`,
`correlation-neighbor` and `correlation-neighbor-rdf`, and its attributes
`mean1`, `mean2`, `variance1`, `variance2` and `covariance`, which are the
means, the mean squares and the mean product. OVITO's plots offer three
normalizations, each a line of NumPy as above: by the radial distribution
function, `correlation / rdf`; by the covariance, divided by
`mean_product`; and the difference correlation,
`(mean_squares.sum() / 2) - correlation`. Atoms whose value is NaN are left
out of the grid and make the means NaN.

## Errors

`SpatialCorrelation` raises `InvalidParameter` for a grid spacing or a
cutoff that is not positive, and a number of bins that is not a positive
integer; `compute` raises it for a property that is not one number per
atom, a system of no atoms, and a grid spacing larger than a quarter of the
shortest distance between opposite faces of the box.

## Definition

Each property is summed over the atoms in the cells of a grid of
$n_a \times n_b \times n_c$ cells along the box vectors, $n_a$ the length
of $\mathbf a$ over the grid spacing, rounded down; an atom falls in the
cell its fractional coordinates times $n$ give, rounded toward zero. With
$\hat P_1$, $\hat P_2$ and $\hat\rho$ the discrete Fourier transforms of the
grids of the two properties and of the number of atoms,

$$
C(\mathbf q) = \frac{V}{N^2}\,\operatorname{Re}\,\hat P_1(\mathbf q)\,\hat P_2^*(\mathbf q),
\qquad
C(\mathbf r) = \frac{1}{N^2}\sum_{\mathbf q} \hat P_1(\mathbf q)\,\hat P_2^*(\mathbf q)\,e^{i\mathbf q\cdot\mathbf r},
$$

the second of which, with $|\hat\rho|^2$, gives the radial distribution
function of the grid. Both are averaged over the grid points in shells of
$|\mathbf q|$ or $|\mathbf r|$, leaving out the origin. The direct sum adds
$P_1(i)\,P_2(j)$ over the ordered pairs closer than the cutoff in bins of
its distance and divides by $N^2 / V$ times the volume of the shell of each
bin, as the radial distribution function does.

## See also

- OVITO [Spatial correlation function](https://docs.ovito.org/reference/pipelines/modifiers/correlation_function.html).
- [Radial distribution function](rdf.md) and [structure factor](structure-factor.md).
