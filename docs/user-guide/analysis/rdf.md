# Radial distribution function

`RadialDistributionFunction` gives the radial distribution function g(r) of
a system and its partial functions for each pair of elements.

```python
import numpy as np
from mdapy.analysis import RadialDistributionFunction
from mdapy.build import build_crystal, random_alloy

# a random Cu-Ni alloy on an fcc lattice of 5 x 5 x 5 cubic cells
crystal = build_crystal("fcc", 3.6, elements="Cu", repeat=(5, 5, 5))
system = random_alloy(crystal, {"Cu": 0.5, "Ni": 0.5}, seed=0)

rdf = RadialDistributionFunction(cutoff=8.0, bins=160).compute(system)
rdf.r, rdf.g  # bin centers in Å and g(r)
rdf.pair("Cu", "Ni")  # a partial function
rdf.coordination[-1]  # mean number of neighbors closer than 8 Å
```

Plotted with [`mdapy.visualization.plot`](../visualization/plots.md), for a
larger alloy whose atoms are shaken by 0.12 Å as at a finite temperature:

```python
from mdapy.build import build_crystal, random_alloy
from mdapy.visualization import plot

alloy = random_alloy(
    build_crystal("fcc", 3.6, repeat=(6, 6, 6)), {"Cu": 1, "Ni": 1}, seed=1
)
shake = np.random.default_rng(0).normal(0, 0.12, (len(alloy), 3))
hot = alloy.with_columns(positions=alloy["positions"] + shake)
partials = RadialDistributionFunction(cutoff=8.0, bins=160).compute(hot)
with plot.style():
    fig, ax = plot.figure("single")
    plot.rdf(partials, ax=ax, partial=True)
    plot.save(fig, "images/rdf", formats=("png",))
```

![Partial radial distribution functions](images/rdf.png)

*The three partial functions of the random alloy agree: Cu and Ni share the
sites at random. The peaks are the neighbor shells of fcc at
a/√2, a, a√(3/2), a√2 and beyond.*

## Parameters

| Name | Type | Default | Unit | Meaning |
|---|---|---|---|---|
| `cutoff` | float64 | | Å | largest distance; it may exceed half the box, periodic images count |
| `bins` | int32 | 200 | | number of bins of equal width from 0 to `cutoff` |

`compute(system, *, neighbors=None, progress=None, cancel=None)` takes the
arguments of [every analysis](overview.md#parameters). A given neighbor list
must reach `cutoff`; without one, the pairs are counted as they are found,
never stored. Partial functions need an `element` column.

## Results

A `RadialDistributionResult`:

| Attribute or method | Type | Meaning |
|---|---|---|
| `r` | (bins,) float64 | bin centers in Å |
| `edges` | (bins + 1,) float64 | bin edges in Å |
| `g` | (bins,) float64 | g(r) |
| `coordination` | (bins,) float64 | mean number of neighbors closer than the upper edge of each bin |
| `partial` | dict of (str, str) to (B,) float64 | g_ab(r) of each pair of elements `(a, b)`, `a` not after `b` by atomic number |
| `pair(a, b)` | (bins,) float64 | g_ab(r), with `a` and `b` in either order |

## Errors

The constructor raises `InvalidParameter` for a `cutoff` that is not
positive or fewer than one bin; `compute` raises it for a neighbor list that
does not fit, and `pair` for a pair of elements without a function.

## Definition

With $n_k$ the number of ordered pairs $(i, j)$, $j$ any atom or periodic
image other than $i$ itself, at a distance in bin $k$ from $r_k$ to
$r_{k+1}$, and $n^{ab}_k$ those with $i$ of element $a$ and $j$ of element
$b$,

$$
g(r_k) = \frac{V}{N^2}\, \frac{n_k}{\tfrac{4}{3}\pi\,(r_{k+1}^3 - r_k^3)},
\qquad
g_{ab}(r_k) = \frac{V}{N_a N_b}\, \frac{n^{ab}_k}{\tfrac{4}{3}\pi\,(r_{k+1}^3 - r_k^3)},
$$

with $V$ the volume of the box, $N$ the number of atoms and $N_a$ that of
element $a$; a pair at distance $r$ is in bin $\lfloor r / \Delta r \rfloor$.
The partial functions add up as $g = \sum_{a,b} x_a x_b\, g_{ab}$ with the
fractions $x_a = N_a / N$. The volume is that of the box also along
directions that are not periodic, where $g$ does not tend to 1.

This is the normalization of OVITO. LAMMPS `compute rdf` divides by
$N(N - 1)$ instead of $N^2$, and by $N_a(N_a - 1)$ for like pairs, so its
values are larger by about $N / (N - 1)$.

## See also

- OVITO [Coordination analysis](https://docs.ovito.org/reference/pipelines/modifiers/coordination_analysis.html).
- LAMMPS [`compute rdf`](https://docs.lammps.org/compute_rdf.html).
