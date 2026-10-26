# CHILL+

`ChillPlus` tells hexagonal ice, cubic ice, clathrate hydrates and their
interfaces from liquid water, by the correlation of the bond order of
neighboring molecules.

<!-- skip-test -->
```python
from mdapy.identify import ChillPlus

oxygen = water.filter(element="O")  # one atom per molecule
result = ChillPlus(cutoff=3.5).compute(oxygen)
result.counts()  # {'other': ..., 'hexagonal_ice': ..., 'cubic_ice': ..., ...}
```

## Parameters

| Name | Type | Default | Unit | Meaning |
|---|---|---|---|---|
| `cutoff` | float | 3.5 | Å | neighbors are the molecules closer than it; 3.5 Å for the oxygen atoms of water |

`compute(system, *, neighbors=None, progress=None, cancel=None)` takes the
arguments of [every analysis](overview.md#parameters); the system holds one
atom per molecule, for water its oxygen atoms. A given neighbor list must
reach the cutoff.

## Results

A [`StructureResult`](overview.md#results) with the types below.

## Errors

The constructor raises `InvalidParameter` for a cutoff that is not positive;
`compute` raises it for a neighbor list that does not fit.

## Definition

Each molecule $i$ gets the bond order $q_{3m}(i) = \sum_j Y_{3m}(\hat{\mathbf
r}_{ij})$ over its neighbors $j$, with the spherical harmonics $Y_{3m}$ of
degree 3, and each bond the correlation

$$
c(i, j) = \frac{\operatorname{Re} \sum_m q_{3m}(i)\, q_{3m}(j)^*}{|q_3(i)|\, |q_3(j)|}.
$$

A bond is staggered when $c < -0.8$ and eclipsed when $-0.35 < c < 0.25$. A
molecule with four neighbors is, checked from the top of the table down:

| Index | Name | Default color | Bonds |
|---|---|---|---|
| 1 | `hexagonal_ice` | light blue | 3 staggered and 1 eclipsed |
| 2 | `cubic_ice` | orange | 4 staggered |
| 3 | `interfacial_ice` | purple | 3 staggered and no eclipsed, or 2 staggered |
| 4 | `hydrate` | green | 4 eclipsed |
| 5 | `interfacial_hydrate` | yellow-green | 3 eclipsed |
| 0 | `other` | light grey | any other molecule |

!!! quote "Please cite"
    [Nguyen and Molinero (2015)](../reference/references.md#nguyen2015).
