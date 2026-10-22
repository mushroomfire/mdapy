# Ackland–Jones analysis

`AcklandJonesAnalysis` gives each atom the crystal structure of its
neighborhood, fcc, hcp, bcc, icosahedral or other, from the distribution of
the angles between the bonds to its neighbors.

```python
import numpy as np

import mdapy
from mdapy.identify import AcklandJonesAnalysis

# bcc iron of 5 x 5 x 5 cubic cells
a = 2.866
basis = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
cells = np.stack(np.meshgrid(*[np.arange(5)] * 3, indexing="ij"), -1).reshape(-1, 1, 3)
positions = ((cells + basis) * a).reshape(-1, 3)
system = mdapy.System(positions, mdapy.Box(5 * a), elements="Fe")

result = AcklandJonesAnalysis().compute(system)
result.counts()  # {'other': 0, 'fcc': 0, 'hcp': 0, 'bcc': 250, 'ico': 0}
```

## Parameters

`AcklandJonesAnalysis()` has no parameters.
`compute(system, *, neighbors=None, progress=None, cancel=None)` takes the
arguments of [every analysis](overview.md#parameters). A given neighbor list
must hold the 14 nearest neighbors of every atom.

## Results

A [`StructureResult`](overview.md#results) with the types of
[common neighbor analysis](cna.md#results): `other`, `fcc`, `hcp`, `bcc` and
`ico`.

## Errors

`compute` raises `InvalidParameter` for a neighbor list that does not hold
the 14 nearest neighbors.

## Definition

The analysis takes the 14 nearest neighbors of an atom. With $r_0^2$ the
mean squared distance of the 6 nearest, $n_0$ and $n_1$ count the neighbors
closer than $\sqrt{1.45}\, r_0$ and $\sqrt{1.55}\, r_0$. The cosines of the
angles between every two of the $n_0$ bonds fall into eight bins with the
bounds −1, −0.945, −0.915, −0.755, −0.195, 0.195, 0.245, 0.795 and 1, which
count $\chi_0, \dots, \chi_7$. The deviations from the ideal structures are

$$
\begin{aligned}
\delta_\mathrm{bcc} &= \frac{0.35\, \chi_4}{\chi_5 + \chi_6 - \chi_4}, &
\delta_\mathrm{cp} &= \Big|1 - \frac{\chi_6}{24}\Big|, \\
\delta_\mathrm{fcc} &= \frac{0.61}{6} \big(|\chi_0 + \chi_1 - 6| + \chi_2\big), &
\delta_\mathrm{hcp} &= \frac{1}{12} \big(|\chi_0 - 3| + |\chi_0 + \chi_1 + \chi_2 + \chi_3 - 9|\big),
\end{aligned}
$$

with $\delta_\mathrm{bcc} = 0$ when $\chi_0 = 7$, $\delta_\mathrm{fcc} = 0$
when $\chi_0 = 6$, $\delta_\mathrm{hcp} = 0$ when $\chi_0 \le 3$, and no bcc
when $\chi_5 + \chi_6 = \chi_4$. The atom is then, checked from the top down:

| Condition | Type |
|---|---|
| $\chi_7 > 0$ | `other` |
| $\chi_4 < 3$ | `ico` when $11 \le n_1 \le 13$, else `other` |
| $\delta_\mathrm{bcc} \le \delta_\mathrm{cp}$ | `bcc` when $n_1 \ge 11$, else `other` |
| $11 \le n_1 \le 12$ | `fcc` when $\delta_\mathrm{fcc} < \delta_\mathrm{hcp}$, else `hcp` |
| otherwise | `other` |

These are the rules of the original implementation, which LAMMPS runs with
`legacy yes`. An atom with fewer than 6 neighbors, as in a system without
periodic directions of fewer than 7 atoms, is `other`.

## See also

- LAMMPS [`compute ackland/atom`](https://docs.lammps.org/compute_ackland_atom.html).

!!! quote "Please cite"
    [Ackland and Jones (2006)](../reference/references.md#ackland2006).
