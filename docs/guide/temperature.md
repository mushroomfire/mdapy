# Atomic temperature

`mdapy.analysis.AtomicTemperature(cutoff)` gives each atom the temperature of
the group formed by the atom and its neighbors closer than `cutoff`, from
their velocities relative to the group's center of mass. It shows hot spots,
for example around a cascade or a moving crack.

<!-- skip-test -->
```python
from mdapy.analysis import AtomicTemperature

result = AtomicTemperature(cutoff=3.0).compute(system)
result.temperature  # (N,) K
system = system.with_columns(result)  # column temperature
```

For the group $G_i$ of $n_i$ atoms with masses $m_j$ and velocities
$\mathbf v_j$,

$$
T_i = \frac{1}{3(n_i - 1)\, k_B} \sum_{j \in G_i} m_j
\lvert \mathbf v_j - \bar{\mathbf v}_i \rvert^2,
\qquad
\bar{\mathbf v}_i = \frac{\sum_{j \in G_i} m_j \mathbf v_j}{\sum_{j \in G_i} m_j} .
$$

Removing the motion of the center of mass takes three degrees of freedom, so
the kinetic energy is divided among $3(n_i - 1)$; a uniform drift of the
atoms changes nothing. Periodic images of the atom itself are not counted.
Without a given neighbor list the neighbors are found on the cell grid,
without storing a list.
Atoms without neighbors get NaN. The system needs `velocities` (Å/ps) and
either `mass` or `element` (standard atomic weights).

## Validation

`tests/analysis/test_temperature.py` checks that velocities drawn from the
Maxwell–Boltzmann distribution at 300 K give a mean of 300 K within 2 %, that
the values agree with a plain Python implementation to 10<sup>−12</sup> with
given masses and with masses of the elements, on the grid and with a given
neighbor list, that adding a uniform velocity changes nothing, and the
errors.
