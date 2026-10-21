# Bond-orientational order

`mdapy.analysis.SteinhardtParameters` computes the bond-orientational order
parameters $q_l$ of Steinhardt, Nelson and Ronchetti, which tell crystals,
liquids and glasses apart and are the inputs of many solid–liquid criteria.

<!-- skip-test -->
```python
from mdapy.analysis import SteinhardtParameters

result = SteinhardtParameters(degrees=(4, 6), num_neighbors=12).compute(system)
result[6]  # (N,) q6 of each atom
result.q  # (N, 2)
SteinhardtParameters(cutoff=3.5, average=True).compute(system)
system = system.with_columns(result)  # columns q4, q6
```

For the bonds from atom $i$ to its $N_b(i)$ neighbors,

$$
q_{lm}(i) = \frac{1}{N_b(i)} \sum_{j} Y_{lm}(\hat{\mathbf r}_{ij}), \qquad
q_l(i) = \sqrt{\frac{4\pi}{2l + 1} \sum_{m=-l}^{l} \lvert q_{lm}(i) \rvert^2 } .
$$

The neighbors are the `num_neighbors` nearest (12 by default) or, with
`cutoff`, all neighbors closer than the cutoff. With `average=True`, the
$q_{lm}$ of each atom are first averaged over the atom and its neighbors
(Lechner and Dellago), which separates crystal structures more sharply:

$$
\bar q_{lm}(i) = \frac{1}{N_b(i) + 1} \sum_{k \in \{i\} \cup N(i)} q_{lm}(k) .
$$

Degrees go from 0 to 20; $q_0$ is always 1. Atoms without neighbors have
$q_l = 0$. The `SteinhardtResult` holds the `degrees`, `q` with one column per
degree, and `average`; `result[l]` is one degree. Its columns are `q4`, `q6`,
…, or `q4_average`, … when averaged.

Values for perfect lattices:

| Lattice | Neighbors | $q_4$ | $q_6$ |
|---|---|---|---|
| fcc | 12 | 0.190941 | 0.574524 |
| hcp | 12 | 0.097222 | 0.484762 |
| bcc | 8 | 0.509175 | 0.628539 |
| bcc | 14 | 0.036369 | 0.510688 |
| icosahedron | 12 | 0 | 0.663325 |

## Validation

The tests in `tests/analysis/test_steinhardt.py` check the values of the
table, $q_0 = 1$, and agreement to 10<sup>−10</sup> with the addition theorem
$q_l(i)^2 = N_b^{-2} \sum_{j,k} P_l(\hat{\mathbf r}_{ij} \cdot \hat{\mathbf r}_{ik})$,
which needs no spherical harmonics, for degrees up to 12 with and without
averaging, on a disordered crystal. They also check invariance under rotation,
the same results for any number of threads and with a given neighbor list.
`tests/identify/test_lammps.py`, where LAMMPS is installed, compares $q_2$
to $q_8$ with LAMMPS `compute orientorder/atom` to 10<sup>−10</sup>.

## References

- P. J. Steinhardt, D. R. Nelson and M. Ronchetti, Bond-orientational order in
  liquids and glasses, *Phys. Rev. B* **28**, 784 (1983).
  doi:10.1103/PhysRevB.28.784
- W. Lechner and C. Dellago, Accurate determination of crystal structures
  based on averaged local bond order parameters, *J. Chem. Phys.* **129**,
  114707 (2008). doi:10.1063/1.2977970
