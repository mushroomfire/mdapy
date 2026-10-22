# Entropy fingerprint

`mdapy.analysis.EntropyFingerprint` computes the pair entropy fingerprint of
Piaggi and Parrinello, a measure of local order that needs no reference
structure: it is low (more negative) in crystals and high in liquids and
gases.

<!-- skip-test -->
```python
from mdapy.analysis import EntropyFingerprint

result = EntropyFingerprint(cutoff=5.0, sigma=0.2, average_cutoff=3.0).compute(system)
result.entropy  # (N,) in units of the Boltzmann constant
system = system.with_columns(result)  # column entropy_average
```

For each atom $i$ a local radial distribution is built from its neighbors
closer than `cutoff`, each smeared by a Gaussian of width $\sigma$:

$$
g_i(r) = \frac{1}{4\pi\rho r^2} \sum_j \frac{1}{\sqrt{2\pi}\,\sigma}
e^{-(r - r_{ij})^2 / 2\sigma^2},
\qquad
s_i = -2\pi\rho \int_0^{r_c} \big[g_i \ln g_i - g_i + 1\big]\, r^2 \,\mathrm dr ,
$$

where $\rho$ is the number density of the system or, with
`local_density=True`, that of the atom's neighbors within the cutoff. As in
LAMMPS `compute entropy/atom`, $g_i$ is evaluated at $r = k\sigma$ up to the
cutoff and the integral is the trapezoid rule over those points; where
$g_i < 10^{-10}$ the integrand is $r^2$. With `average_cutoff`, each value is
averaged with those of the neighbors closer than it.

Unlike LAMMPS, which spreads each neighbor over seven points only, mdapy uses
the whole Gaussian, so that equivalent atoms get equal values; the two agree to
about 0.5 %.

Without a given neighbor list the neighbors are found on the cell grid and
used at once, without storing a list; a million atoms with a 5 Å cutoff take
about 0.3 s on ten threads.

The `EntropyResult` has the per-atom `entropy` and whether it is the
`average`; its column is `entropy` or `entropy_average`.

## Validation

`tests/analysis/test_entropy.py` checks agreement to 10<sup>−12</sup> with a
NumPy implementation of the formulas, with global and local density and with
averaging; that every atom of a perfect crystal gets the same value; and that
a perfect crystal, a disordered crystal and random atoms come in order of
increasing entropy. `tests/identify/test_lammps.py` compares with LAMMPS
`compute entropy/atom`, with and without averaging, to 1 %.

## References

- P. M. Piaggi and M. Parrinello, Entropy based fingerprint for local
  crystalline order, *J. Chem. Phys.* **147**, 114112 (2017).
  doi:10.1063/1.4998408
