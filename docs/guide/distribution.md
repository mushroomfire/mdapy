# Radial distribution and coordination

<!-- skip-test -->
```python
from mdapy.analysis import CoordinationNumber, RadialDistributionFunction

rdf = RadialDistributionFunction(r_max=8.0, bins=200).compute(system)
rdf.r, rdf.g  # bin centers in Å and g(r)
rdf.pair("Cu", "Ni")  # a partial function
rdf.coordination  # running coordination number

cn = CoordinationNumber(cutoff=3.0).compute(system)
cn.coordination  # (N,) neighbors of each atom
CoordinationNumber(cutoffs={("Si", "O"): 2.0}).compute(system)
```

## Radial distribution function

`RadialDistributionFunction(r_max, bins=200)` counts the pairs of atoms in
`bins` shells of equal width from 0 to `r_max` (Å), including periodic
images, and normalizes the counts by those of an ideal gas of the same
density:

$$
g(r_k) = \frac{V}{N^2}\, \frac{n_k}{\tfrac{4}{3}\pi\,(r_{k+1}^3 - r_k^3)},
\qquad
g_{ab}(r_k) = \frac{V}{N_a N_b}\, \frac{n^{ab}_k}{\tfrac{4}{3}\pi\,(r_{k+1}^3 - r_k^3)},
$$

where $n_k$ counts ordered pairs $(i, j)$ in shell $k$, $n^{ab}_k$ those with
$i$ of element $a$ and $j$ of element $b$, and $V$ is the volume of the box.
The partial functions satisfy $g = \sum_{a,b} x_a x_b\, g_{ab}$ with the
fractions $x_a = N_a / N$.

The `RadialDistributionResult` has the bin centers `r`, the `edges`, `g`,
the running coordination number `coordination` (the mean number of neighbors
closer than the upper edge of each bin) and the `partial` functions of each
pair of elements present, keyed `(a, b)` with `a` not after `b` by atomic
number; `pair(a, b)` takes them in either order. Partial functions need an
`element` column.

The normalization uses the volume of the box also for directions that are not
periodic, so there $g$ does not tend to 1.

Without `neighbors=`, the pairs are counted during the neighbor search and
never stored, so long distances in large systems need little memory: 8 Å in
a million-atom crystal takes under a second on ten threads.

## Structure factor

`StructureFactor(q_max, bins=200)` computes the static structure factor of a
box periodic in every direction from the wave vectors that fit the box, the
reciprocal lattice vectors $\mathbf q = h\mathbf b_1 + k\mathbf b_2 +
l\mathbf b_3$ with $0 < |\mathbf q| \le q_\text{max}$:

$$
S(\mathbf q) = \frac{1}{N} \Big| \sum_j e^{i\mathbf q \cdot \mathbf r_j} \Big|^2,
\qquad
S_{ab}(\mathbf q) = \frac{\operatorname{Re}\big[\rho_a(\mathbf q)\, \rho_b(\mathbf q)^*\big]}{\sqrt{N_a N_b}},
\quad \rho_a(\mathbf q) = \sum_{j \in a} e^{i\mathbf q \cdot \mathbf r_j},
$$

averaged over the vectors in each of `bins` shells of $|\mathbf q|$ (one of
each pair $\pm\mathbf q$, which have the same factor). The partial factors
follow Ashcroft and Thornton and recombine as
$S = \sum_{a,b} \sqrt{c_a c_b}\, S_{ab}$. The `StructureFactorResult` has the
bin centers `q` (1/Å), the `edges`, `s` (NaN in shells without vectors), the
number of `vectors` per shell and the `partial` factors; `pair(a, b)` takes
them in either order. A perfect crystal gives $N$ at its Bragg reflections and
0 elsewhere; an ideal gas gives 1.

<!-- skip-test -->
```python
from mdapy.analysis import StructureFactor

factor = StructureFactor(q_max=8.0, bins=200).compute(system)
factor.q, factor.s
```

## Coordination number

`CoordinationNumber(cutoff)` counts the neighbors of each atom closer than
`cutoff`; `CoordinationNumber(cutoffs={(a, b): r, ...})` uses a cutoff for
each pair of elements, in either order, and pairs not given are never
neighbors. The `CoordinationResult` has the per-atom `coordination` and, with
an `element` column, the `partial` counts of neighbors of each element. Its
columns are `coordination` and `coordination_<element>`.

## Validation

`tests/analysis/test_structure_factor.py` checks that perfect fcc has
exactly $N$ per Bragg vector at its 111, 200, 220 and 311 reflections and
nothing elsewhere, agreement with a NumPy sum over the wave vectors to
10<sup>−10</sup> for the total and partial factors of a disordered alloy,
that the partial factors recombine, and that an ideal gas gives 1 within 2 %.

The tests in `tests/analysis/test_distribution.py` check:

- the running coordination of perfect fcc after each of its first seven
  shells (12, 18, 42, 54, 78, 86, 134) and the position of the first peak;
- the sum rule: the counts behind $g$ add up to the number of pairs of the
  neighbor list;
- $g = 1$ to 2 % for 20 000 randomly placed atoms;
- the partial functions recombine to $g$, and a cell doubled along **a** has
  the same functions;
- coordination 12 in fcc, and 8 unlike neighbors, 6 like neighbors or 14 of
  both in the CsCl structure with cutoffs per pair of elements.
