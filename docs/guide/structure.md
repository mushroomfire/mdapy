# Structure identification

`mdapy.identify` finds the local crystal structure of each atom.

<!-- skip-test -->
```python
from mdapy.identify import CentroSymmetryParameter, CommonNeighborAnalysis

types = CommonNeighborAnalysis().compute(system)
types.structure  # (N,) int8, an index into types.names
types.counts()  # {'other': ..., 'fcc': ..., 'hcp': ..., 'bcc': ..., 'ico': ...}
csp = CentroSymmetryParameter(num_neighbors=12).compute(system)
system = system.with_columns(types, csp)  # columns 'structure' and 'csp'
```

## Common neighbor analysis

`CommonNeighborAnalysis(cutoff=None)` gives each atom one of the types

| Index | Name | Default color |
|---|---|---|
| 0 | `other` | light grey |
| 1 | `fcc` | green |
| 2 | `hcp` | red |
| 3 | `bcc` | blue |
| 4 | `ico` | yellow |

in a `StructureResult` with the per-atom `structure`, the `names` and the
default `colors` (RGB from 0 to 1) of the types, and `counts()`. Its column
is `structure`.

Each bond from an atom to one of its neighbors gets a signature of three
numbers: the number of neighbors the two atoms have in common, the number of
bonds among those common neighbors, and the number of bonds in the largest
connected group of those bonds. An atom with 12 neighbors is fcc when all 12
signatures are (4, 2, 1), hcp when six are (4, 2, 1) and six (4, 2, 2), and
icosahedral when all are (5, 5, 5). An atom with 14 neighbors is bcc when eight
signatures are (6, 6, 6) and six (4, 4, 4). Any other atom is `other`.

Without `cutoff`, the analysis is adaptive: each atom has its own cutoff, from
its nearest neighbors, so strained or hot crystals and any lattice constant
are handled without a parameter. For fcc, hcp and icosahedral order it takes
the 12 nearest neighbors and the cutoff

$$
r_c = \frac{1 + \sqrt 2}{2}\, \frac{1}{12} \sum_{j=1}^{12} r_j ,
$$

halfway between the first and second shells of fcc; neighbors and bonds
between them count when they are no longer than $r_c$. For bcc it takes the 14
nearest neighbors and

$$
r_c = \frac{1 + \sqrt 2}{2}\, \frac{1}{14}
\Big(\frac{2}{\sqrt 3} \sum_{j=1}^{8} r_j + \sum_{j=9}^{14} r_j\Big).
$$

With `cutoff` (in Å), the neighbors of an atom are those closer than the
cutoff, and an atom must have exactly 12 or 14 of them; the same cutoff decides
the bonds between neighbors. A cutoff between the first and second neighbor
shells is $0.854\,a$ for fcc and $1.207\,a$ for bcc of lattice constant $a$.

## Diamond structures

`IdentifyDiamond()` tells cubic from hexagonal diamond (lonsdaleite), for
silicon, germanium, carbon and other tetrahedrally bonded crystals, where
common neighbor analysis of the four nearest neighbors says nothing. The
twelve second neighbors of an atom, reached through its four nearest
neighbors, form an fcc shell in cubic diamond and an hcp shell in hexagonal
diamond, and adaptive common neighbor analysis of those twelve decides. The
`StructureResult` has the types

| Index | Name | Default color |
|---|---|---|
| 0 | `other` | light grey |
| 1 | `cubic_diamond` | blue |
| 2 | `hexagonal_diamond` | orange |

Every atom needs four neighbors. The neighbors of diamond atoms that are not
diamond themselves, which some programs mark separately, are `other` here.

## Ice and hydrates

`ChillPlus(cutoff=3.5)` applies CHILL+ to water: give it one atom per
molecule, the oxygen atoms, whose neighbors are those closer than `cutoff`
(Å). Each molecule $i$ gets $q_{3m}(i) = \sum_j Y_{3m}(\hat{\mathbf r}_{ij})$,
and each bond the correlation

$$
c(i, j) = \frac{\operatorname{Re} \sum_m q_{3m}(i)\, q_{3m}(j)^*}{|q_3(i)|\, |q_3(j)|},
$$

a bond being staggered when $c < -0.8$ and eclipsed when $-0.35 < c < 0.25$.
A molecule with four neighbors is

| Index | Name | Bonds |
|---|---|---|
| 0 | `other` | any other molecule |
| 1 | `hexagonal_ice` | 3 staggered and 1 eclipsed |
| 2 | `cubic_ice` | 4 staggered |
| 3 | `interfacial_ice` | 3 staggered and no eclipsed, or 2 staggered |
| 4 | `hydrate` | 4 eclipsed |
| 5 | `interfacial_hydrate` | 3 eclipsed |

checked from the top of the table down, as in the original method; the
result is a `StructureResult`.

## Centro-symmetry parameter

`CentroSymmetryParameter(num_neighbors=12)` measures how far the neighbors of
an atom are from coming in opposite pairs. Of the $N$ nearest neighbors
($N$ even: 12 for fcc, 8 for bcc), every pair $(a, b)$ gives
$|\mathbf r_a + \mathbf r_b|^2$, and the parameter is the sum of the $N/2$
smallest of these values, in Å². It is zero in a perfect fcc or bcc crystal,
$a^2$ for the twelve neighbors of ideal hcp of lattice constant $a$, and large
at surfaces, dislocation cores and stacking faults. The result is a
`CentroSymmetryResult` with the per-atom `csp`, which is also its column.

The sum of the smallest values is the definition of LAMMPS `compute
centro/atom`; it can pair one neighbor twice, which a pairing of each neighbor
with exactly one other would not.

## Validation

The tests in `tests/identify/test_structure.py` check that:

- perfect fcc, hcp and bcc crystals are identified completely by both forms,
  also when rotated; the center of an icosahedral cluster is `ico`; diamond is
  `other`; a vacancy turns exactly its twelve neighbors to `other`; a one-atom
  primitive fcc cell is fcc;
- on crystals with random displacements, where some atoms are no longer
  identified, the adaptive analysis agrees atom by atom with a plain Python
  implementation written from the definitions;
- perfect cubic and hexagonal diamond are identified completely, also when
  rotated, fcc and bcc are not diamond, and on disordered diamonds the types
  agree atom by atom with a plain Python implementation;
- the oxygen atoms of perfect cubic and hexagonal ice are identified
  completely, and on hot ice and random points, where every type occurs, the
  types agree with a Python implementation that computes the correlations
  from the addition theorem, $\sum_m q_{3m}(i)\, q_{3m}(j)^* = \frac{7}{4\pi}
  \sum_{a, b} P_3(\hat{\mathbf r}_a \cdot \hat{\mathbf r}_b)$, without
  spherical harmonics;
- the centro-symmetry parameter is zero for perfect fcc and bcc, exactly
  $a^2$ for hcp, agrees with a plain Python implementation to 10<sup>−12</sup>
  on disordered crystals and does not change under rotation;
- results are the same for any number of threads and with a given neighbor
  list.

`tests/identify/test_lammps.py` runs where LAMMPS is installed and compares
with LAMMPS stable_30Sep2026 on disordered fcc, hcp and bcc crystals where
only part of the atoms keep their structure: the conventional analysis gives
the type of `compute cna/atom` for every atom, and the centro-symmetry
parameter agrees with `compute centro/atom` to 10<sup>−10</sup>.

## References

- J. D. Honeycutt and H. C. Andersen, Molecular dynamics study of melting and
  freezing of small Lennard-Jones clusters, *J. Phys. Chem.* **91**, 4950
  (1987). doi:10.1021/j100303a014
- A. Stukowski, Structure identification methods for atomistic simulations of
  crystalline materials, *Modelling Simul. Mater. Sci. Eng.* **20**, 045021
  (2012). doi:10.1088/0965-0393/20/4/045021
- E. Maras, O. Trushin, A. Stukowski, T. Ala-Nissila and H. Jónsson, Global
  transition path search for dislocation formation in Ge on Si(001),
  *Comput. Phys. Commun.* **205**, 13 (2016). doi:10.1016/j.cpc.2016.04.001
- A. H. Nguyen and V. Molinero, Identification of clathrate hydrates, hexagonal
  ice, cubic ice, and liquid water in simulations: the CHILL+ algorithm,
  *J. Phys. Chem. B* **119**, 9369 (2015). doi:10.1021/jp510289t
- C. L. Kelchner, S. J. Plimpton and J. C. Hamilton, Dislocation nucleation
  and defect structure during surface indentation, *Phys. Rev. B* **58**,
  11085 (1998). doi:10.1103/PhysRevB.58.11085
