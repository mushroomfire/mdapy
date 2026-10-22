# Clusters, bond angles and short-range order

Three analyses of `mdapy.analysis` look at the bonds between neighboring
atoms.

<!-- skip-test -->
```python
from mdapy.analysis import BondAngleDistribution, ClusterAnalysis, WarrenCowley

clusters = ClusterAnalysis(cutoff=3.0).compute(system)
clusters.sizes  # atoms in each cluster, largest first
system = system.with_columns(clusters)  # column 'cluster'

angles = BondAngleDistribution(cutoff=3.0, bins=180).compute(system)
angles.angle, angles.density  # degrees, per degree
angles.triplet("O", "H", "H")  # counts of the H-O-H angles

order = WarrenCowley(num_neighbors=12).compute(system)
order["Cu", "Ni"]
```

## Clusters

`ClusterAnalysis(cutoff)` joins two atoms closer than `cutoff` (Å), across
periodic boundaries, and finds the connected groups. With
`cutoffs={(a, b): r, ...}` each pair of elements has its own bond length, and
pairs not given are never bonded. The `ClusterResult` has the per-atom
`cluster`, numbered from 0 by decreasing size (equal sizes in the order of
their first atom), and the `sizes` of the clusters. Its column is `cluster`.
An atom without bonds is a cluster of its own. Without a given neighbor list
the bonds are found on the cell grid, each once, without storing a list; the
clusters are the same.

## Bond angles

`BondAngleDistribution(cutoff, bins=180)` counts the angles $\theta_{jik}$
between every two bonds $ij$ and $ik$ of each atom $i$ shorter than `cutoff`,
in `bins` bins of equal width from 0 to 180 degrees. As for clusters,
`cutoffs={(a, b): r, ...}` gives each pair of elements its own bond length;
for water, `cutoffs={("O", "H"): 1.2}` keeps only the O–H bonds. The
`BondAngleResult` has the bin centers `angle`, the `edges`, the `counts` and
the `density`, the counts normalized so that their integral over degrees
is 1. With an element column, `partial` also holds the counts for each
triplet of elements: `partial[a, b, c]` counts the angles with an atom of
element `a` at the vertex and neighbors of elements `b` and `c`, `b` not after
`c` by atomic number, and `triplet(a, b, c)` takes `b` and `c` in either
order. The partial counts add up to `counts`. Without a given neighbor list
the bonds are found on the cell grid, without storing a list; the counts are
the same.

## Warren–Cowley parameters

`WarrenCowley(cutoff)` or `WarrenCowley(num_neighbors=...)` measures chemical
short-range order among the neighbors of each atom:

$$
\alpha_{ab} = 1 - \frac{p(b \mid a)}{c_b},
$$

where $p(b \mid a)$ is the fraction of atoms of element $b$ among the
neighbors of all atoms of element $a$, and $c_b$ is the fraction of $b$ in the
system. It is 0 for a random alloy, negative for pairs that attract and
positive for pairs that avoid each other; in the CsCl structure with 8
neighbors, $\alpha_{ab} = -1$ between unlike and $1$ between like elements. The
`WarrenCowleyResult` has the `elements`, by atomic number, and the matrix
`alpha`, row $a$ and column $b$; `result[a, b]` is one entry. An element whose
atoms have no neighbors gets NaN.

## Validation

The tests in `tests/analysis/test_bonds.py` check that:

- clusters of separated groups of atoms agree with a breadth-first search,
  in size and order; bonds cross periodic boundaries; per-pair bond lengths
  in the CsCl structure join only the like atoms;
- the bond angles of perfect fcc (24 at 60°, 12 at 90°, 24 at 120°, 6 at
  180° per atom) and bcc are counted exactly, and the counts on a disordered
  crystal equal a NumPy histogram for any number of threads;
- the counts for each triplet of elements in a disordered ternary crystal
  with a bond length per pair equal a NumPy histogram of a brute-force
  enumeration, and add up to the total; a water molecule has one H–O–H angle
  of 104.5°;
- the Warren–Cowley parameters are exactly ±1 in the CsCl structure, the
  same by cutoff and by nearest neighbors, and near 0 in a random alloy.

`tests/analysis/test_lammps.py` runs where LAMMPS is installed: on a
disordered fcc crystal broken into many clusters, every atom is in the same
cluster as with `compute cluster/atom`.

## References

- J. M. Cowley, An approximate theory of order in alloys, *Phys. Rev.* **77**,
  669 (1950). doi:10.1103/PhysRev.77.669
