# Neighbor lists

A `NeighborList` holds every pair of atoms closer than a cutoff, including
periodic images, in any box. Analyses and calculators take one with the
keyword `neighbors=`, so a list can be built once and shared.

```python
import numpy as np
import mdapy

rng = np.random.default_rng(0)
box = mdapy.Box([[10.0, 0, 0], [2.0, 9.0, 0], [-1.0, 1.0, 11.0]])
system = mdapy.System(box.to_cartesian(rng.random((200, 3))), box, elements="Ar")

nl = mdapy.NeighborList.build(system, cutoff=4.0)
nl.counts  # neighbors of each atom
nl.i, nl.j, nl.shifts  # the pairs
nl.distances, nl.vectors
shorter = nl.within(3.0)  # no new search
twelve = mdapy.NeighborList.nearest(system, num_neighbors=12)
```

## Building

| Method | Parameters | Result |
|---|---|---|
| `NeighborList.build(system, cutoff, *, progress=None, cancel=None)` | `cutoff` in Å, positive | every pair closer than `cutoff` |
| `NeighborList.nearest(system, num_neighbors)` | `num_neighbors`, positive | the `num_neighbors` nearest neighbors of every atom, nearest first |

Any box works: triclinic, left-handed, periodic in some directions only, and
smaller than the cutoff. An atom then sees several images of the same
neighbor, and images of itself, each as its own pair. `nearest` raises
[`InvalidParameter`](../reference/api.md#mdapy.errors.InvalidParameter) when
a system without periodic directions has too few atoms.

## Contents

| Attribute | Shape and type | Meaning |
|---|---|---|
| `offsets` | (N + 1,) int64 | the pairs of atom `i` are `offsets[i]:offsets[i + 1]` |
| `counts` | (N,) int64 | number of pairs of each atom |
| `i`, `j` | (P,) int32 | the two atoms of each pair |
| `shifts` | (P, 3) int32 | the periodic image of `j`: zero along non-periodic directions |
| `distances` | (P,) float64 | distance in Å |
| `vectors` | (P, 3) float64 | from atom `i` to the image of atom `j`, in Å |
| `cutoff` | float | every pair is closer than it |

The vector of a pair is

$$
\mathbf{r}_{ij} = \mathbf{x}_j - \mathbf{x}_i + \mathbf{n}_{ij} \mathbf{H},
$$

with $\mathbf{n}_{ij}$ the integer `shifts` and $\mathbf{H}$ the box matrix,
whose rows are the lattice vectors; this holds for the positions as stored,
inside the box or not. Every pair appears in both directions. Within an atom,
pairs are sorted by `j` and then by the image; `sorted()` sorts them by
distance instead, and `within(r)` keeps the pairs closer than a smaller
cutoff. The arrays are read-only.

## Method

The search bins the atoms into cells in fractional coordinates. Along each
lattice direction a cell is at least as thick as the cutoff where the box
allows it; a thinner box is covered by searching more cells, which enumerates
the periodic images it needs. Each atom looks at the cells within reach and
keeps the candidates whose vector, computed as above, is shorter than the
cutoff. The cells keep a copy of their atoms side by side and the atoms are
searched cell by cell, so memory is read in sequence even when the atoms come
in random order, as in the dump of a parallel run. The atoms are divided among
threads; each atom's pairs are sorted, so the result does not depend on the
number of threads.

`nearest` estimates a cutoff from the density, enlarges it by half until
every atom has enough neighbors, and keeps the nearest ones.

## Validation

The tests in `tests/neighbor/test_neighbor_list.py` check that:

- for five boxes (orthogonal, triclinic, rotated, left-handed, flat), four
  combinations of periodic directions and three cutoffs, up to larger than
  the box, the pairs are exactly those of a brute-force enumeration of
  periodic images in NumPy, with the same distances to the last bit;
- atoms outside the box, and a box smaller than the cutoff, give the right
  pairs and images;
- pairs are symmetric and sorted, and `vectors` and `distances` follow the
  formula above;
- translation, rotation and reordering of the atoms change nothing, perfect
  fcc and bcc lattices have 12, 18 and 8, 14 neighbors in their first shells,
  and a primitive cell agrees with a supercell;
- `within`, `sorted` and `nearest` agree with a new search and with the brute
  force;
- results are the same for 1, 3 and 8 threads, which
  `tests/cpp/thread_checks.cpp` also checks under ThreadSanitizer.

## Performance

On an Apple M5 laptop, 256,000 fcc atoms with a 5 Å cutoff (10.8 million
pairs) take 0.35 s on one thread and 0.11 s on ten; in random order, 0.57 s
and 0.14 s.
