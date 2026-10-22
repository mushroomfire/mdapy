# Neighbor lists

A `NeighborList` holds pairs of atoms, including periodic images, in any box:
either every pair closer than a cutoff, or the nearest neighbors of each atom.
Analyses and calculators take one with the keyword `neighbors=`, so a list can
be built once and shared.

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

twelve = mdapy.NeighborList.build(system, num_neighbors=12)
twelve.distances.reshape(-1, 12)  # nearest first
```

## Building

| Call | Result |
|---|---|
| `NeighborList.build(system, cutoff)` | every pair closer than `cutoff` (Å) |
| `NeighborList.build(system, num_neighbors=k)` | the `k` nearest neighbors of each atom, nearest first |

Give exactly one of `cutoff` and `num_neighbors`. Both kinds accept the
keyword-only `progress=` and `cancel=` described in
[Parallelism](parallelism.md#progress-and-cancellation).

Any box works: triclinic, left-handed, periodic in some directions only, and
thinner than the cutoff or than the distance to the `k`-th neighbor. An atom
then sees several images of the same neighbor, and images of itself, each as
its own pair; a single atom in a small periodic cell has a full list. A system
without periodic directions needs more than `num_neighbors` atoms, otherwise
[`InvalidParameter`](../reference/api.md#mdapy.errors.InvalidParameter) is
raised.

## Contents

| Attribute | Shape and type | Meaning |
|---|---|---|
| `offsets` | (N + 1,) int64 | the pairs of atom `i` are `offsets[i]:offsets[i + 1]` |
| `counts` | (N,) int64 | number of pairs of each atom |
| `i`, `j` | (P,) [`mdapy.index_dtype`](#large-systems) | the two atoms of each pair |
| `shifts` | (P, 3) int8 | the periodic image of `j`: zero along non-periodic directions |
| `distances` | (P,) float64 | distance in Å |
| `vectors` | (P, 3) float64 | from atom `i` to the image of atom `j`, in Å |
| `cutoff` | float | every pair is closer than it, in Å |
| `num_neighbors` | int or `None` | pairs per atom in a list of nearest neighbors, `None` in a list within a cutoff |

The vector of a pair is

$$
\mathbf{r}_{ij} = \mathbf{x}_j - \mathbf{x}_i + \mathbf{n}_{ij} \mathbf{H},
$$

with $\mathbf{n}_{ij}$ the integer `shifts` and $\mathbf{H}$ the box matrix,
whose rows are the lattice vectors; this holds for the positions as stored,
inside the box or not. The distances are computed with this formula in the
same order of operations by both kinds of list, so a pair has the same
distance, to the last bit, in either.

In a list within a cutoff every pair appears in both directions, and the pairs
of an atom come in the order of the search, which depends on the positions,
the box and the cutoff, never on the number of threads; `sorted()` orders them
by distance. In a list of nearest neighbors the pairs of an atom come nearest
first, pairs at the same distance ordered by `j` and then by the image, so
that the equal neighbors of a perfect crystal come in a definite order; `j`
may be among the nearest of `i` without `i` being among the nearest of `j`.
`within(r)` keeps the pairs closer than a smaller cutoff, in their order; from
a list of nearest neighbors it gives a list within a cutoff. The arrays are
read-only.

## Large systems

A pair takes 15 bytes: 4 for `j`, 3 for `shifts` and 8 for `distances`; an
atom takes 8 more for `offsets`. `i` and `vectors` (4 and 24 bytes per pair)
are computed when first used and then kept. Ten million fcc atoms with a 5 Å
cutoff have 420 million pairs, 6.3 GB.

Atom indices are `mdapy.index_dtype`, int32, enough for 2,147,483,647 atoms.
For more, install mdapy built with 64-bit indices, where a pair takes 19 bytes
(see [Installation](../getting-started/installation.md#64-bit-atom-indices));
the 32-bit build raises `InvalidParameter` for such a system.

```python
mdapy.index_dtype  # dtype('int32'), or dtype('int64') in the 64-bit build
```

A shift is at most 127 boxes. Larger shifts come from positions far outside
the box, such as unwrapped coordinates after a long diffusion, or from a
cutoff more than 127 times the thickness of the box. `build` then raises
`InvalidParameter`; wrap the positions into the box first with
[`Box.wrap`](box.md).

## Method

**Within a cutoff.** The atoms are binned into cells in fractional
coordinates. Along each lattice direction a cell is half as thick as the
cutoff where the box allows it; a thinner box is covered by searching more
cells, which enumerates the periodic images it needs. The cells keep a copy of
their atoms side by side, so memory is read in sequence even when the atoms
come in random order, as in the dump of a parallel run, and the cells along
the last direction are read as one run of memory. A candidate is first
measured from the positions wrapped into the box, which is cheap, with a
margin far above rounding; the candidates that pass are measured with the
formula above, which decides. The search runs twice: the first pass counts
the pairs of each atom, and needs the formula only for candidates within the
margin of the cutoff; the second writes the pairs straight into the final
arrays, so nothing is held in between and the memory used is that of the list
plus about 70 bytes per atom for the cells.

**Nearest neighbors.** The atoms, wrapped into the box, go into a k-d tree as
in mdapy 1.x, after OVITO's nearest neighbor finder: the tree is implicit,
with no node records; each node holds its atoms side by side in memory with
its median atom in the middle, splitting the others across x, y and z in
turn, down to leaves of at most 16 atoms. A search visits the side of the
query first and the other side only when the cell there, whose distance is
kept up to date one direction at a time, can hold a nearer atom. It thus
adapts to the density: a cluster in vacuum or a slab costs no more than a
crystal. Periodic images are searched by moving the query by lattice vectors,
over exactly those images whose atoms can be nearer than the `k`-th neighbor
found so far, so boxes of any size are searched exactly. The search ranks
candidates by their distance from the wrapped positions and keeps those within
a margin of the `k`-th; their distances by the formula above then give the
final order.

Both searches divide the atoms among threads; each atom's pairs are found and
ordered by one thread alone, so the result does not depend on the number of
threads.

## Validation

The tests in `tests/neighbor/` check that:

- for five boxes (orthogonal, triclinic, rotated, left-handed, flat), four
  combinations of periodic directions and three cutoffs, up to larger than
  the box, the pairs are exactly those of a brute-force enumeration of
  periodic images in NumPy, with the same distances to the last bit; and
  that cutoffs set exactly to pair distances, or to the next number above or
  below, keep exactly the pairs of the brute force;
- for the same boxes and periodic directions, nearest neighbors from 1 to 13,
  down to a cell of 3 atoms with 13 neighbors, are exactly those of a brute
  force, in the same order and with the same distances to the last bit;
- in a perfect fcc crystal, where 12 neighbors are equally near, 1 to 19
  nearest neighbors come in the order by atom and image;
- a single atom in a primitive fcc cell has 12 neighbors at $a/\sqrt{2}$ and 6
  at $a$, and a box thinner than the distances along one direction gives 40
  neighbors exactly;
- clusters, slabs and half-filled boxes, periodic or not, give exactly the
  nearest neighbors of the brute force;
- the nearest neighbors are the first pairs of the sorted list within their
  cutoff, with the same distances;
- atoms outside the box, and a box smaller than the cutoff, give the right
  pairs and images; shifts of 127 boxes are kept and 128 raise an error;
- pairs within a cutoff are symmetric, and `vectors` and `distances` follow
  the formula above;
- translation, rotation and reordering of the atoms change nothing, perfect
  fcc and bcc lattices have 12, 18 and 8, 14 neighbors in their first shells,
  and a primitive cell agrees with a supercell;
- `within`, `sorted` and following moved atoms agree with a new search;
- the tree finds the nearest atoms of any points, inside the box, outside it
  and far away, as a brute force does;
- results are the same for 1, 3 and 8 threads, which
  `tests/cpp/thread_checks.cpp` also checks under ThreadSanitizer;
- the whole test suite passes with 64-bit indices, which CI builds and tests.

## Performance

fcc copper with $a$ = 3.615 Å in boxes of 100 × 100 cells across y and z, as
in the benchmark of mdapy 1.x, on an Apple M5 laptop (4 performance and 6
efficiency cores) with 10 threads; time to build the list from Python, the
best of two runs. The last three systems are a 255,524-atom spherical
particle in vacuum, a 1,000,188-atom slab, and a 505,094-atom box half filled
with crystal and half with gas.

| System | List | mdapy 1.0.8 | mdapy 2.0 |
|---|---|---|---|
| 1,000,000 atoms | within 5 Å | 0.14 s | 0.14 s |
| 3,400,000 atoms | within 5 Å | 0.48 s | 0.46 s |
| 10,000,000 atoms | within 5 Å | 1.70 s | 1.56 s |
| 1,000,000 atoms | 12 nearest | 0.17 s | 0.24 s |
| 3,400,000 atoms | 12 nearest | 0.53 s | 0.69 s |
| 10,000,000 atoms | 12 nearest | 1.48 s | 2.20 s |
| particle | 12 nearest | 0.04 s | 0.05 s |
| slab | 12 nearest | 0.15 s | 0.20 s |
| half crystal, half gas | 12 nearest | 0.08 s | 0.10 s |

mdapy 1.0.8 stores up to a given number of neighbors per atom and no image
shifts, takes the nearest image of each neighbor, which needs a box at least
twice as thick as the cutoff, and searches a fixed set of images for nearest
neighbors. mdapy 2.0 stores the shift of every pair, works in any box, and
orders nearest neighbors exactly; most of its extra time for nearest
neighbors goes to those exact distances and images.
