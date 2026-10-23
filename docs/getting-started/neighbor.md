# Neighbor lists

A `NeighborList` holds pairs of atoms, including periodic images, in any box:
every pair closer than a cutoff, the nearest neighbors of each atom, or the
atoms that share a face of their Voronoi cells.
Analyses and calculators take one with the keyword `neighbors=`, so a list can
be built once and shared: it serves every system with the same positions in
the same box, whatever their other columns.

```python
import numpy as np
import mdapy as mp

rng = np.random.default_rng(0)
box = mp.Box([[10.0, 0, 0], [2.0, 9.0, 0], [-1.0, 1.0, 11.0]])
system = mp.System(box.to_cartesian(rng.random((200, 3))), box, elements="Ar")

nl = mp.NeighborList.build(system, cutoff=4.0)
nl.counts  # neighbors of each atom
nl.i, nl.j, nl.shifts  # the pairs
nl.distances, nl.vectors
shorter = nl.within(3.0)  # no new search

twelve = mp.NeighborList.build(system, num_neighbors=12)
twelve.distances.reshape(-1, 12)  # nearest first

faces = mp.NeighborList.build(system, voronoi=True)
faces.areas  # Å², the face that each pair shares
```

## Building

`NeighborList.build(system, cutoff=None, *, num_neighbors=None, voronoi=False, face_threshold=0, relative_face_threshold=0, edge_threshold=0, progress=None, cancel=None)`

| Name | Type | Default | Unit | Meaning |
|---|---|---|---|---|
| `system` | `System` | | | the atoms and the box |
| `cutoff` | float | | Å | build every pair closer than this, positive |
| `num_neighbors` | int | | | build the `num_neighbors` nearest neighbors of each atom, positive |
| `voronoi` | bool | `False` | | build the [Voronoi neighbors](#voronoi-neighbors) of each atom |
| `face_threshold`, `relative_face_threshold`, `edge_threshold` | float | 0 | Å², , Å | with `voronoi`, the faces that count, as in the [Voronoi analysis](../analysis/voronoi.md#parameters) |
| `progress` | callable | `None` | | called as `progress(done, total)`, see [Parallelism](parallelism.md#progress-and-cancellation) |
| `cancel` | `CancelToken` | `None` | | stops the search |

Give exactly one of `cutoff`, `num_neighbors` and `voronoi`.

Any box works: triclinic, left-handed, periodic in some directions only, and
thinner than the cutoff or than the distance to the farthest neighbor. An
atom then sees several images of the same neighbor, and images of itself,
each as its own pair; a single atom in a small periodic cell has a full list.

`build` raises
[`InvalidParameter`](../reference/api.md#mdapy.errors.InvalidParameter) when:

- not exactly one of `cutoff`, `num_neighbors` and `voronoi` is given, or
  `cutoff` or `num_neighbors` is not a positive number (an integer for
  `num_neighbors`);
- a threshold is negative, `relative_face_threshold` is 1 or more, or a
  threshold is given without `voronoi`;
- with `voronoi`, two atoms are at the same position;
- a system without periodic directions has `num_neighbors` atoms or fewer;
- a pair is more than 127 boxes away, which comes from positions far outside
  the box, such as unwrapped coordinates after a long diffusion; wrap them
  into the box first with [`Box.wrap`](box.md).

## Contents

| Attribute | Shape and type | Meaning |
|---|---|---|
| `system` | `System` | the system the list was built for |
| `offsets` | (N + 1,) int64 | the pairs of atom `i` are `offsets[i]:offsets[i + 1]` |
| `j` | (P,) `mdapy.index_dtype` | the second atom of each pair |
| `shifts` | (P, 3) int8 | the periodic image of `j`: zero along non-periodic directions |
| `distances` | (P,) float64 | distance in Å |
| `areas` | (P,) float64 or `None` | in a list of Voronoi neighbors, the area in Å² of the face of each pair; `None` otherwise |
| `cutoff` | float | every pair is closer than it, in Å; for nearest and Voronoi neighbors, the largest distance, slightly enlarged |
| `num_neighbors` | int or `None` | pairs per atom in a list of nearest neighbors, `None` otherwise |
| `voronoi` | bool | whether the list holds Voronoi neighbors |

The arrays are read-only. `mdapy.index_dtype` is int32, or int64 in mdapy
built with [64-bit atom indices](installation.md#64-bit-atom-indices). A list
keeps these arrays only, 15 bytes per pair with int32 indices; `i` and
`vectors` (below) are computed when asked for and not kept.
The vector from atom `i` to the image of atom `j` is

$$
\mathbf{r}_{ij} = \mathbf{x}_j - \mathbf{x}_i + \mathbf{n}_{ij} \mathbf{H},
$$

with $\mathbf{n}_{ij}$ the `shifts` and $\mathbf{H}$ the box matrix, whose
rows are the lattice vectors, for the positions as stored, inside the box or
not. Both kinds of list compute a distance the same way, so a pair has the
same distance, to the last bit, in either.

In a list within a cutoff every pair appears in both directions. The pairs of
an atom come in an order that depends on the positions, the box and the
cutoff, never on the number of threads, until an analysis puts the nearest
first (below). In a list of nearest neighbors each
atom has `num_neighbors` pairs, nearest first; pairs at the same distance are
ordered by `j` and then by the image, so the equal neighbors of a perfect
crystal come in a definite order. There `j` may be among the nearest of `i`
without `i` being among the nearest of `j`.

## Voronoi neighbors

With `voronoi=True`, the neighbors of an atom are the atoms across the faces
of its [Voronoi cell](../analysis/voronoi.md), periodic images included, one
pair for each face that counts, with the area of the face in `areas`. The
walls that close a box in a direction that is not periodic are not
neighbors. A face counts for each of its two atoms separately: with a
`relative_face_threshold`, which depends on the surface of each cell, a pair
may be in the list of one atom only; without it every pair appears in both
directions, with the same area. The pairs of an atom come in an order that
depends on the positions and the box only.

A list of Voronoi neighbors goes to the analyses that take one, such as the
[Steinhardt parameters](../analysis/steinhardt.md) with `voronoi=True`; the
analyses of the neighbors within a cutoff or of the nearest neighbors refuse
it, since it holds neither.

## Lists given to analyses

An analysis given a list with `neighbors=` reads it where it is, without a
copy. An analysis of the neighbors within a cutoff takes a list with that
cutoff or a longer one, and leaves out the pairs farther than its own.

An analysis of the k nearest neighbors of each atom takes a list with at
least k pairs per atom. From a list within a cutoff, the first such
analysis puts the k nearest pairs of each atom first, in the order of a list
of nearest neighbors, in the list itself: the pairs of each atom stay the
same and change their order, in `j`, `shifts` and `distances` together. The list keeps the number sorted, so a later analysis of as many
neighbors or fewer finds them in place, and one of more sorts a few more.
Sums of floating-point numbers over the pairs of an atom, such as the
Steinhardt parameters within a cutoff, may then differ in their last bits
from those of the same list before. A list of nearest neighbors is in that
order already. Read the arrays of a list after the analyses, not before.

One kernel at a time reads a list and none while it is sorted, so analyses
running in several threads may share one.

## Methods

| Name | Returns |
|---|---|
| `counts` | (N,) int64 number of pairs of each atom |
| `i` | (P,) `mdapy.index_dtype` first atom of each pair, a new array made from `offsets` on each access |
| `vectors` | (P, 3) float64 vectors $\mathbf{r}_{ij}$ in Å, a new array computed from the positions and `shifts` on each access |
| `within(cutoff)` | the pairs closer than a smaller cutoff, in their order, without a new search; the list itself for its own cutoff. From a list of nearest neighbors it gives a list within a cutoff; a list of Voronoi neighbors refuses it. |
| `sorted()` | the same pairs, each atom's sorted by distance, pairs at the same distance by atom and then image, as in a list of nearest neighbors; a list of Voronoi neighbors keeps its areas |
| `fits(system)` | whether the list serves a system: the one it was built for, or any with equal positions in an equal box |
| `len(nl)` | the number of pairs P |

The arrays work directly with NumPy, for example to sum over the neighbors of
each atom or to look at the pairs of one atom:

```python
weights = np.exp(-nl.distances)
per_atom = np.bincount(nl.i, weights=weights, minlength=len(system))

pairs = slice(nl.offsets[5], nl.offsets[6])
nl.j[pairs], nl.distances[pairs]  # the neighbors of atom 5

ordered = nl.sorted()
nearest = ordered.j[ordered.offsets[:-1][nl.counts > 0]]  # each atom's nearest
```
