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

`NeighborList.build(system, cutoff=None, *, num_neighbors=None, progress=None, cancel=None)`

| Name | Type | Default | Unit | Meaning |
|---|---|---|---|---|
| `system` | `System` | | | the atoms and the box |
| `cutoff` | float | | Å | build every pair closer than this, positive |
| `num_neighbors` | int | | | build the `num_neighbors` nearest neighbors of each atom, positive |
| `progress` | callable | `None` | | called as `progress(done, total)`, see [Parallelism](parallelism.md#progress-and-cancellation) |
| `cancel` | `CancelToken` | `None` | | stops the search |

Give exactly one of `cutoff` and `num_neighbors`.

Any box works: triclinic, left-handed, periodic in some directions only, and
thinner than the cutoff or than the distance to the farthest neighbor. An
atom then sees several images of the same neighbor, and images of itself,
each as its own pair; a single atom in a small periodic cell has a full list.

`build` raises
[`InvalidParameter`](../reference/api.md#mdapy.errors.InvalidParameter) when:

- both or neither of `cutoff` and `num_neighbors` are given, or either is not
  a positive number (an integer for `num_neighbors`);
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
| `cutoff` | float | every pair is closer than it, in Å; for nearest neighbors, the largest distance, slightly enlarged |
| `num_neighbors` | int or `None` | pairs per atom in a list of nearest neighbors, `None` in a list within a cutoff |

The arrays are read-only. `mdapy.index_dtype` is int32, or int64 in mdapy
built with [64-bit atom indices](installation.md#64-bit-atom-indices).
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
cutoff, never on the number of threads. In a list of nearest neighbors each
atom has `num_neighbors` pairs, nearest first; pairs at the same distance are
ordered by `j` and then by the image, so the equal neighbors of a perfect
crystal come in a definite order. There `j` may be among the nearest of `i`
without `i` being among the nearest of `j`.

## Methods

| Name | Returns |
|---|---|
| `counts` | (N,) int64 number of pairs of each atom |
| `i` | (P,) `mdapy.index_dtype` first atom of each pair, computed when first used |
| `vectors` | (P, 3) float64 vectors $\mathbf{r}_{ij}$ in Å, computed when first used |
| `within(cutoff)` | the pairs closer than a smaller cutoff, in their order, without a new search; the list itself for its own cutoff. From a list of nearest neighbors it gives a list within a cutoff. |
| `sorted()` | the same pairs, each atom's sorted by distance, pairs at the same distance in their order |
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
