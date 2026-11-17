# Generated structures

`generate_structures` makes the structures of a training set without a
model or a calculation: small periodic cells of random shape, size and
composition, some of them slabs or clusters in vacuum, whose atoms, box and
elements are moved, one structure after the other, as long as they raise the
entropy of the descriptors of the set. The structures spread over the
environments atoms can have, from compressed to expanded, ordered to
disordered, inside a solid or at its surface, rather than around a few
crystals. They complement, not replace, structures near equilibrium such as
[perturbed crystals](perturbation.md), and an existing training set can be
given for them to complement.

```python
import numpy as np
from mdapy.build import build_crystal
from mdapy.io import write
from mdapy.mlip import QUESTS, generate_structures, information_gain, perturb
from mdapy.visualization import plot

# aluminium, 16.6 Å³ per atom in its crystal, from half to twice that:
# 300 structures here, 1000 by default
result = generate_structures("Al", 300, volume=(8.3, 33.2), seed=0)
write("generated.xyz", result.structures)

# the same random cells, not moved
cells = generate_structures("Al", 300, volume=(8.3, 33.2), steps=0, seed=0)

# shaken crystals that neither set was made from
quests = QUESTS()
fcc = build_crystal("fcc", 4.05, elements="Al", repeat=(3, 3, 3))
bcc = build_crystal("bcc", 3.24, elements="Al", repeat=(3, 3, 3))
squeezed = fcc.with_box(fcc.box.matrix * 0.8, scale_positions=True)
crystals = [fcc, bcc, squeezed]
test = np.concatenate(
    [
        quests.descriptors(f)
        for c in crystals
        for f in perturb(c, 2, displacement=0.1, seed=1)
    ]
)

with plot.style():
    fig, (left, right) = plot.figure("double", ncols=2)
    # from the tenth structure on, against the atoms on a scale of doublings
    left.plot(result.atoms[9:], result.entropy[9:])
    left.set_xscale("log", base=2)
    left.set_xticks([128, 256, 512, 1024, 2048], ["128", "256", "512", "1024", "2048"])
    left.set_ylabel("Entropy of the set (nats)")
    for name, made in (("generated", result), ("random cells", cells)):
        atoms, gain = [], []
        for n in range(10, len(made.structures) + 1, 5):
            x = np.concatenate([quests.descriptors(f) for f in made.structures[:n]])
            atoms.append(len(x))
            gain.append(np.median(information_gain(test, x)))
        right.plot(atoms, gain, label=name)
    right.axhline(0, color="0.6", lw=0.8)
    right.set_ylabel("Information gain of the crystals (nats)")
    right.legend()
    for ax in (left, right):
        ax.set_xlabel("Atoms in the set")
    plot.save(fig, "images/generation", formats=("png",))
```

![The entropy of the generated set, and its coverage of shaken crystals against random cells](images/generation.png)

*Left: the entropy of the set as each structure joins it, which `result`
gives: about one and a half nats each time the set doubles. Right: what shaken fcc, bcc and compressed fcc aluminium would add to
each set, by the median [information gain](coverage.md) of their atoms: the
lower, the better the set covers them, though no structure was made from a
crystal. The sets hold cells of 2 to 32 atoms, a fifth of the random cells
slabs or clusters. The first structures go to the edges of the volumes and
shapes allowed; then the set covers the crystals better than the random
cells.*

## How many structures

The entropy of the set rises fast with its first structures, then by about
the same each time the set doubles, a little less as it grows: each structure
adds less, but the entropy never stops rising, for the set never holds every
environment there is. So the generation makes `count` structures, by
default 1000 for each element (some 7000 atoms for aluminium), and no more.
That is a start, not an end:

- the entropy in `result`, against the atoms on a scale of doublings, shows
  what the last structures still added: what a doubling of the set would
  add, at the cost of twice the calculations;
- the [information gain](coverage.md) of the structures a model must
  describe, against the set, as on the right above, tells whether the set
  covers them;
- `result.extend(500)` adds 500 structures where the generation ended, as a
  larger `count` would have made them, without making the first ones again;
  the same `seed` with a larger `count` makes the same structures from the
  start; and [farthest point sampling](selection.md) takes fewer from a set
  too large to compute.

## How it works

Each structure starts as a cell of a random number of atoms within
`atoms`, as many cells of each doubling of the number, 2 to 4, 4 to 8 and
so on, so that small cells are many and large ones few; a random volume per
atom within `volume`, drawn on a logarithmic scale; and a random shape. With
several elements, its composition is drawn uniformly over all compositions,
and each atom's element from it. The atoms are placed at random, no two
closer than `closest` times the sum of their covalent radii, also across the
periodic boundaries, or, in cells too dense for that, on a jittered grid.
A fraction `vacuum` of the cells are slabs or clusters, half of each. The
box of a slab is first stretched along its third edge and narrowed along the
others, its volume kept, until it is as thick as the reach of the
descriptor, 2.4 times twice the largest covalent radius (5.8 Å for
aluminium), so that the atoms of its surfaces have others beneath them. Once
the atoms are placed, the box is widened along its third edge, or along all
three for a cluster, so that the atoms are 10 Å farther from their images
across it, or more when the reach of the descriptor is longer. The atoms stay
in the box they were placed in, the box of the atoms, and their surfaces
give the set the environments of atoms with neighbors on one side only.

The points of the set are the descriptors of its atoms (`per="atom"`), or
their mean in each structure (`per="structure"`), each value divided by its
standard deviation over 20 random cells. The entropy of the set is that of
a Gaussian of these points: half the logarithm of the determinant of their
covariance, with a thousandth added to its diagonal. `steps` times per atom, the
structure is moved, one of:

- in two moves out of ten, the box of the atoms strained by a symmetric
  strain of components of 0.03, the atoms with it, when the volume per atom
  stays within `volume`, no face of the box is narrower than a quarter of
  its longest edge and a slab stays as thick as the reach; the vacuum keeps
  its width;
- with atoms of several elements, in one move out of ten, the elements of
  two of them swapped;
- in the other moves, an atom displaced by a Gaussian of 0.3 times the mean
  covalent radius of the elements, in a slab or cluster only within the box
  of the atoms;

and the move is taken when it raises the entropy of the set with the
structure, keeps the atoms apart and, with a `calculator`, keeps the energy
per atom at most `max_energy`. The structure is kept when it raises the
entropy of the set, and joins it.

The descriptor of an atom is in a fixed basis, on the distance of each
neighbor over the sum of their covalent radii, s = r / (R<sub>i</sub> + R<sub>j</sub>):
the sums over the neighbors of the radial functions of NEP,
f<sub>k</sub>(s) = [T<sub>k</sub>(2(s/s<sub>c</sub> − 1)² − 1) + 1] f<sub>c</sub>(s) / 2,
k < 8, with T<sub>k</sub> the Chebyshev polynomials and
f<sub>c</sub>(s) = [1 + cos(π s / s<sub>c</sub>)] / 2 up to s<sub>c</sub> = 2.4;
the sums over pairs of neighbors j, k up to s<sub>c</sub> = 1.7 of
f<sub>n</sub>(s<sub>j</sub>) f<sub>n</sub>(s<sub>k</sub>) P<sub>l</sub>(cos θ<sub>jk</sub>),
n < 4, 1 ≤ l ≤ 4; and with several elements, the sums of the first two
radial functions over the neighbors of each element: 24 values with one
element, 24 + 2 per element with several.

The generation ends after `count` structures; should 1000 candidates in a
row raise the entropy no more, or all exceed `max_energy`, it ends with those
it kept.

## Parameters

`generate_structures(elements, count=None, *, volume, atoms=(2, 32), vacuum=0.2, closest=0.75, per="atom", steps=10, reference=None, calculator=None, max_energy=None, seed=None, progress=None, cancel=None)`

| Name | Type | Default | Unit | Meaning |
|---|---|---|---|---|
| `elements` | str or sequence of str | | | the elements; with several, each structure has a random composition |
| `count` | int32 | 1000 per element | | the structures to make; the same `seed` with a larger count makes the same first ones and more |
| `volume` | tuple of float64 | | Å³ | the smallest and largest volume per atom of the box of the atoms, without vacuum, such as half and twice that of the crystal |
| `atoms` | tuple of int32 | (2, 32) | | the fewest and most atoms of a structure, with as many structures of each doubling of the number of atoms |
| `vacuum` | float64 | 0.2 | | the fraction of the random cells that are slabs, at least as thick as the reach of the descriptor, with vacuum across the third edge, or clusters, with vacuum across all three, half of each; 0 for periodic cells only |
| `closest` | float64 | 0.75 | | the shortest distance of two atoms, as a fraction of the sum of their covalent radii ([Cordero et al. 2008](../../reference/references.md#cordero2008)) |
| `per` | str | `"atom"` | | the points of the set: `"atom"`, the descriptor of each atom; `"structure"`, their mean in each structure |
| `steps` | int32 | 10 | | the moves tried on each structure per atom; 0 keeps the random cells that raise the entropy |
| `reference` | sequence of `System` | `None` | | a training set of the elements, whose atoms are in the set from the start, for the structures to complement |
| `calculator` | calculator | `None` | | computes the energy of each structure and move, with `max_energy` |
| `max_energy` | float64 | `None` | eV/atom | the highest energy per atom of a structure |
| `seed` | int64 or `numpy.random.Generator` | `None` | | seed of the random numbers |

With `max_energy`, the moves push the structures toward the bound: the
entropy rises toward the edges of what is allowed. A bound some eV per atom
above the crystal keeps the structures within what DFT converges and a model
should learn.

## Results

A `GeneratedStructures`:

| Attribute | Type | Unit | Meaning |
|---|---|---|---|
| `structures` | tuple of `System` | | the structures in the order they were kept, `count` of them |
| `atoms` | (S,) int64 | | the atoms of the first 1, 2, ... S structures |
| `entropy` | (S,) float64 | nats | the entropy of the set of the first 1, 2, ... S structures, with the `reference` if one was given |

| Method | Meaning |
|---|---|
| `extend(count, *, progress=None, cancel=None)` | these structures and `count` more, made where their generation ended, in a new `GeneratedStructures`; these stay as they are |

The structures are periodic cells with an `element` column, the atoms
wrapped into the box, without energies or forces; a slab or cluster is a
periodic cell whose atoms leave the vacuum free. The `config_type` in the
`info` of each, `"bulk"`, `"slab"` or `"cluster"`, says which it is, to
group them. The first n structures make a set of their own, so that sets of
several sizes come from one generation.

## Errors

`generate_structures` raises `InvalidParameter` for no element, an unknown
element or one without a covalent radius, an element given twice; a count,
`atoms` or `steps` that is not a non-negative integer or a pair of positive
ones; a `volume` that is not a pair of positive numbers, the smaller first;
a `vacuum` outside 0 to 1; a `closest` not above zero; an unknown `per`; `calculator` without
`max_energy` or the reverse; a
`reference` that is not a sequence of systems or holds other elements; and
volumes too small for the atoms to be placed apart in 200 cells in a row. It
raises `MissingProperty` for a reference frame without an `element` column.
`extend` raises `InvalidParameter` for a count that is not a non-negative
integer, and for structures that `generate_structures` did not make.

!!! quote "Please cite"
    [Subramanyam and Perez (2025)](../../reference/references.md#subramanyam2025)
    and [Bochkarev et al. (2026)](../../reference/references.md#bochkarev2026),
    whose method of raising the entropy of a Gaussian of the descriptors this
    follows.

## See also

- [Coverage of descriptor space](coverage.md): the entropy and the
  information gain of the QUESTS descriptor, to compare sets.
- [Perturbed structures](perturbation.md).
- [Training sets from VASP](vasp.md): computing the structures with DFT.
