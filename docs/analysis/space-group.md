# Space group

`SpaceGroup` finds the space group of a crystal with
[spglib](https://spglib.readthedocs.io): its number and symbol, its
symmetry operations, and the atoms they make equivalent. The same
operations let [phonons](../simulation/phonons.md) and
[elastic constants](../simulation/elastic-constants.md) skip the
displacements and strains that symmetry relates to others.

<!-- needs: spglib -->

```python
from mdapy.analysis import SpaceGroup
from mdapy.build import build_crystal

cell = build_crystal("l1_2", 3.75, elements=("Cu", "Au"))
result = SpaceGroup().compute(cell)
result.number, result.symbol  # 221, 'Pm-3m'
len(result.rotations)  # 48 operations
result.equivalent  # the Au corner on its own, the three Cu face centers together
```

spglib is an optional dependency: `pip install spglib`, or the extra
`symmetry` of mdapy.

## Parameters

| Name | Type | Default | Unit | Meaning |
|---|---|---|---|---|
| `tolerance` | float | 1e-5 | Å | atoms closer than it count as images of each other |

`compute(system)` takes the cell of a crystal, periodic in every direction,
of at most 2000 atoms: spglib searches its translations pair by pair, which
takes minutes for 100,000 atoms where a unit cell takes milliseconds.
Atoms of different elements are different; without an `element` column,
atoms of different types; without either, all atoms are alike. Relaxed
crystals keep their symmetry to well below the default tolerance; a
structure whose atoms have moved more, such as a crystal at a temperature,
needs a larger one to show the symmetry of its lattice.

## Results

A `SpaceGroupResult`:

| Attribute | Type | Meaning |
|---|---|---|
| `number` | int | the number of the space group in the International Tables, 1 to 230 |
| `symbol` | str | its Hermann–Mauguin symbol, such as `"Fm-3m"` |
| `rotations` | (K, 3, 3) float64 | the rotations of the K symmetry operations, on Cartesian column vectors |
| `translations` | (K, 3) float64 | their translations in Å: operation k moves the point **r** to `rotations[k] @ r + translations[k]` |
| `equivalent` | (N,) int64 | the first atom that symmetry makes each atom equivalent to |

The operations of a supercell include the translations of its lattice that
are not translations of the supercell: a cubic cell of fcc has 192, four
times the 48 of its primitive cell.

## Errors

The constructor raises `InvalidParameter` for a tolerance that is not
positive; `compute` raises it for a box that is not periodic, a system
without atoms or of more than 2000, and without spglib.

## Definition

The space group is the group of the rotations, reflections and translations
that map the crystal onto itself, each atom onto an atom of its kind.
spglib ([Togo et al. 2024](../reference/references.md#togo2024)) finds it
from the lattice and the positions, within the tolerance.
