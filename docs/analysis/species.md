# Chemical species

`ChemicalSpecies` finds the molecules of a system, the groups of atoms
joined by bonds, and their chemical formulas, from the radii of the
elements or bond lengths given by hand; and counts the molecules of each
formula along a trajectory, as the products of reactions form.

```python
import numpy as np

import mdapy as mp
from mdapy.analysis import ChemicalSpecies

# two water molecules and a carbon dioxide in a 20 Å box
water = np.array([[0, 0, 0.119], [0, 0.763, -0.477], [0, -0.763, -0.477]])
co2 = np.array([[0, 0, 0], [0, 0, 1.16], [0, 0, -1.16]])
positions = np.vstack([water + 2, water + 8, co2 + 14])
system = mp.System(
    positions, mp.Box(20.0), elements=["O", "H", "H"] * 2 + ["C", "O", "O"]
)

result = ChemicalSpecies().compute(system)
result.formulas  # ['H2O', 'H2O', 'CO2']
result.counts()  # formula and count: H2O 2, CO2 1
system = system.with_columns(result)  # the column 'molecule', from 1

# by the van der Waals radii, as VMD bonds atoms
vdw = ChemicalSpecies("vdw", scale=0.6, tolerance=0.0).compute(system)
```

## Parameters

`ChemicalSpecies(radii="covalent", *, scale=1.0, tolerance=0.45, cutoffs=None)`

| Name | Type | Default | Unit | Meaning |
|---|---|---|---|---|
| `radii` | str or dict | `"covalent"` | Å | the radius of each element: `"covalent"`, the covalent radii, `"vdw"`, the van der Waals radii, or radii by element such as `{"H": 0.4, "O": 0.7}` |
| `scale` | float | 1 | | factor on the sum of the radii of two atoms |
| `tolerance` | float | 0.45 | Å | length added to it |
| `cutoffs` | dict | `None` | Å | bond lengths for pairs of elements `(a, b)`, in either order, instead of the radii; 0 for pairs never bonded |

Two atoms are bonded when they are closer than
$s(r_i + r_j) + t$, with $r$ their radii, $s$ the scale and $t$ the
tolerance, or than the cutoff of their pair of elements. The defaults,
covalent radii plus 0.45 Å, are the rule of Jmol and OpenBabel, and of the
bonds mdapy draws in [pictures](../visualization/rendering.md); the van der
Waals radii with a scale of 0.6 and no tolerance are the rule of VMD.
Hydrogen bonds and other contacts are left out by both.

`compute(system, *, neighbors=None, progress=None, cancel=None)` takes the
arguments of [every analysis](overview.md#parameters); `neighbors` must
reach the longest bond. `bond_lengths(elements)` gives the bond length of
each pair of elements.

`count(frames)` computes each frame and gives a table of the number of
molecules of each formula in each frame: a `frame` column and a column for
each formula, the most common first.

## Results

A `SpeciesResult`:

| Attribute or method | Type | Meaning |
|---|---|---|
| `molecule` | (N,) int64 | the molecule of each atom, numbered from 1 by decreasing size, ties by their first atom |
| `formulas` | (M,) str | the chemical formula of each molecule |
| `sizes` | (M,) int64 | the number of atoms of each molecule |
| `counts()` | polars `DataFrame` | `formula` and `count`, the most common first |

The column is `molecule`, the molecule ID of LAMMPS data files. Formulas
are in Hill order: carbon, then hydrogen, then the other elements
alphabetically, or all alphabetically without carbon, so that ammonia is
`H3N` and sodium chloride `ClNa`; isomers have the same formula.

## Errors

The constructor raises `InvalidParameter` for unknown radii, elements or
pairs, radii that are not positive, a scale that is not positive, a
negative tolerance or cutoff; `compute` raises `MissingProperty` without an
`element` column and `InvalidParameter` for an element without a radius,
such as `"X"`, unless cutoffs cover its pairs.

## Definition

Molecules are the [clusters](clusters.md) of atoms joined by bonds,
across periodic boundaries too. The covalent radii are those of
[Cordero et al. (2008)](../reference/references.md#cordero2008), the van der
Waals radii those of [Alvarez (2013)](../reference/references.md#alvarez2013),
both in [`mdapy.elements`](../reference/elements.md).
