# Combining systems

`combine` adds the atoms of one system to those of another, as OVITO's
combine datasets does: a solvent around a solute, an adsorbate on a slab,
a projectile above a target.

```python
import numpy as np

import mdapy as mp
from mdapy.build import add_vacuum, build_crystal
from mdapy.transform import combine
from mdapy.visualization import render

# a copper slab and a small gold cluster above it
slab = add_vacuum(build_crystal("fcc", 3.615, elements="Cu", repeat=(8, 8, 4)), 20.0)
gold = build_crystal("fcc", 4.078, elements="Au", repeat=(2, 2, 2))
center = slab.box.matrix.sum(axis=0) / 2
above = gold.with_columns(
    positions=gold["positions"] - gold["positions"].mean(axis=0) + center + [0, 0, 10]
)
both = combine(slab, above)  # the box of the slab, the gold atoms after the copper
render(both, "images/combine.png", view=(0.3, -1.0, 0.25), width=600, height=500)
```

![A gold cluster above a copper slab](images/combine.png)

*Thirty-two gold atoms of a cube of fcc gold added above a copper slab,
in the box of the slab.*

## Parameters

`combine(system, other)`

| Name | Type | Meaning |
|---|---|---|
| `system` | `System` | the atoms and the box that are kept |
| `other` | `System` | the atoms added after them, at their positions; its box is not used |

To combine the frames of two trajectories, combine them frame by frame,
as OVITO does with a trajectory as its second source:

<!-- skip-test -->
```python
frames = [
    combine(a, b)
    for a, b in zip(
        mp.read_trajectory("solute.dump"), mp.read_trajectory("solvent.dump")
    )
]
```

## Results

A `System` with the atoms of `system`, then those of `other`, in the box
of `system`:

| What | How it is combined |
|---|---|
| columns | those of either, `system`'s first; where one has no column, zeros: 0, false, an empty text, the element X, and missing values for other types such as lists; integers in one and floats in the other become floats |
| `id` | where `system` has ids, the added atoms are numbered on from its largest, in their order, whatever their own ids; without ids in `system`, its atoms get 0 and the added atoms keep theirs |
| `molecule` | the molecules of `other` above 0 offset by the largest of `system`; 0, no molecule, stays 0 |
| topology | the bonds, angles, dihedrals and impropers of both, those of `other` moved to its atoms; types as they are |
| info | that of `system`, without `energy` and `virial`, which no longer describe the atoms |

OVITO offsets molecule 0 as well, so that atoms of `other` in no molecule
join a molecule of `system`; it matches particle types by name, as the
elements here are.

## Errors

`combine` raises `InvalidParameter` for a column whose types in the two
cannot be joined, such as text in one and numbers in the other.
