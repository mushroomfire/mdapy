# Planar faults

`PlanarFaults` finds the planar faults of fcc crystals, stacking faults and
coherent twin boundaries, from the hcp-like atoms that
[polyhedral template matching](ptm.md) finds in their {111} planes, and
estimates their areas. It gives the same types as OVITO's Identify fcc
planar faults, atom by atom.

```python
import numpy as np

from mdapy.build import build_crystal, slip
from mdapy.identify import PlanarFaults
from mdapy.transform import replicate

# fcc copper with z along [111], slipped by a/6 [11-2] above one plane
a = 3.615
cell = build_crystal(
    "fcc", a, elements="Cu", orientation=[(1, -1, 0), (1, 1, -2), (1, 1, 1)]
)
crystal = replicate(cell, (6, 4, 8))
faulted = slip(crystal, [0, a / np.sqrt(6), 0], height=0.51)

result = PlanarFaults().compute(faulted)
result.counts()["intrinsic stacking fault"]  # 96: two planes of 48 atoms
result.areas()["intrinsic stacking fault"]  # 271.6 Å², the area of the box
faulted = faulted.with_columns(result)  # planar_fault
```

## Parameters

`PlanarFaults(structures=("fcc", "hcp", "bcc"), *, rmsd_cutoff=0.1)`

| Name | Type | Default | Unit | Meaning |
|---|---|---|---|---|
| `structures` | iterable of str | `("fcc", "hcp", "bcc")` | | the structures polyhedral template matching looks for, fcc and hcp among them |
| `rmsd_cutoff` | float | 0.1 | | atoms that deviate more from their best template are other; 0 for no cutoff |

`compute(system, *, ptm=None, neighbors=None, progress=None, cancel=None)`
runs polyhedral template matching, or takes a `PTMResult` of the same
system already computed with `template_neighbors=True`; `neighbors`,
`progress` and `cancel` are those of [every analysis](overview.md#parameters).

## Results

A `PlanarFaultResult`:

| Attribute or method | Type | Meaning |
|---|---|---|
| `fault` | (N,) int8 | the type of each atom, an index into `names` |
| `names` | tuple of str | `non-hcp`, `other hcp`, `intrinsic stacking fault`, `coherent twin boundary` and `multilayer stacking fault`, numbered as in OVITO |
| `colors` | tuple of RGB | OVITO's colors of the types |
| `interatomic_distance` | (N,) float64 | the distance in Å between nearest neighbors of the lattice each atom matched |
| `counts()` | dict | the number of atoms of each type |
| `areas()` | dict | the area in Å² of the intrinsic stacking faults and of the coherent twin boundaries |

The column is `planar_fault`.

## Errors

The constructor raises `InvalidParameter` for structures without fcc and
hcp, or a negative RMSD cutoff; `compute` raises it for a `PTMResult` of
another system or without the neighbors in the order of the templates.

## Definition

An hcp-like atom has six neighbors in its basal plane, a {111} plane of the
fcc crystal, and three above and three below it. Two hcp-like atoms one
above the other are stacked, their basal planes parallel, unless an atom
lies in both their basal planes. Each hcp-like atom is first classified by
its hcp-like neighbors in its basal plane, those stacked above and below
it, and its fcc neighbors above and below:

- stacked hcp-like neighbors on one side only: an **intrinsic stacking
  fault**, two planes of hcp-like atoms;
- no stacked hcp-like neighbors, at least one hcp-like neighbor in the
  basal plane and fcc neighbors on both sides: a **coherent twin
  boundary**, one plane;
- stacked hcp-like neighbors on both sides: a **multilayer stacking
  fault**, three planes or more;
- otherwise **other hcp**, an hcp-like atom of no planar fault.

A second pass then goes through the hcp-like atoms in the order of their
indices, as OVITO does, and changes the types as it goes: a twin-boundary
or other atom whose hcp-like basal neighbors include stacking-fault atoms
and no twin-boundary atoms becomes a stacking-fault atom, and the other way
round; the stacking-fault atoms stacked on a multilayer atom become
multilayer atoms. Atoms at the edges of faults, next to partial
dislocations, may thus depend on the order of the atoms.

The area of an atom in a {111} plane is $\tfrac{\sqrt3}{2}d^2$, with $d$
the interatomic distance; a stacking fault is two planes of atoms, a twin
boundary one:

$$
A_\mathrm{ISF} = \frac{\sqrt3}{4}\sum_{i \in \mathrm{ISF}} d_i^2, \qquad
A_\mathrm{TB} = \frac{\sqrt3}{2}\sum_{i \in \mathrm{TB}} d_i^2 .
$$

Adjacent faults of different kinds can look alike: a multilayer stacking
fault may be several stacking faults or twin boundaries on neighboring
planes.

## See also

- [Polyhedral template matching](ptm.md), which finds the hcp-like atoms.
- OVITO [Identify fcc planar faults](https://docs.ovito.org/reference/pipelines/modifiers/identify_fcc_planar_faults.html).
