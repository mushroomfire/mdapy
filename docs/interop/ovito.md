<!-- needs: ovito -->
# OVITO

`mdapy.interop.to_ovito` and `from_ovito` convert a `System` to an OVITO
data collection and back, so that a system goes through the modifiers of an
OVITO pipeline and their results come back as columns.

```python
from ovito.modifiers import CommonNeighborAnalysisModifier
from ovito.pipeline import Pipeline, StaticSource

from mdapy.build import build_crystal
from mdapy.interop import from_ovito, to_ovito

crystal = build_crystal("fcc", 3.615, elements="Cu", repeat=(5, 5, 5))
pipeline = Pipeline(source=StaticSource(data=to_ovito(crystal)))
pipeline.modifiers.append(CommonNeighborAnalysisModifier())
analyzed = from_ovito(pipeline.compute())
analyzed.table["structure"]  # 1, fcc, for every atom
analyzed.info["CommonNeighborAnalysis.counts.FCC"]  # 500
```

Any file OVITO reads becomes a System the same way:

<!-- skip-test -->
```python
from ovito.io import import_file

system = from_ovito(import_file("run.dump").compute(10))  # frame 10
```

Both need OVITO (`pip install ovito`).

## Parameters

| Function | Argument | Returns |
|---|---|---|
| `to_ovito(system)` | a `System` | a new `ovito.data.DataCollection` |
| `from_ovito(data)` | an `ovito.data.DataCollection`, such as `pipeline.compute()` | a new `System` |

## Results

| System | OVITO |
|---|---|
| `box` | simulation cell, its vectors as columns and the origin as the fourth |
| `box.pbc` | periodic boundary conditions of the cell |
| `positions` | `Position` |
| `type`, `element`, `type_label` | `Particle Type`: the `type` column, or types numbered from 1 by first appearance of the elements; elements, or type labels, are the names of the types |
| `velocities`, `forces`, `mass`, `charge` | `Velocity`, `Force`, `Mass`, `Charge` |
| `id`, `molecule`, `image` | `Particle Identifier`, `Molecule Identifier`, `Periodic Image` |
| `atomic_energies` | `Potential Energy` |
| `structure`, `csp`, `coordination`, `cluster` | `Structure Type`, `Centrosymmetry`, `Coordination`, `Cluster` |
| `rmsd`, `interatomic_distance`, `orientation` | `RMSD`, `Interatomic Distance`, `Orientation` |
| `shear_strain`, `volumetric_strain`, `d2min` | `Shear Strain`, `Volumetric Strain`, `Nonaffine Squared Displacement` |
| `voronoi_volume` | `Atomic Volume` |
| other numeric columns | properties of the same name; `movable` as 0 and 1 |
| topology | bonds, angles, dihedrals and impropers, with their types and type labels; each bond with the periodic image of its second atom that is nearest to the first |
| `info` | global attributes |

`from_ovito` always gives a `type` column, with `element` where the type
names are chemical symbols and `type_label` where they are other names.
Other properties become columns of their names in lowercase with
underscores, such as `selection` for `Selection`; properties of strings are
left out. Values keep mdapy's units both ways; OVITO itself assumes none.

## See also

- [OVITO's Python module](https://docs.ovito.org/python/).
- [mdapy and OVITO](../performance/ovito.md): timings of the analyses both
  compute.
