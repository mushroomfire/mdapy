# Coordination polyhedra

`CoordinationPolyhedra` builds the convex hull of the neighbors of some
atoms, as OVITO's coordination polyhedra do: the octahedra of oxygen about
the titanium of a perovskite, the tetrahedra of a silicate. The polyhedra
come as one triangle mesh for [`render`](rendering.md), with the volume of
each.

```python
import numpy as np

from mdapy.analysis import CoordinationPolyhedra
from mdapy.build import build_crystal
from mdapy.visualization import render

srtio3 = build_crystal(
    "perovskite", 3.905, elements=["Sr", "Ti", "O"], repeat=(3, 3, 3)
)
titanium = np.asarray(srtio3["element"]) == "Ti"
octahedra = CoordinationPolyhedra(2.2).compute(srtio3, titanium)
octahedra.volumes  # 27 octahedra of 4/3 (a/2)³ = 9.93 Å³

# the octahedra in blue, strontium and oxygen as small atoms
strontium = np.asarray(srtio3["element"]) == "Sr"
render(
    srtio3,
    "images/polyhedra.png",
    atoms=~titanium,
    radius=np.where(strontium, 0.9, 0.45),
    mesh=octahedra.mesh,
    mesh_color="#4f86c6",
    mesh_alpha=0.85,
    view=(-1.0, -0.6, -0.45),
    width=600,
    height=600,
)
```

![Octahedra of a perovskite](images/polyhedra.png)

*SrTiO₃ repeated 3 × 3 × 3 times: the TiO₆ octahedra (blue), the oxygen
(red) at their corners and the strontium (green) between them. The
octahedra about the atoms on the faces of the box reach out of it, whole.
`mesh_color` also takes one color per triangle, such as that of the atom
of each, `colors[octahedra.atoms[octahedra.owners]]`.*

## Parameters

`CoordinationPolyhedra(cutoff=None, *, num_neighbors=None, bonds=False)`

| Name | Type | Default | Unit | Meaning |
|---|---|---|---|---|
| `cutoff` | float | | Å | the neighbors closer than this |
| `num_neighbors` | int | | | the nearest neighbors, this many |
| `bonds` | bool | `False` | | the atoms bonded to each, by the bonds of the [topology](../getting-started/topology.md), as OVITO does |

Give exactly one of them. `compute(system, atoms=None)` builds the
polyhedra about the atoms of a boolean mask or of indices, all by default.
Neighbors are found through the periodic boundaries with the
[neighbor lists](../getting-started/neighbor.md) of mdapy; each polyhedron
is built about its atom, whole, even where it reaches out of the box.
Neighbors in one plane make one face, cut into triangles.

## Results

A `CoordinationPolyhedraResult`:

| Attribute | Type | Meaning |
|---|---|---|
| `mesh` | `TriangleMesh` | the triangles of every polyhedron, counterclockwise seen from outside; their vertices are the neighbors next to the atom |
| `atoms` | (M,) int64 | the atom in the middle of each polyhedron |
| `owners` | (F,) int64 | the polyhedron of each triangle, an index of `atoms` |
| `volumes` | (M,) float64 | the volume of each polyhedron in Å³; 0 for fewer than four neighbors or neighbors in a plane |

## Errors

`CoordinationPolyhedra` raises `InvalidParameter` unless exactly one of
`cutoff`, `num_neighbors` and `bonds` is given; `compute` raises
`InvalidParameter` for atoms that are not a mask or indices of the system
and as [`NeighborList.build`](../getting-started/neighbor.md#building) does,
and `MissingProperty` with `bonds` for a system without bonds.
