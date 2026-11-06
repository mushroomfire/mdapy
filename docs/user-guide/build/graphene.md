# Graphene on grain boundaries

`mdapy.build.add_graphene(polycrystal, *, layers=1, coverage=1.0,
clearance=3.0, bond=1.42, spacing=3.35, seed=None)` puts graphene on the
grain boundaries of a polycrystal from
[`build_polycrystal`](polycrystals.md): one or more layers on each face
between two grains, or on a fraction of them.

```python
from mdapy.build import add_graphene, build_crystal, build_polycrystal

copper = build_polycrystal(build_crystal("fcc", 3.615, elements="Cu"), 60.0, 6, seed=1)
composite = add_graphene(copper, layers=2, coverage=0.6, seed=2)
composite.table["element"].value_counts()  # Cu and C
```

One layer on every boundary, two layers, and one layer on half the area of
the boundaries, each seen through a slice 10 Å thick, the atoms colored by
[polyhedral template matching](../analysis/ptm.md), which knows graphene:

```python
import numpy as np
from mdapy.identify import PolyhedralTemplateMatching
from mdapy.visualization import render, write_png

ptm = PolyhedralTemplateMatching(("fcc", "hcp", "bcc", "graphene"))
pictures = []
for layers, coverage in ((1, 1.0), (2, 1.0), (1, 0.5)):
    composite = add_graphene(copper, layers=layers, coverage=coverage, seed=2)
    structures = ptm.compute(composite)
    middle = np.abs(composite["positions.y"] - 30.0) < 5.0
    pictures.append(
        render(
            composite,
            atoms=middle,
            color=structures,
            view="front",
            legend=True,
            width=600,
            height=460,
        )
    )
write_png("images/graphene.png", np.hstack(pictures))
```

![Graphene on the boundaries of a copper polycrystal](images/graphene.png)

*A copper polycrystal of six grains with graphene: one layer on every
boundary (left), two layers (middle), and one layer on half the area of
the boundaries (right). The copper inside the grains is fcc, the carbon
graphene; the copper next to the carbon is other, as some of its
neighbors were removed to make room, until a relaxation closes the gap.*

| Name | Type | Default | Unit | Meaning |
|---|---|---|---|---|
| `polycrystal` | `System` | | | a polycrystal from `build_polycrystal`, with `info["seeds"]` and a `grain` column |
| `layers` | int32 | 1 | | the number of graphene layers on each face, AB stacked and centered on it |
| `coverage` | float64 | 1 | | the fraction of the area of the boundaries covered, from 0 to 1 |
| `clearance` | float64 | 3.0 | Å | atoms of the polycrystal closer than this to a carbon atom are removed, about the distance of graphene from copper |
| `bond` | float64 | 1.42 | Å | the carbon–carbon bond of graphene |
| `spacing` | float64 | 3.35 | Å | the distance between graphene layers |
| `seed` | int64 or `numpy.random.Generator` | | | seed of the faces chosen and of the orientations of the graphene |

Each face between two grains, the face of their Voronoi cells, is covered
once by a flat piece of graphene, rotated at random about the normal of
the face, shifted at random within it, and cut to the face; where two
faces meet, a layer off the middle ends where the planes halfway between
them cross, so that the layers of the two faces meet there. With
`coverage` below 1, whole faces are covered in a random order until that
fraction of the boundary area is. Where faces meet, the carbon atoms of a
later face closer than 0.85 bond lengths to those of an earlier one are
left out: the pieces join with their edges, without bonds made across.

## Results

The atoms of the polycrystal left, then the carbon atoms, whose `grain` is
−1; the box and `info` of the polycrystal. With no face covered, the
polycrystal itself.

## Errors

`add_graphene` raises `MissingProperty` for a system without
`info["seeds"]` or a `grain` column, and `InvalidParameter` for fewer than
one layer, a coverage outside 0 to 1, and lengths that are not positive.
