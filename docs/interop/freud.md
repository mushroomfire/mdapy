<!-- needs: freud -->
# freud

`mdapy.interop.to_freud` gives the box and the points that the analyses of
freud take, and `from_freud` makes a `System` of them.

```python
import freud

from mdapy.build import build_crystal
from mdapy.interop import from_freud, to_freud

crystal = build_crystal("fcc", 3.615, elements="Cu", repeat=(6, 6, 6))
box, points = to_freud(crystal)
rdf = freud.density.RDF(bins=100, r_max=6.0).compute((box, points))
rdf.bin_centers[rdf.rdf.argmax()]  # 2.55 Å, the nearest neighbors
q6 = freud.order.Steinhardt(6).compute((box, points), {"num_neighbors": 12})
q6.particle_order[0]  # 0.575, the q6 of fcc
back = from_freud(box, points)
```

Both need freud (`pip install freud-analysis`).

## Parameters

| Function | Argument | Returns |
|---|---|---|
| `to_freud(system)` | a `System` in a right-handed box | a `freud.box.Box` and (N, 3) float32 points |
| `from_freud(box, points)` | a three-dimensional `freud.box.Box` and (N, 3) points | a new `System` |

## Results

freud keeps a box with its first vector along x and its second in the xy
plane, centered on the origin, and points in single precision.
`to_freud` gives that box, of the same shape and periodicity as the box of
the system, and the atoms in it: their fractional coordinates are the same,
moved into the box along periodic directions. The box and the atoms of a
system in another orientation are turned together, so distances and angles
do not change.

`from_freud` gives a box with freud's vectors, its origin at the corner of
the box centered on the origin, and the periodicity of the freud box.

The neighbor queries of freud need a box periodic in every direction.

## Errors

`to_freud` raises
[`InvalidParameter`](../reference/api.md#mdapy.errors.InvalidParameter)
for a left-handed box, and `from_freud` for a two-dimensional one.

## See also

- freud [boxes](https://freud.readthedocs.io/en/latest/modules/box.html).
