# Rendering

`mdapy.visualization.render` draws a system by ray tracing: atoms as
spheres, optionally bonds, the box and Voronoi cells, with ambient
occlusion, shadows and outlines, in a few styles. A `Scene` takes any
spheres, cylinders and polygons for pictures made by hand.

```python
import mdapy
from mdapy.build import build_crystal, build_polycrystal
from mdapy.identify import CommonNeighborAnalysis
from mdapy.visualization import render

copper = build_crystal("fcc", 3.615, elements="Cu")
grains = build_polycrystal(copper, mdapy.Box(60.0), 6, seed=1)
structures = CommonNeighborAnalysis().compute(grains)

image = render(grains, color=structures, width=800, height=600)  # (600, 800, 4)
boundaries = render(
    grains, atoms=structures.structure != 1, color=structures, style="goodsell"
)
```

![The styles](images/styles.png)

*The styles on a polycrystal colored by common neighbor analysis: studio,
goodsell and glossy above; toon, flat and preview below.*

## Parameters

`render(system, path=None, *, atoms=None, color=None, colormap="viridis",
value_range=None, radius=None, bonds=None, bond_radius=0.25, box=True,
box_radius=None, polyhedra=None, polyhedra_alpha=0.6,
polyhedra_color="atom", polyhedra_edges=None, **options)`

| Name | Type | Default | Unit | Meaning |
|---|---|---|---|---|
| `path` | str or path | `None` | | a PNG file to write |
| `atoms` | mask or indices | all | | the atoms shown |
| `color` | | elements | | the colors of the atoms, below |
| `colormap` | str | `"viridis"` | | `viridis`, `plasma`, `coolwarm`, `jet` or `gray`, for values |
| `value_range` | (low, high) | finite range | | the values at the ends of the colormap |
| `radius` | float or (N,) | elements | Å | the radius of the atoms |
| `bonds` | float or `NeighborList` | `None` | Å | bonds between atoms closer than a cutoff, or the pairs of a list |
| `bond_radius` | float | 0.25 | Å | |
| `box`, `box_radius` | bool, float | `True`, 0.1 × radius | Å | the edges of the box |
| `polyhedra` | `VoronoiPolyhedra` | `None` | | the [Voronoi cells](../analysis/voronoi.md) of the atoms shown |
| `polyhedra_alpha` | float | 0.6 | | their opacity |
| `polyhedra_color` | str | `"atom"` | | faces colored as their atoms, or `"edges"` by their number of edges |
| `polyhedra_edges` | float | 0.05 × radius | Å | the radius of the edges of the cells, 0 for none |

`color` is one of: `None` for the colors of the elements; a single color,
`"#rrggbb"`, a name such as `"red"` or three numbers from 0 to 1; (N, 3)
colors; a column name or (N,) values, through the colormap; or a structure
result such as that of [`CommonNeighborAnalysis`](../analysis/cna.md), in
its colors. The colors and radii of the elements are those of OVITO.
Bonds are drawn as two halves in the colors of their atoms; a bond to a
periodic image is drawn as the half next to each atom only, so nothing
leaves the box.

The `options` are those of `Scene.render`:

| Name | Type | Default | Meaning |
|---|---|---|---|
| `width`, `height` | int | 1200, 900 | size in pixels |
| `view` | str or (3,) | `"iso"` | `top`, `bottom`, `front`, `back`, `left`, `right`, `iso`, or a direction of view |
| `up` | (3,) | of the view | the direction up in the picture |
| `perspective`, `fov` | bool, float | `False`, 30 | a perspective camera and its vertical angle in degrees |
| `zoom` | float | 1 | magnification of the scene framed to fill the picture |
| `style` | str | `"studio"` | one of the styles below |
| `background` | color or `None` | `"white"` | `None` for transparent |
| `progress`, `cancel` | | `None` | as in [Parallelism](../getting-started/parallelism.md#progress-and-cancellation) |

| Style | Look |
|---|---|
| `studio` | soft key and fill lights, ambient occlusion and shadows |
| `goodsell` | flat colors darkened by ambient occlusion, outlined, after D. Goodsell |
| `glossy` | shiny spheres with strong highlights |
| `toon` | cel shading in three levels, outlined |
| `flat` | flat colors, every atom outlined |
| `preview` | quick: no occlusion, no shadows, one ray per pixel |

Any setting of a style can be changed by name: `ambient`, `diffuse`,
`specular`, `shininess`, `key` and `fill` (directions of the lights,
right, up and toward the viewer), `fill_intensity`, `ao_samples`,
`ao_distance` (Å, default 3 times the median radius), `ao_strength`,
`shadows`, `shadow_strength`, `flat`, `toon_levels`, `outline_width`
(pixels), `outline_depth` (Å), `outline_color`, `fog` and `supersample`
(rays per pixel along each axis).

## Scenes

```python
import numpy as np

from mdapy.visualization import Scene

scene = Scene()
scene.add_spheres([[0, 0, 0], [2.5, 0, 0]], 1.0, [[0.8, 0.5, 0.2], [0.2, 0.5, 0.8]])
scene.add_cylinders([[0, 0, 0]], [[2.5, 0, 0]], 0.3, (0.5, 0.5, 0.5))
scene.add_triangles([[[-2, -2, -1.5], [4, -2, -1.5], [1, 3, -1.5]]], "gray", alpha=0.5)
picture = scene.render(width=400, height=300, style="glossy", view="front")
```

| Method | Adds |
|---|---|
| `add_spheres(centers, radii, colors, *, ids=None)` | spheres |
| `add_cylinders(starts, ends, radii, colors, *, ids=None)` | cylinders without caps |
| `add_triangles(vertices, colors, *, alpha=1.0, ids=None)` | (T, 3, 3) triangles, seen from both sides, opaque or not |
| `add_polygons(vertices, corner_offsets, corners, colors, *, alpha=1.0, ids=None)` | convex polygons, in the layout of `VoronoiPolyhedra` |
| `add_box(box, *, radius=0.15, color="black")` | the twelve edges of a box |

Outlines separate primitives of different `ids`. `write_png(path, image)`
writes a picture; `colormap(values, name, value_range)`,
`element_colors(elements)` and `element_radii(elements)` give the colors
and radii used by `render`.

## Results

An (height, width, 4) uint8 array of RGBA pixels in sRGB, rows from the top,
the same for any number of threads.

## Errors

They raise `InvalidParameter` for an unknown style, view, colormap, color
or option, a size that is not positive, arrays of the wrong shape, an
opacity out of [0, 1], a polygon of fewer than three corners, a negative
id or an invalid mask of atoms; `Cancelled` when cancelled.

## Definition

Every pixel traces `supersample`² rays from an orthographic or perspective
camera into a bounding volume hierarchy of the primitives. At the nearest
hit, the color of the primitive, in linear RGB, is lit by an ambient term
weighted by ambient occlusion, the fraction of `ao_samples` rays in a
cosine-weighted hemisphere that leave without hitting anything closer than
`ao_distance`, and by the diffuse and Blinn–Phong specular light of the key
light, darkened by its shadow, and of the fill light. See-through triangles
add their color and let the ray go on. An outline is drawn on the pixels
next to a different primitive that lies at least `outline_depth` closer, or
next to the background. The rays of a pixel are averaged, over the
background, into sRGB.
