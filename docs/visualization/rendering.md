# Rendering

`mdapy.visualization.render` draws a system by ray tracing: atoms as
spheres, optionally bonds, the box and Voronoi cells, with ambient
occlusion, shadows and outlines, in a few styles. A `Scene` takes any
spheres, cylinders and polygons for pictures made by hand.

```python
import mdapy as mp
from mdapy.build import build_crystal, build_polycrystal
from mdapy.identify import CommonNeighborAnalysis
from mdapy.visualization import render

copper = build_crystal("fcc", 3.615, elements="Cu")
grains = build_polycrystal(copper, mp.Box(60.0), 6, seed=1)
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
value_range=None, representation="spheres", radius=None, bonds=None,
bond_radius=None, box=True,
box_radius=None, polyhedra=None, polyhedra_alpha=0.6,
polyhedra_color="atom", polyhedra_edges=None, **options)`

| Name | Type | Default | Unit | Meaning |
|---|---|---|---|---|
| `path` | str or path | `None` | | a PNG file to write |
| `atoms` | mask or indices | all | | the atoms shown |
| `color` | | elements | | the colors of the atoms, below |
| `colormap` | str | `"viridis"` | | `viridis`, `plasma`, `coolwarm`, `jet` or `gray`, for values |
| `value_range` | (low, high) | finite range | | the values at the ends of the colormap |
| `representation` | str | `"spheres"` | | how atoms and bonds are drawn, below; it sets the defaults of `radius`, `bonds` and `bond_radius` |
| `radius` | float, (N,) or str | of the representation | Å | the radius of the atoms, or those of the elements: `"display"`, `"covalent"` or `"vdw"` |
| `bonds` | float, `NeighborList` or `"covalent"` | of the representation | Å | bonds between atoms closer than a cutoff, the pairs of a list, or covalent bonds |
| `bond_radius` | float | of the representation | Å | |
| `box`, `box_radius` | bool, float | `True`, 0.1 × radius | Å | the edges of the box |
| `polyhedra` | `VoronoiPolyhedra` | `None` | | the [Voronoi cells](../analysis/voronoi.md) of the atoms shown |
| `polyhedra_alpha` | float | 0.6 | | their opacity |
| `polyhedra_color` | str | `"atom"` | | faces colored as their atoms, or `"edges"` by their number of edges |
| `polyhedra_edges` | float | 0.05 × radius | Å | the radius of the edges of the cells, 0 for none |

| Representation | Atoms | Bonds |
|---|---|---|
| `spheres` | display radii | none, radius 0.25 Å when given |
| `ball_and_stick` | a quarter of the van der Waals radii | covalent, radius 0.15 Å |
| `space_filling` | van der Waals radii | none |
| `licorice` | as wide as the bonds | covalent, radius 0.2 Å |

`color` is one of: `None` for the colors of the elements; a single color,
`"#rrggbb"`, a name such as `"red"` or three numbers from 0 to 1; (N, 3)
colors; a column name or (N,) values, through the colormap; or a structure
result such as that of [`CommonNeighborAnalysis`](../analysis/cna.md), in
its colors. The colors and display radii of the elements are those of
OVITO, and for the elements OVITO leaves out the colors of Jmol and the
covalent radii; covalent radii are those of
[Cordero et al. (2008)](../reference/references.md#cordero2008), van der
Waals radii those of [Alvarez (2013)](../reference/references.md#alvarez2013),
1.5 Å and 2 Å for elements without one. Covalent bonds,
`covalent_bonds(system)`, join the atoms closer than the sum of their
covalent radii and 0.45 Å, as in Jmol.
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
| `device` | str | `"cpu"` | `"cpu"`, `"gpu"`, or `"auto"` for the GPU when there is one; see below |
| `progress`, `cancel` | | `None` | as in [Parallelism](../getting-started/parallelism.md#progress-and-cancellation) |

| Style | Look |
|---|---|
| `studio` | soft key and fill lights, ambient occlusion and shadows |
| `goodsell` | flat colors darkened by ambient occlusion, outlined, after D. Goodsell |
| `glossy` | shiny spheres with strong highlights |
| `toon` | cel shading in three levels, outlined |
| `flat` | flat colors, every atom outlined |
| `preview` | quick: no occlusion, no shadows, one ray per pixel |
| `photo` | global illumination: light that bounces between the atoms from a bright sky and a wide key light, with soft shadows; about ten times slower than `studio` |

Any setting of a style can be changed by name: `ambient`, `diffuse`,
`specular`, `shininess`, `key` and `fill` (directions of the lights,
right, up and toward the viewer), `fill_intensity`, `ao_samples`,
`ao_distance` (Å, default 3 times the median radius), `ao_strength`,
`shadows`, `shadow_strength`, `flat`, `toon_levels`, `outline_width`
(pixels), `outline_depth` (Å), `outline_color`, `fog`, `supersample`
(rays per pixel along each axis), and for global illumination `bounces`
(0 for none), `samples` (paths per ray of `supersample`) and `light_size`
(the angular radius of the key light in radians, 0 for a point).

## On the GPU

With `device="gpu"` the rays are traced on the GPU, with Metal on macOS;
`gpu_available()` tells whether there is one. The picture is the same as on
the processor, with the same random samples in the same precision, up to
rounding at a few edges. A million atoms take 0.34 s for 1920 × 1080 in the
studio style against 1.4 s on the processor, and 1.2 s for 3840 × 2160
against 4.1 s; the photo style gains as much (Apple M5).

```python
from mdapy.visualization import gpu_available

device = "gpu" if gpu_available() else "cpu"
picture = render(grains, color=structures, device=device)
```

## Molecules

Organic molecules are drawn with covalent bonds, found from the elements:

```python
import numpy as np

# benzene: C-C 1.39 Å, C-H 1.09 Å
angles = np.arange(6) * np.pi / 3
ring = np.stack([np.cos(angles), np.sin(angles), np.zeros(6)], axis=1)
benzene = mp.System(
    np.vstack([1.39 * ring, 2.48 * ring]), elements=["C"] * 6 + ["H"] * 6
)
sticks = render(benzene, representation="ball_and_stick", box=False, view="iso")
spheres = render(benzene, representation="space_filling", box=False, view="iso")
```

![Molecules](images/molecules.png)

*Benzene, ethanol and C60 in the representations ball_and_stick, licorice
and space_filling.*

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
`element_colors(elements)` and `element_radii(elements, kind="display")` give the colors
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

With `bounces` above 0, each ray instead follows `samples` paths, the first
through the center of its part of the pixel and the others anywhere in it.
At each surface a path adds the light of the key light, sampled over a disk
`light_size` wide and blocked by anything in the way, then turns into a
cosine-weighted direction about the normal: if it leaves the scene it adds
the sky, of brightness `ambient`, otherwise it goes on to the next surface,
its light now colored by the surfaces it met, for at most `bounces` more
surfaces. Shadows soften and corners darken without ambient occlusion, and
colors bleed onto neighbors; the noise falls as the samples grow.
