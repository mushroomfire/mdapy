# Voronoi cells

`mdapy.analysis.VoronoiAnalysis` divides space into the Voronoi cells of the
atoms, the regions closer to one atom than to any other, periodic images
included. mdapy computes them with its own compiled code.

<!-- skip-test -->
```python
from mdapy.analysis import VoronoiAnalysis

cells = VoronoiAnalysis().compute(system)
cells.volume  # (N,) Å³
cells.coordination  # (N,) faces shared with other atoms
cells.index()  # (N, 6) faces with 3, 4, ..., 8 vertices
# columns voronoi_volume, voronoi_coordination and cavity_radius
system = system.with_columns(cells)
```

Each cell starts as a cube around its atom and is cut by the plane halfway to
each neighbor, roughly nearest first (in narrow distance bins), until the
next neighbors are more than twice as far as the farthest vertex of the cell;
no farther atom can cut it then. The neighbors are searched on a grid of
cells; the cells they do not close are computed again with a longer reach. A
million atoms take about 6 s on one thread and 1.0 s on ten. In
directions that are not periodic, the cells are closed by the faces of the
box, moved out to include every atom. Two atoms at the same position raise `ValueError`.

The `VoronoiResult` has, per atom, the `volume`, the `coordination` (faces
shared with other atoms) and the `radius`, the distance to the farthest vertex
of the cell. The faces are in CSR form: the faces of atom `i` are
`offsets[i]:offsets[i + 1]`, with the atom across each face in `neighbors`
(−1 for a wall of the box) and its periodic image in `shifts`, the `areas` and
the number of `vertices`. `index(largest=8)` counts the faces with 3 to
`largest` vertices of each cell, the Voronoi index.

`VoronoiAnalysis(area_threshold=0.01)` leaves out faces smaller than that
fraction of their cell's surface from the coordination and the faces. In a
crystal with thermal displacements, the four-fold vertices of the fcc cell
split into small faces between second neighbors, which such a threshold
removes; the volume is unchanged.

| Crystal | Cell | Faces | Volume |
|---|---|---|---|
| fcc | rhombic dodecahedron | 12 rhombi | $a^3/4$ |
| bcc | truncated octahedron | 6 squares, 8 hexagons | $a^3/2$ |
| hcp | trapezo-rhombic dodecahedron | 12 quadrilaterals | $\sqrt 2\, a^3 / 2$ |
| cubic diamond | | 12 triangles, 4 hexagons | $a^3/8$ |

## Validation

The tests in `tests/analysis/test_voronoi.py` check that:

- the cells of fcc, bcc, hcp and diamond have the volumes, face areas and
  Voronoi indices of the table, to 10<sup>−12</sup>;
- the cells of random atoms fill a triclinic box, periodic, partly periodic or
  closed, to 10<sup>−12</sup>; the wall faces add up to the faces of the box;
  every face appears in the cells of both its atoms, with the same area;
- the volume of each cell agrees, within five standard deviations, with the
  fraction of 40 000 random points closest to its atom;
- results do not change under rotation and translation, are the same for any
  number of threads, and a one-atom box is a single cell with six faces.

## References

- G. Voronoi, Nouvelles applications des paramètres continus à la théorie des
  formes quadratiques, *J. Reine Angew. Math.* **134**, 198 (1908).
- C. H. Rycroft, VORO++: A three-dimensional Voronoi cell library in C++,
  *Chaos* **19**, 041111 (2009). doi:10.1063/1.3215722
