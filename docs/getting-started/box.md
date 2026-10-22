# Simulation box

`Box` describes the simulation box: three lattice vectors, the position of
one corner, and whether each direction is periodic. A box never changes;
operations that would change it return a new one.

```python
import numpy as np
from mdapy import Box

box = Box([10.0, 10.0, 12.0])  # orthogonal box
hexagonal = Box.from_lengths_angles(2.5, 2.5, 4.0, gamma=120.0)
slab = Box(np.diag([10.0, 10.0, 30.0]), pbc=[True, True, False])

positions = np.array([[11.0, -1.0, 5.0], [3.0, 4.0, 25.0]])
wrapped, image = box.wrap(positions)
# wrapped: [[1, 9, 5], [3, 4, 1]]   image: [[1, -1, 0], [0, 0, 2]]
box.minimum_image([[9.0, 0.0, 0.0]])  # [[-1, 0, 0]]
```

## Creating a box

| Name | Type | Default | Unit | Meaning |
|---|---|---|---|---|
| `matrix` | float, 3 floats or (3, 3) array | | Å | A number gives a cube; three numbers an orthogonal box; a (3, 3) array a general box whose **rows** are the lattice vectors **a**, **b**, **c**. |
| `origin` | 3 floats | (0, 0, 0) | Å | Position of the corner from which the lattice vectors start. |
| `pbc` | bool or 3 bools | `True` | | Periodicity along **a**, **b** and **c**. |

Other constructors:

| Constructor | Meaning |
|---|---|
| `Box.from_lengths_angles(a, b, c, alpha=90, beta=90, gamma=90, origin, pbc)` | Edge lengths in Å and angles in degrees (alpha between **b** and **c**, beta between **a** and **c**, gamma between **a** and **b**). **a** lies along x and **b** in the xy plane. |
| `Box.from_lammps(lo, hi, tilt=(0, 0, 0), pbc)` | LAMMPS bounds `(xlo, ylo, zlo)`, `(xhi, yhi, zhi)` and tilt factors `(xy, xz, yz)`. |
| `Box.enclosing(positions, padding=1.0)` | Non-periodic orthogonal box around a set of positions, for molecules and clusters. |

Lattice vectors that are not finite or are linearly dependent raise
[`InvalidParameter`](../reference/api.md#mdapy.errors.InvalidParameter).

## Properties

| Name | Shape | Unit | Meaning |
|---|---|---|---|
| `matrix` | (3, 3) | Å | Lattice vectors, one per row. |
| `origin` | (3,) | Å | Corner of the box. |
| `pbc` | (3,) bool | | Periodicity. |
| `inverse` | (3, 3) | 1/Å | Inverse of `matrix`. |
| `volume` | | Å³ | Volume. |
| `lengths` | (3,) | Å | Lengths of **a**, **b**, **c**. |
| `angles` | (3,) | degree | alpha, beta, gamma. |
| `thickness` | (3,) | Å | Distance between each pair of opposite faces. |
| `is_orthogonal` | | | Whether the lattice vectors lie along +x, +y, +z. |

The arrays are read-only. Boxes compare equal with `==` when all values are
identical, and `box.allclose(other)` compares them within a tolerance.

## Coordinates

| Method | Returns |
|---|---|
| `to_fractional(positions)` | Coordinates in units of the lattice vectors, relative to `origin`. |
| `to_cartesian(fractional)` | Positions in Å. |
| `wrap(positions)` | `(wrapped, image)`: positions mapped into the box along periodic directions, and integer image flags. |
| `minimum_image(vectors)` | The shortest periodic image of each displacement vector. |
| `to_lammps()` | The box in LAMMPS restricted triclinic form, with the rotation to apply to positions and other vectors. |

## Conventions

With the lattice vectors as rows of the matrix \(H\), a position
\(\mathbf r\) has fractional coordinates
\(\mathbf s = (\mathbf r - \mathbf r_0) H^{-1}\), where \(\mathbf r_0\) is
the origin.

**Wrapping** gives image flags \(\mathbf n = \lfloor \mathbf s \rfloor\)
along periodic directions, so that the wrapped position has
\(0 \le s_i < 1\) and \(\mathbf r = \mathbf r_\text{wrapped} + \mathbf n H\).
Shifting a position by whole lattice vectors and wrapping it back returns
the original numbers exactly.

**Minimum image** returns the shortest periodic image of each vector for
every box whose tilt factors are within half of the corresponding edge
length, the limit LAMMPS uses.

**Thickness** is the distance between the two faces not crossed by lattice
vector \(i\).

**LAMMPS form.** LAMMPS needs **a** along +x, **b** in the xy plane and
positive diagonal entries: \(H' = \begin{pmatrix} l_x & 0 & 0\\ xy & l_y & 0\\
xz & yz & l_z \end{pmatrix}\). `to_lammps()` returns \(H'\) and the rotation
\(Q\) with \(H' = HQ\); every vector \(\mathbf v\), positions and velocities
included, becomes \(\mathbf v Q\). A box already in this form comes back
unchanged, with \(Q\) the identity. Left-handed lattice vectors, which LAMMPS
rejects, are made right-handed by swapping **b** and **c**, which describes
the same lattice.
