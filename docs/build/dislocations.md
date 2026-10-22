# Dislocations

`insert_dislocation` displaces the atoms of a system by the elastic field of
a straight dislocation; `edge_dislocation` joins two crystals with one
period more below a line than above it.

```python
import numpy as np

from mdapy.build import build_crystal, edge_dislocation, insert_dislocation

# a screw dislocation along [111] in iron
iron = build_crystal(
    "bcc",
    2.87,
    elements="Fe",
    orientation=[(1, 1, -2), (-1, 1, 0), (1, 1, 1)],
    repeat=(10, 16, 2),
)
center = iron.box.matrix.diagonal() / 2 + [0.1, 0.1, 0.0]
screw = insert_dislocation(iron, [0, 0, 2.87 * np.sqrt(3) / 2], line="z", center=center)

# an edge dislocation in copper: x along b = 1/2 [1-10], y along the line,
# z normal to the (111) glide plane
cell = build_crystal(
    "fcc", 3.615, elements="Cu", orientation=[(1, -1, 0), (1, 1, -2), (1, 1, 1)]
)
edge = edge_dislocation(cell, (30, 2, 12))
```

## Parameters

`insert_dislocation(system, burgers, *, line=2, normal=1, center=None, poisson=1/3)`

| Name | Type | Default | Unit | Meaning |
|---|---|---|---|---|
| `burgers` | (3,) float | | Å | the Burgers vector, in the plane of cut: its part along `line` is screw, the rest edge |
| `line` | int or str | `"z"` | | the Cartesian axis along the dislocation line |
| `normal` | int or str | `"y"` | | the Cartesian axis normal to the plane of cut, other than `line` |
| `center` | (3,) float | center of the box | Å | a point of the line |
| `poisson` | float | 1/3 | | Poisson's ratio, for the edge part |

`edge_dislocation(cell, repeat)` takes an orthogonal periodic `cell` whose
period along x is the Burgers vector, and `repeat` = (nx, ny, nz): nx + 1
periods along x below the line and nx above, ny along y, nz along z, at
least 2.

## Results

`insert_dislocation` returns the same atoms, displaced, in the same box. The
displacements do not fit a box periodic across the line: the plane of cut
reaches one face of the box, where the crystal is no longer continuous.

`edge_dislocation` returns the lower crystal compressed and the upper one
stretched along x to nx + 1/2 periods, in a box periodic along x and y and
not along z; the line lies between them at the middle of x. Relaxed with a
potential, the dislocation takes its core; in fcc metals it splits into two
Shockley partials around a ribbon of stacking fault.

## Errors

`insert_dislocation` raises `InvalidParameter` for the same axis as line and
normal, a Burgers vector out of the plane of cut or not of three
components, a Poisson's ratio out of (−1, 0.5), an unknown axis, or an atom
on the line; `edge_dislocation` for a cell that is not orthogonal and
periodic, or a `repeat` too small.

## Definition

With $x_1$ along the edge part $b_e$ of the Burgers vector, $x_2$ along
`normal` and $x_3$ along `line`, measured from `center`, the screw part
$b_s$ displaces the atoms along the line and the edge part across it, as in
isotropic elasticity:

$$
u_3 = \frac{b_s}{2\pi} \operatorname{atan2}(x_2, x_1),
$$

$$
u_1 = \frac{b_e}{2\pi}\left[\arctan\frac{x_2}{x_1} + \frac{x_1 x_2}{2(1-\nu)(x_1^2 + x_2^2)}\right],
\qquad
u_2 = -\frac{b_e}{2\pi}\left[\frac{1-2\nu}{4(1-\nu)} \ln(x_1^2 + x_2^2) + \frac{x_1^2 - x_2^2}{4(1-\nu)(x_1^2+x_2^2)}\right],
$$

with $\nu$ Poisson's ratio. The cut of the screw part is the half plane
$x_1 < 0$; that of the edge part the half plane $x_1 > 0$, where the atoms
below the plane first move by $|b_e|/2$ along $x_1$ and those above it move
back by $|b_e|/2$ after the displacement, as atomsk does with `-dislocation
... edge`.

!!! quote "Please cite"
    [Hirth and Lothe (1982)](../reference/references.md#hirth1982).

## See also

- atomsk [`-dislocation`](https://atomsk.univ-lille.fr/doc/en/option_disloc.html),
  whose isotropic screw and edge dislocations `insert_dislocation` gives to
  rounding.
