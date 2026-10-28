# Trajectory lines

`trajectory_lines` gives the path of each atom over the frames of a
trajectory, followed through the periodic boundaries, as OVITO's generate
trajectory lines does; `render(lines=...)` draws the paths.

```python
import numpy as np

import mdapy as mp
from mdapy.analysis import trajectory_lines
from mdapy.visualization import render

# twelve atoms that wander through a periodic box for 300 frames
rng = np.random.default_rng(1)
box = mp.Box(20.0)
walk = rng.random((12, 3)) * 20 + np.cumsum(rng.normal(0, 0.3, (300, 12, 3)), axis=0)
frames = [
    mp.System(box.wrap(x)[0], box, elements="Ar").with_columns(id=np.arange(1, 13))
    for x in walk
]

paths = trajectory_lines(frames)  # (12, 300, 3), unwrapped
some = trajectory_lines(frames, [0, 3, 5])  # (3, 300, 3)
render(
    frames[-1],
    "images/trajectory-lines.png",
    radius=0.8,
    lines=paths,
    line_radius=0.12,
    line_color="#3b6fb6",
    wrap_lines=True,
    view=(-1.0, -0.7, -0.5),
    width=600,
    height=600,
)
```

![Paths of wandering atoms](images/trajectory-lines.png)

*The paths of twelve atoms over 300 frames, cut at the faces of the
periodic box and moved in through the opposite faces, and the atoms where
they end.*

## Parameters

`trajectory_lines(frames, atoms=None, *, unwrap=True)`

| Name | Type | Default | Meaning |
|---|---|---|---|
| `frames` | iterable of `System` | | the frames, such as a [`Trajectory`](../getting-started/trajectory.md) or a slice of one, read one at a time |
| `atoms` | mask or indices | all | the atoms of the first frame whose paths are wanted |
| `unwrap` | bool | `True` | follow the atoms through the periodic boundaries, or take the positions as they are |

The atoms are matched by their ids and unwrapped as for the
[mean squared displacement](msd.md#parameters): by the image flags, or else
through the nearest periodic image of each step. OVITO follows the nearest
image only, so that its lines of atoms that move more than half the box
between frames differ.

`render` draws polylines with `lines`, `line_radius` (Å, 0.1 as OVITO's
line width of 0.2), `line_color` and `wrap_lines`, which cuts the lines at
the periodic faces of the box and moves the parts outside in through the
opposite faces; see [Rendering](../visualization/rendering.md#parameters).

## Results

A (M, T, 3) float64 array: the position of each atom chosen in each frame,
the atoms in the order of the first frame.

## Errors

`trajectory_lines` raises `InvalidParameter` as the
[mean squared displacement](msd.md#errors) does, and for atoms that are not
a mask or indices of the first frame.
