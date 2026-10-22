# Mean squared displacement

`MeanSquaredDisplacement` gives the mean squared displacement of the atoms
of a trajectory against the lag between frames, from which the diffusion
coefficient follows.

```python
import numpy as np

import mdapy
from mdapy.analysis import MeanSquaredDisplacement

# 100 frames of 50 atoms on a random walk in a 10 Å box, positions wrapped
rng = np.random.default_rng(0)
box = mdapy.Box(10.0)
walk = np.cumsum(rng.normal(scale=0.1, size=(100, 50, 3)), axis=0)
frames = [mdapy.System(box.wrap(x)[0], box, elements="Ar") for x in walk]

result = MeanSquaredDisplacement().compute(frames)  # or a Trajectory
result.lag, result.msd  # lag in frames, MSD in Å²
dt = 0.1  # ps between frames
slope = np.polyfit(result.lag[1:50] * dt, result.msd[1:50], 1)[0]
diffusion = slope / 6  # Å²/ps, about 0.05
```

## Parameters

| Name | Type | Default | Unit | Meaning |
|---|---|---|---|---|
| `window` | bool | `True` | | average over every time origin, or measure from the first frame only |

`compute(frames)` takes a sequence of systems, such as a list or a
[`Trajectory`](../getting-started/trajectory.md), with the same atoms in the
same order in every frame. Each atom is followed from frame to frame through
the periodic boundaries:

- by its image flags, the `image` column, as in a LAMMPS dump with
  `ix iy iz`, where two consecutive frames both have them;
- otherwise by the periodic image nearest to its position in the previous
  frame, which needs every atom to move less than half the box between two
  frames.

The positions may be wrapped into the box or not, and the box may change
from frame to frame.

## Results

A `DisplacementResult`:

| Attribute | Type | Meaning |
|---|---|---|
| `lag` | (T,) int64 | the lag $m$ in frames, from 0 to T − 1; the lag time is $m$ times the time between frames |
| `msd` | (T,) float64 | the mean squared displacement in Å² for each lag |
| `components` | (T, 3) float64 | its x, y and z parts in Å² |

## Errors

`compute` raises `InvalidParameter` without frames, for frames with
different numbers of atoms and for frames without atoms.

## Definition

With the positions $\mathbf r_i(k)$ of the $N$ atoms in frame $k$ of $T$,
followed through the periodic boundaries as above, the mean squared
displacement averaged over every time origin (`window=True`) is

$$
\text{MSD}(m) = \frac{1}{N} \sum_i \frac{1}{T - m} \sum_{k=0}^{T-m-1}
\lvert \mathbf r_i(k + m) - \mathbf r_i(k) \rvert^2 ,
$$

computed with fast Fourier transforms as described by
[Calandrini et al. (2011)](../reference/references.md#calandrini2011); from
the first frame only (`window=False`) it is

$$
\text{MSD}(m) = \frac{1}{N} \sum_i \lvert \mathbf r_i(m) - \mathbf r_i(0) \rvert^2 .
$$

The average over every origin is smoother, but the largest lags have few
origins: $T - m$, only one at $m = T - 1$. For atoms that diffuse with the
coefficient $D$, $\text{MSD} \to 6 D t$ at long times $t$, and each
component tends to $2 D t$.

!!! quote "Please cite"
    With `window=True`, [Calandrini et al. (2011)](../reference/references.md#calandrini2011).

## See also

- LAMMPS [`compute msd`](https://docs.lammps.org/compute_msd.html), the same
  as `window=False`.
- OVITO [Calculate displacements](https://docs.ovito.org/reference/pipelines/modifiers/displacement_vectors.html)
  after [Unwrap trajectories](https://docs.ovito.org/reference/pipelines/modifiers/unwrap_trajectories.html).
