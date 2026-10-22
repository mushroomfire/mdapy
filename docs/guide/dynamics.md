# Displacements and melting

Two analyses of `mdapy.analysis` take a whole trajectory: a list of systems
or a `Trajectory`, with the same atoms in the same order in every frame.

<!-- skip-test -->
```python
import mdapy
from mdapy.analysis import LindemannIndex, MeanSquaredDisplacement

frames = mdapy.read_trajectory("run.dump")
msd = MeanSquaredDisplacement().compute(frames)
msd.lag, msd.msd  # frames, Å²
lindemann = LindemannIndex().compute(frames)
lindemann.index, lindemann.running
```

## Unwrapped positions

Atoms that leave a periodic box usually come back on the other side. Both
analyses follow each atom from frame to frame by the periodic image nearest
to its previous position, in fractional coordinates, so the positions may be
wrapped as long as no atom moves by half the box between two frames.

## Mean squared displacement

`MeanSquaredDisplacement(*, window=True)` gives, for each lag $m$ in frames,

$$
\text{MSD}(m) = \frac{1}{N} \sum_i \frac{1}{T - m} \sum_{k=0}^{T-m-1}
\lvert \mathbf r_i(k + m) - \mathbf r_i(k) \rvert^2 ,
$$

averaged over every time origin $k$ and computed with fast Fourier
transforms; with `window=False` only from the first frame,
$\frac{1}{N}\sum_i |\mathbf r_i(m) - \mathbf r_i(0)|^2$. The
`DisplacementResult` has the `lag`, the `msd` and its x, y and z
`components`. The diffusion coefficient is the slope of the MSD over time
divided by 6.

## Lindemann index

`LindemannIndex()` measures how much the distances between atoms fluctuate:

$$
\delta = \frac{2}{N(N-1)} \sum_{i<j}
\frac{\sqrt{\langle r_{ij}^2 \rangle - \langle r_{ij} \rangle^2}}{\langle r_{ij} \rangle},
$$

with time averages over the frames. It is small in a solid and jumps at
melting, which makes it useful for clusters and nanoparticles. The
`LindemannResult` has the `index`, the `atomic` index of each atom (averaged
over its pairs; column `lindemann`) and the `running` index of the frames up
to each one. Distances are taken between the unwrapped positions, as written
above, which suits clusters in a box that is not periodic. The time grows as
the square of the number of atoms, the memory only as the size of the
trajectory: 10,000 atoms over 50 frames take about 2 s on ten threads.

## Validation

`tests/analysis/test_dynamics.py` checks the windowed MSD of a random walk,
from wrapped positions in a triclinic box, against the sum over every time
origin, and the MSD from the first frame, to 10<sup>−9</sup>; that ballistic
motion gives $v^2 m^2$; that a trajectory read from a file gives the same
result; that the Lindemann index of a vibrating cluster agrees with a direct
computation to 10<sup>−10</sup> and is exactly zero for a rigid motion; and
the errors.
