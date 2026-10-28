# Training sets from VASP

The ionic steps of VASP calculations, read from their OUTCAR files with
their energies, forces and virials, become the extended XYZ training set of
NEP, GPUMD and other machine-learned potentials in a few lines:

<!-- skip-test -->
```python
from pathlib import Path

import numpy as np

import mdapy as mp

frames = [
    frame.with_info(converged=None)
    for outcar in sorted(Path("calculations").glob("*/OUTCAR"))
    for frame in mp.read_trajectory(outcar)
    # the converged steps, without those of huge forces
    if frame.info["converged"] and np.abs(frame["forces"]).max() <= 30.0
]
mp.write("train.xyz", frames)  # energy, virial and forces of each frame
```

## Reading

Each OUTCAR is read as a [trajectory](../getting-started/trajectory.md) of
one frame per ionic step, of a single point, a relaxation or molecular
dynamics, in the format [`outcar`](../reference/formats/outcar.md):

| Where | Name | Meaning |
|---|---|---|
| column | `forces` | (N, 3) eV/Å, `TOTAL-FORCE` |
| `info` | `energy` | eV, the free energy `TOTEN` |
| `info` | `virial` | (3, 3) eV, from `FORCE on cell =-STRESS`; only when VASP computed the stress (ISIF ≥ 1) |
| `info` | `converged` | whether the electronic loop reached EDIFF |

## Writing

[Extended XYZ](../reference/formats/extxyz.md) files keep the `energy`
and `virial` of each frame in its comment line and the `forces` as a
column, as NEP and GPUMD read them. `converged` is set to `None` above so
that the file leaves it out.
