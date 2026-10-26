# GPUMD

mdapy writes the `model.xyz` that GPUMD starts from, reads the extended XYZ
files GPUMD writes, computes with the NEP potentials GPUMD trains, and builds
their training sets.

```python
import numpy as np

import mdapy as mp
from mdapy.build import build_crystal

crystal = build_crystal("fcc", 3.615, elements="Cu", repeat=(10, 10, 10))
rng = np.random.default_rng(0)
upper = crystal["positions"][:, 2] > crystal.box.matrix[2, 2] / 2
crystal = crystal.with_columns(
    velocities=rng.normal(0.0, 2.0, (len(crystal), 3)),  # Å/ps
    group=upper.astype(np.int64),  # one grouping method: lower 0, upper 1
)
mp.write("model.xyz", crystal)
# Properties=species:S:1:pos:R:3:vel:R:3:group:I:1, velocities in Å/fs
```

Then, after the run:

<!-- skip-test -->
```python
from mdapy.calculators import NEP

with mp.read_trajectory("dump.xyz") as frames:  # from dump_exyz
    last = frames[-1]
potential = NEP("nep.txt")
potential.compute(last, ("energy", "forces", "virial"))
```

## What goes between them

| GPUMD | mdapy | Page |
|---|---|---|
| `model.xyz`: `species`, `pos`, `mass`, `charge`, `vel` (Å/fs), `group` | `mdapy.write` of extended XYZ: `element`, `positions`, `mass`, `charge`, `velocities` (Å/ps), `group` | [Extended XYZ](../reference/formats/extxyz.md) |
| `dump_exyz`, `train.xyz`, `test.xyz` | `mdapy.read_trajectory` | [Extended XYZ](../reference/formats/extxyz.md) |
| NEP potentials, `nep.txt` | the calculator `NEP` | [NEP](../simulation/nep.md) |
| training sets | `outcar_to_xyz`, choosing frames, perturbed structures | [Training sets from VASP](../mlip/vasp.md) |

GPUMD needs at least two atoms and a `Lattice`, which mdapy always writes.
A `group` column of integers, or of k integers per atom for k grouping
methods, is written as `group:I:k`. GPUMD has no origin of the box and
ignores the `Origin` that mdapy writes for a box with one.

## See also

- [GPUMD documentation](https://gpumd.org/).
