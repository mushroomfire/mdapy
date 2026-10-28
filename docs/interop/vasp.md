# VASP

mdapy writes the POSCAR of a calculation and reads its CONTCAR and OUTCAR,
runs VASP as a calculator and turns its results into training sets of
machine-learned potentials.

```python
import numpy as np

import mdapy as mp
from mdapy.build import add_vacuum, build_crystal

slab = add_vacuum(build_crystal("fcc", 3.52, elements="Ni", repeat=(2, 2, 4)), 10.0)
bottom = slab["positions"][:, 2] < 2.0
movable = np.repeat(~bottom[:, None], 3, axis=1)  # the bottom layers fixed
mp.write("POSCAR", slab.with_columns(movable=movable))  # selective dynamics
mp.read("POSCAR").table["movable"]  # back as written
```

After the run:

<!-- skip-test -->
```python
relaxed = mp.read("CONTCAR")
steps = mp.read_trajectory("OUTCAR")  # every ionic step, with energies, forces, virials
```

## What goes between them

| VASP | mdapy | Page |
|---|---|---|
| POSCAR, CONTCAR: selective dynamics, velocities | `mdapy.read`, `mdapy.write`; the columns `movable` and `velocities` | [VASP POSCAR](../reference/formats/poscar.md) |
| OUTCAR: ionic steps, free energies TOTEN, forces, virials, convergence | `mdapy.read`, `mdapy.read_trajectory` | [VASP OUTCAR](../reference/formats/outcar.md), [Training sets from VASP](../mlip/vasp.md) |
| single points run by a command, as in a job script | the calculator `VASP` | [VASP calculator](../simulation/vasp.md) |

## See also

- [VASP wiki](https://www.vasp.at/wiki/).
