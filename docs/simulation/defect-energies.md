# Defect energies

`vacancy_formation_energy`, `surface_energy` and `stacking_fault_energy`
relax the configurations of the [structure builder](../build/defects.md)
with a calculator and give the energies of their defects.

```python
import numpy as np

from mdapy.build import build_crystal
from mdapy.calculators import EAM
from mdapy.simulation import (
    stacking_fault_energy,
    surface_energy,
    vacancy_formation_energy,
)

copper = EAM.zhou(["Cu"])
a = 3.615
crystal = build_crystal("fcc", a, elements="Cu", repeat=(4, 4, 4))
vacancy_formation_energy(crystal, copper).energy  # eV, about 1.28

layers = build_crystal(
    "fcc",
    a,
    elements="Cu",
    orientation=[(1, 1, -2), (-1, 1, 0), (1, 1, 1)],
    repeat=(2, 3, 4),
)
surface_energy(layers, copper).energy  # J/m², the (111) surface

# the generalized stacking fault energy along [11-2] on (111)
curve = stacking_fault_energy(layers, copper, [a * np.sqrt(6) / 2, 0, 0], points=13)
curve.energy[4]  # J/m², the intrinsic stacking fault at 1/6 [11-2]
```

## Parameters

| Function | Parameters |
|---|---|
| `vacancy_formation_energy(crystal, calculator, *, atom=0, cell=None, max_force=1e-4, max_steps=10000)` | the perfect `crystal`, the `atom` removed, and how the box relaxes (`cell` of [minimize](minimize.md)) |
| `surface_energy(crystal, calculator, *, axis=2, vacuum=15.0, max_force=1e-4, max_steps=10000)` | the periodic `crystal`, the box vector `axis` across the slab, and the `vacuum` in Å added along it |
| `stacking_fault_energy(crystal, calculator, direction, *, points=21, height=None, axis=2, relax=True, max_force=1e-4, max_steps=10000)` | the `crystal` with the plane of the fault normal to the Cartesian `axis`, the slip vector `direction` in Å, the number of `points` from 0 to the whole vector, the fractional `height` of the fault (default 0.5) and whether the atoms relax along the normal |

`max_force` (eV/Å) and `max_steps` are those of each
[minimization](minimize.md), with FIRE; a minimization that does not
converge raises an error. The crystal must be large enough that a defect
does not meet its periodic images, and the slab thick enough that its
surfaces do not feel each other. With a crystal periodic along the normal of
the fault, the box moves with the upper half, so that the box holds a single
fault (see [`slip`](../build/defects.md)); without it, the fault lies
between two free surfaces, whose energy cancels in the difference.

## Results

| Result | Attributes |
|---|---|
| `VacancyResult` | `energy` (eV), `perfect` and `defect` energies (eV), the relaxed `system` |
| `SurfaceResult` | `energy` (J/m²), `bulk` and `slab` energies (eV), the `area` of one surface (Å²), the relaxed `system` |
| `StackingFaultResult` | `fraction` of the slip vector, `displacement` (P, 3) in Å, `energy` (P,) in J/m² relative to no slip, the `area` of the fault (Å²) |

## Errors

They raise `InvalidParameter` for a crystal of fewer than two atoms, a
crystal that is not periodic for a surface, a vacuum that is not positive,
fewer than two points, a plane of fault not normal to the axis, an unknown
axis, and a minimization that does not converge.

## Definition

With $E_N$ the energy of the perfect crystal of $N$ atoms and $E_{N-1}$ that
of the crystal with a vacancy, both relaxed,

$$
E_v = E_{N-1} - \frac{N-1}{N} E_N .
$$

With $E_\text{slab}$ the energy of the crystal opened by the vacuum, relaxed,
and $A$ the area of each of its two surfaces,

$$
\gamma_s = \frac{E_\text{slab} - E_\text{bulk}}{2A} .
$$

The generalized stacking fault energy at the slip $f\,\mathbf d$ is

$$
\gamma(f) = \frac{E(f\,\mathbf d) - E(0)}{A},
$$

each configuration relaxed along the normal of the fault only: a `movable`
column keeps the atoms still along the plane, as LAMMPS `fix setforce 0 0
NULL` does. In fcc along $[11\bar2]$ the curve has the unstable stacking
fault energy as its first maximum and the intrinsic stacking fault energy
at $\tfrac16[11\bar2]$.
