# VASP OUTCAR

The format `outcar` reads the OUTCAR files of VASP, one frame per ionic
step of a single point, a relaxation or molecular dynamics, with the
forces, the energy and the virial of each. Files named `OUTCAR*`, or with
extension `.outcar`, are recognized. mdapy reads OUTCAR files but does not
write them.

<!-- skip-test -->
```python
import mdapy as mp

steps = mp.read_trajectory("OUTCAR")  # one frame per ionic step
last = mp.read("OUTCAR", frame=-1)
last["forces"]  # (N, 3) eV/Å
last.info["energy"]  # eV, the free energy TOTEN
last.info["virial"]  # (3, 3) eV
last.info["converged"]  # whether the electronic loop reached EDIFF
```

## Reading

| In the OUTCAR | In mdapy |
|---|---|
| `POTCAR:` lines, each written twice, and `ions per type` | `element`; a type of no ions is allowed |
| `direct lattice vectors`, the last before the positions of a step | `box.matrix`; periodic in all directions |
| `POSITION ... TOTAL-FORCE (eV/Angst)` | `positions` (Å) and `forces` (eV/Å) |
| `Total` of `FORCE on cell =-STRESS` (eV), when VASP computed the stress (ISIF ≥ 1) | `info["virial"]`, (3, 3) |
| `free  energy   TOTEN` | `info["energy"]` (eV), the free energy, of which the forces are the slope |
| `aborting loop because EDIFF is reached` in the step | `info["converged"]` |

Every ionic step is read, whether its electronic loop converged or not,
so that a single point that did not converge still gives its atoms;
`info["converged"]` tells which steps reached EDIFF, and a training set
keeps those (see [Training sets from VASP](../../mlip/vasp.md)).

A step ends at its line `FREE ENERGIE OF THE ION-ELECTRON SYSTEM`; the
lines of the machine-learned force field of VASP 6, which start with `ML`,
are not steps. A step cut short at the end of the file is left out.
Numbers that VASP writes without a space between them, as
`-10293.69148-10013.02274`, are read apart.

The stress in GPa is `-virial / volume` (see
[Units and conventions](../units.md)), the same as that of ASE, which takes
it from the line in kB. ASE's `get_potential_energy()` gives the energy
extrapolated to zero smearing, and `get_potential_energy(force_consistent=True)`
the free energy TOTEN that mdapy reads.

## Errors

[`FileFormatError`](../api.md#mdapy.errors.FileFormatError) at the line of
the fault for: no line of ions per type before the first step, one that
cannot be read, POTCAR lines that do not match the ions per type, an
unknown element, a step without lattice vectors, positions and forces or
free energy, and a line with fewer numbers than it should have.

## See also

- [Training sets from VASP](../../mlip/vasp.md)
- [VASP](../../interop/vasp.md) and the [VASP calculator](../../simulation/vasp.md)
