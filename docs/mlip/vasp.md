# Training sets from VASP

`read_outcar` reads the ionic steps of a VASP OUTCAR, of a single point, a
relaxation or molecular dynamics, with their energies, forces and virials.
`outcar_to_xyz` writes the steps of many OUTCARs to one extended XYZ file,
the training set of NEP, GPUMD and other machine-learned potentials.

<!-- skip-test -->
```python
from pathlib import Path

from mdapy.mlip import outcar_to_xyz, read_outcar

steps = read_outcar("relax/OUTCAR")  # one System per ionic step
last = steps[-1]
last.info["energy"]  # eV, the free energy TOTEN
last.info["virial"]  # (3, 3) eV
last.table["forces"]  # eV/Å

# every converged step of every calculation, without the steps of huge forces
outcars = sorted(Path("calculations").glob("*/OUTCAR"))
outcar_to_xyz(outcars, "train.xyz", max_force=30.0)  # number of frames written
```

## Parameters

| Function | Parameters |
|---|---|
| `read_outcar(path, *, unconverged=False)` | the OUTCAR, and whether to keep the steps whose electronic loop stopped before EDIFF |
| `outcar_to_xyz(outcars, path, *, unconverged=False, max_force=None)` | one OUTCAR or a sequence of them, the extended XYZ file written, the steps kept, and the largest force component (eV/Å) of a frame kept |

The elements come from the POTCAR lines and the `ions per type`; a type of
no ions is allowed. A single point is a file of one step; a relaxation or
molecular dynamics gives one frame per ionic step.

## Results

`read_outcar` returns one `System` per ionic step, with

| Where | Name | Meaning |
|---|---|---|
| column | `forces` | (N, 3) eV/Å, `TOTAL-FORCE` |
| `info` | `energy` | eV, the free energy `TOTEN` |
| `info` | `virial` | (3, 3) eV, from `FORCE on cell =-STRESS`; only when VASP computed the stress (ISIF ≥ 1) |
| `info` | `converged` | whether the electronic loop reached EDIFF |

The stress in GPa is `-virial / volume` (see
[Units and conventions](../reference/units.md)), the same as that of ASE.
`outcar_to_xyz` leaves out `converged` and returns the number of frames
written.

## Errors

`read_outcar` raises `FileFormatError` for forces before the number of ions
and the lattice, a cut line of positions and forces, ions per type that do
not add up to the ions, and ions per type without POTCAR elements.
`outcar_to_xyz` raises `InvalidParameter` for a `max_force` that is not
positive.
