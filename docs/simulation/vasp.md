# VASP

<!-- skip-test -->
```python
from mdapy.calculators import VASP
from mdapy.simulation import minimize

vasp = VASP(
    {"ENCUT": 520, "EDIFF": 1e-6, "ISMEAR": 1, "SIGMA": 0.1, "NCORE": 8},
    kpoints=(4, 4, 4),
    setups={"Ni": "Ni_pv"},
    command="srun vasp_std",  # as the job script of the cluster runs VASP
    directory="relax",
)
result = vasp.compute(system, ("energy", "forces", "virial"))
relaxed = minimize(system, vasp, max_force=0.01).system
```

`VASP(incar=None, *, kpoints=None, potcars=None, setups=None, command=None,
directory="vasp", restart=True)` makes VASP a calculator: each call writes
POSCAR, INCAR, KPOINTS and POTCAR in the directory, runs VASP there with the
command and reads the energy, forces and stress of its OUTCAR. A Python
script on a cluster, in the job script, so drives VASP through
[`minimize`](minimize.md), [elastic constants](elastic-constants.md) or
[defect energies](defect-energies.md) as any other calculator.

| Name | Type | Default | Unit | Meaning |
|---|---|---|---|---|
| `incar` | mapping | `None` | | INCAR tags; NSW, IBRION and ISIF are set for a single point with the stress and may not be given |
| `kpoints` | three int | `None` | | a Gamma-centered mesh of k-points; without it VASP takes `KSPACING` from the INCAR |
| `potcars` | path | `$VASP_PP_PATH` or its `potpaw_PBE` | | the folder of the POTCARs, one folder per element or setup |
| `setups` | mapping | `None` | | the POTCAR of an element when not its symbol, such as `{"Ti": "Ti_sv"}` |
| `command` | str | `$MDAPY_VASP_COMMAND`, `$ASE_VASP_COMMAND` or `"vasp_std"` | | the command that runs VASP in the directory, such as `"mpirun -np 128 vasp_std"` |
| `directory` | path | `"vasp"` | | where VASP runs, made when missing |
| `restart` | bool | `True` | | keep WAVECAR and CHGCAR between calls, so that each run starts from the wave functions of the last |

| Attribute or method | Meaning |
|---|---|
| `cutoff` | 0: VASP needs no neighbor list |
| `compute(system, properties, *, neighbors=None)` | the free energy (TOTEN) in eV, forces in eV/Å and virial in eV; `neighbors` is not used |

The system needs an `element` column. VASP holds the atoms of each element
together: the POSCAR lists the elements in the order they first appear,
the POTCAR follows it, and the forces come back in the order of the atoms
of the system. Python booleans become `.TRUE.` and `.FALSE.` in the INCAR,
and sequences numbers separated by spaces, as for `MAGMOM`.

The constructor raises `InvalidParameter` for NSW, IBRION or ISIF in the
INCAR and k-points that are not three positive integers; `compute` raises
`InvalidParameter` without POTCARs or for an element without its POTCAR,
`MissingProperty` without an `element` column, and
[`CalculationFailed`](../reference/api.md#mdapy.errors.CalculationFailed)
when VASP stops with an error or leaves no converged electronic step, with
the end of what VASP wrote.

!!! quote "Please cite"
    [Kresse and Furthmüller (1996)](../reference/references.md#kresse1996).
