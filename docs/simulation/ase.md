# ASE calculator

`mdapy.interop.to_ase_calculator` makes any mdapy calculator, such as `EAM`
or `NEP`, an ASE calculator, so that the optimizers, filters and molecular
dynamics of ASE can use it.

```python
from ase.build import bulk
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS

from mdapy.calculators import EAM
from mdapy.interop import to_ase_calculator

atoms = bulk("Cu", "fcc", a=3.5, cubic=True)
atoms.calc = to_ase_calculator(EAM.zhou(["Cu"]))
BFGS(FrechetCellFilter(atoms), logfile=None).run(fmax=1e-4)
atoms.cell.lengths()  # [3.615, 3.615, 3.615]
atoms.get_potential_energy() / len(atoms)  # -3.540 eV
```

It needs ASE (`pip install mdapy[ase]`).

## Parameters

`to_ase_calculator(calculator)`

| Name | Type | Default | Unit | Meaning |
|---|---|---|---|---|
| `calculator` | `Calculator` | | | any mdapy calculator |

## Results

An ASE calculator with the properties that the mdapy calculator computes:

| ASE property | Unit | From |
|---|---|---|
| `energy`, `free_energy` | eV | the energy; both the same |
| `energies` | eV | the atomic energies |
| `forces` | eV/Å | the forces |
| `stress` | eV/Å³ | the virial $W$, as $-W/V$ in the order xx, yy, zz, yz, xz, xy, positive under tension |
| `stresses` | eV/Å³ | the atomic virials, as the stress |

Each calculation converts the atoms with [`from_ase`](../getting-started/ase.md)
and searches their neighbors again.

## Errors

ASE raises its `PropertyNotImplementedError` for a property that the mdapy
calculator does not compute.

## See also

- [ASE atoms](../getting-started/ase.md): `to_ase` and `from_ase`.
- [The calculator interface](calculators.md).
- [ASE calculators](https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html).
