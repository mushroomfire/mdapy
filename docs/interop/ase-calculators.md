# ASE calculators

`mdapy.interop.to_ase_calculator` makes any mdapy calculator, such as `EAM`
or `NEP`, an ASE calculator, for the optimizers, filters and molecular
dynamics of ASE; `from_ase_calculator` makes any ASE calculator, such as the
calculator of a machine-learned potential, an mdapy calculator, for
[`minimize`](../simulation/minimize.md),
[defect energies](../simulation/defect-energies.md) and
[elastic constants](../simulation/elastic-constants.md).

```python
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS

from mdapy.build import build_crystal
from mdapy.calculators import EAM
from mdapy.interop import from_ase_calculator, to_ase_calculator
from mdapy.simulation import minimize

# an mdapy potential in ASE
atoms = bulk("Cu", "fcc", a=3.5, cubic=True)
atoms.calc = to_ase_calculator(EAM.zhou(["Cu"]))
BFGS(FrechetCellFilter(atoms), logfile=None).run(fmax=1e-4)
atoms.cell.lengths()  # [3.615, 3.615, 3.615]

# an ASE calculator in mdapy
copper = build_crystal("fcc", 3.5, elements="Cu", repeat=(2, 2, 2))
relaxed = minimize(copper, from_ase_calculator(EMT()), cell="isotropic")
relaxed.system.box.lengths / 2  # [3.590, 3.590, 3.590], the lattice of EMT
```

Any ASE calculator serves, for example the NEP calculator of torchnep:

<!-- skip-test -->
```python
from torchnep.ase_calculator import NEP

potential = from_ase_calculator(NEP("nep.txt"))
potential.compute(system, ("energy", "forces", "virial"))
```

Both need ASE (`pip install mdapy[ase]`).

## Parameters

| Function | Argument | Returns |
|---|---|---|
| `to_ase_calculator(calculator)` | any mdapy `Calculator` | an ASE calculator |
| `from_ase_calculator(calculator)` | any ASE calculator | an mdapy `Calculator` |

## Results

The properties of each side and those of the other they come from:

| ASE property | Unit | mdapy property | Unit |
|---|---|---|---|
| `energy`, `free_energy` | eV | `energy` | eV |
| `energies` | eV | `atomic_energies` | eV |
| `forces` | eV/Å | `forces` | eV/Å |
| `stress` | eV/Å³ | `virial`, $W = -\sigma V$ | eV |
| `stresses` | eV/Å³ | `atomic_virials`, $-\sigma_i V$ | eV |

ASE gives stresses in the order xx, yy, zz, yz, xz, xy, positive under
tension. Each side offers the properties that the other computes. An mdapy
calculator in ASE gives its energy as both `energy` and `free_energy`; an
ASE calculator in mdapy gives `free_energy` as the energy where it computes
one, since the forces derive from it.

Each calculation converts the atoms with [`from_ase`](ase.md) or
[`to_ase`](ase.md). An ASE calculator in mdapy has a `cutoff` of 0: it
searches its own neighbors and takes no neighbor list of mdapy.

## Errors

ASE raises its `PropertyNotImplementedError` for a property that the mdapy
calculator does not compute; an ASE calculator in mdapy raises
[`InvalidParameter`](../reference/api.md#mdapy.errors.InvalidParameter) for
one that the ASE calculator does not compute.

## See also

- [The calculator interface](../simulation/calculators.md).
- [ASE calculators](https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html).
