# ASE atoms

`mdapy.interop.to_ase` and `from_ase` convert a `System` to an `ase.Atoms`
and back, with velocities, charges, constraints and the results of a
calculation, so that a structure can move between mdapy and ASE.

```python
from ase.build import bulk
from ase.constraints import FixAtoms

from mdapy.interop import from_ase, to_ase

atoms = bulk("Cu", "fcc", a=3.615, cubic=True).repeat(2)
atoms.set_constraint(FixAtoms(indices=[0, 1]))
system = from_ase(atoms)  # 32 atoms in a 7.23 Å box
system.table["movable"]  # False for the first two atoms
back = to_ase(system)
back.constraints  # [FixAtoms(indices=[0, 1])]
```

Both need ASE (`pip install mdapy[ase]`).

## Parameters

| Function | Argument | Returns |
|---|---|---|
| `to_ase(system)` | a `System` | a new `ase.Atoms` |
| `from_ase(atoms)` | an `ase.Atoms` | a new `System` |

The result shares no memory with its source.

## Results

| System | ase.Atoms |
|---|---|
| `positions` | positions |
| `box.matrix`, `box.pbc` | cell, pbc |
| `box.origin` | cell displacement (`celldisp`) |
| `element` | chemical symbols; `X` without an element column |
| `mass` | masses, when set on the atoms |
| `velocities` (Å/ps) | momenta, in ASE's unit of time, Å·(amu/eV)<sup>1/2</sup> |
| `charge` | initial charges |
| `movable` | `FixAtoms` for atoms that cannot move, `FixCartesian` for atoms fixed along some axes |
| other numeric columns | arrays of the same name |
| `info` | `info` |
| `info["energy"]`, `forces`, `info["stress"]` (GPa) or `info["virial"]` (eV) | a single-point calculator with the energy, forces and stress (eV/Å³) |

Columns of strings, such as `type_label`, are not converted. ASE gives the
stress in eV/Å³ in the order xx, yy, zz, yz, xz, xy; mdapy keeps it as a
(3, 3) array in GPa. Both are positive under tension.

## Errors

`from_ase` raises [`InvalidParameter`](../reference/api.md#mdapy.errors.InvalidParameter)
for constraints other than `FixAtoms` and `FixCartesian`.

## See also

- [ASE calculator](../simulation/ase.md): an mdapy calculator in ASE.
- The [ase.Atoms](https://wiki.fysik.dtu.dk/ase/ase/atoms.html) object.
