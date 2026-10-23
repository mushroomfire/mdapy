<!-- needs: pymatgen.core -->
# pymatgen

`mdapy.interop.to_pymatgen` and `from_pymatgen` convert a `System` to a
pymatgen `Structure` and back, for the symmetry analysis, structure
matching and materials databases of pymatgen.

```python
from pymatgen.core import Lattice, Structure

from mdapy.interop import from_pymatgen, to_pymatgen

nacl = Structure.from_spacegroup(
    "Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
)
system = from_pymatgen(nacl)  # 8 atoms
structure = to_pymatgen(system)
structure.get_space_group_info()  # ('Fm-3m', 225)
```

Both need pymatgen (`pip install pymatgen`, or `pymatgen-core` for the
structures alone).

## Parameters

| Function | Argument | Returns |
|---|---|---|
| `to_pymatgen(system)` | a `System` with an `element` column | a new `Structure` |
| `from_pymatgen(structure)` | a `Structure` or a `Molecule` | a new `System` |

## Results

| System | pymatgen |
|---|---|
| `box.matrix`, `box.pbc` | lattice and its periodicity |
| `positions` | Cartesian coordinates, from the origin of the box |
| `element` | species |
| `velocities` (Å/ps) | site property `velocities` (Å/fs, as in POSCAR files) |
| `movable` | site property `selective_dynamics` |
| other numeric columns | site properties of the same name |
| `info` | properties |

pymatgen has no origin of the lattice: `to_pymatgen` moves the atoms by
minus the origin, and `from_pymatgen` gives a box at the origin. A
`Molecule` gets a box that encloses it, not periodic. Columns of strings
and the topology are not converted.

## Errors

`to_pymatgen` raises
[`MissingProperty`](../reference/api.md#mdapy.errors.MissingProperty)
without an `element` column; `from_pymatgen` raises
[`InvalidParameter`](../reference/api.md#mdapy.errors.InvalidParameter)
for a disordered site, with more than one element.

## See also

- pymatgen [Structure](https://pymatgen.org/pymatgen.core.html#pymatgen.core.structure.Structure).
