# Extended XYZ

The format `extxyz` (extensions `.xyz` and `.extxyz`) stores one or more
frames, each with per-atom columns, a lattice and global values such as the
energy. mdapy reads and writes the conventions of ASE and GPUMD.

```python
from pathlib import Path

import numpy as np
import mdapy

system = mdapy.System(
    positions=[[0.0, 0.0, 0.0], [1.8, 1.8, 0.0]],
    box=mdapy.Box(3.6),
    elements=["Cu", "Cu"],
    info={"energy": -7.0, "config_type": "fcc"},
).with_columns(forces=np.zeros((2, 3)))
mdapy.write("cu.xyz", system)
print(Path("cu.xyz").read_text())
```

writes

```text
2
Lattice="3.6 0.0 0.0 0.0 3.6 0.0 0.0 0.0 3.6" Properties=species:S:1:pos:R:3:forces:R:3 pbc="T T T" energy=-7.0 config_type=fcc
Cu 0 0 0 0 0 0
Cu 1.8 1.8 0 0 0 0
```

Each frame is a line with the number of atoms, a comment line of
`key=value` pairs, and one line per atom whose columns are described by the
`Properties` key. Frames follow each other directly; blank lines may only end
the file.

## Options

Reading has no options. Writing has one:

| Name | Type | Default | Meaning |
|---|---|---|---|
| `columns` | list of str | all columns | Per-atom columns to write. `element` and `positions` are always written, first. |

## Per-atom columns

`Properties` lists each column as `name:type:width`, with type `R` (real),
`I` (integer), `L` (logical, `T` or `F`) or `S` (string). Some names become
mdapy columns:

| Property | mdapy column | Conversion |
|---|---|---|
| `species:S:1` | `element` | Symbols are capitalized, so `CU` becomes `Cu`. |
| `Z:I:1` | `element` | Atomic numbers; used when there is no `species`. |
| `pos:R:3` | `positions` | |
| `vel:R:3` | `velocities` | Å/fs (GPUMD) to Å/ps. |
| `momenta:R:3` | `velocities` | ASE momenta divided by the masses (from `mass`, or the standard atomic weights), from amu·Å per ASE time unit, Å·(amu/eV)<sup>1/2</sup> ≈ 10.18 fs, to Å/ps. |
| `force:R:3`, `forces:R:3` | `forces` | |
| `mass:R:1`, `masses:R:1` | `mass` | |
| `charge:R:1`, `initial_charges:R:1` | `charge` | |
| `energies:R:1`, `energy_atom:R:1` | `atomic_energies` | |
| `atomic_virials:R:9` | `atomic_virials` | 3×3, row by row. |

When several properties give the same column, such as `vel` and `momenta`,
the first one gives it and the others keep their own names. Any other
property becomes a column of the same name: a float, integer, boolean or
string column, or for a width above 1 an array column. Without `Properties`,
a frame has `species:S:1:pos:R:3`, as plain XYZ files do.

When writing, `element` is written as `species`, `positions` as `pos`,
`velocities` as `vel` in Å/fs, and `atomic_energies` as `energies`. Other
columns keep their names; nested array columns are flattened row by row.
Columns of other types, missing values, and strings that are empty or
contain blanks cannot be written.

## The comment line

| Key | Meaning | In mdapy |
|---|---|---|
| `Lattice` | 9 numbers: the lattice vectors **a**, **b**, **c**, one after the other | `box.matrix` |
| `pbc` | `T` or `F` for each direction; with a `Lattice`, the default is `T T T` | `box.pbc` |
| `Origin` | 3 numbers: the corner of the box; written only when not zero | `box.origin` |
| `Properties` | the per-atom columns, see above | |
| `energy` | total energy, eV | `info["energy"]` |
| `virial` | 9 numbers in Fortran order, or 6 in Voigt order (xx yy zz yz xz xy), eV | `info["virial"]`, 3×3 |
| `stress` | the same layout, eV/Å<sup>3</sup>, positive under tension | `info["stress"]`, 3×3, GPa |

These keys are recognized in any letter case, as GPUMD writes them in lower
case. A frame without `Lattice` gets a non-periodic box around its atoms
(see [Simulation box](../guide/box.md)), and may then not have `pbc` true or
an `Origin`. A `Lattice` of zeros with `pbc` false counts as no `Lattice`.

Every other key becomes an entry of `info`, keeping its case. Its value is
read as an integer, a float, a boolean (`T`, `F`, `True`, `False`, `true`,
`false`, `TRUE`, `FALSE`), an array of one of these (separated by blanks or
commas), JSON after the prefix `_JSON `, or else text. A key without a value
is `True`. Values with blanks are quoted with `""`, `''`, `{}` or `[]`, and a
backslash makes the next character literal.

When writing, numbers, booleans, text and 1-dimensional arrays are written
as such. Other values (arrays of other shapes, mappings, `None`, and text that
would read back as a number) are written as `_JSON` so that they read back
unchanged. `info` cannot use the keys `Lattice`, `pbc`, `Origin` and
`Properties`, in any case.

## Other programs

- **ASE** stores momenta rather than velocities; mdapy reads them and writes
  `vel`, which ASE keeps as an array named `vel`. ASE reads a `charge`
  property as charges computed by a calculator.
- **GPUMD** training files (`train.xyz`) give `energy`, `virial` in eV or
  `stress` in eV/Å<sup>3</sup> with the sign above, and `force` per atom.
  GPUMD's `dump_exyz` writes its pressure tensor under the key `stress`; it is
  positive under compression, the opposite sign. mdapy reads the value as
  written, so check the sign of `stress` from such files.

## References

- Extended XYZ specification, libAtoms: <https://github.com/libAtoms/extxyz>
- A. H. Larsen et al., The atomic simulation environment—a Python library
  for working with atoms, *J. Phys.: Condens. Matter* **29**, 273002 (2017).
  doi:10.1088/1361-648X/aa680e
- Z. Fan et al., GPUMD: A package for constructing accurate machine-learned
  potentials and performing highly efficient atomistic simulations,
  *J. Chem. Phys.* **157**, 114801 (2022). doi:10.1063/5.0106617
