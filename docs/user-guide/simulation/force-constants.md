# Force constants

`mdapy.lattice_dynamics.ForceConstants` holds the second order force constants of a
crystal in a supercell, writes them as phonopy's `FORCE_CONSTANTS` file, and
`read_force_constants` reads such a file: force constants pass between MDAPY and
[phonopy](https://phonopy.github.io/phonopy/), from a calculation with VASP to the
[phonons](phonons.md) of MDAPY and back.

```python
from mdapy.build import build_crystal
from mdapy.calculators import EAM
from mdapy.lattice_dynamics import (
    ForceConstants,
    phonons,
    primitive_cell,
    read_force_constants,
)
from mdapy.simulation import minimize

potential = EAM.zhou(["Cu"])
crystal = minimize(
    build_crystal("fcc", 3.615, elements="Cu"), potential, cell="full", max_stress=1e-4
).system
cell = primitive_cell(crystal)
result = phonons(cell, potential, supercell=4)

constants = ForceConstants(cell, result.supercell, result.force_constants)
constants.write("FORCE_CONSTANTS")  # phonopy reads it

back = read_force_constants("FORCE_CONSTANTS", cell, 4)
back.phonons().frequencies([0.5, 0.5, 0.5])  # THz at L, as result gives
```

The force constants of a calculation with phonopy, here with VASP, give the phonons
of MDAPY, with their band structures, densities of states, thermal properties and
plots:

<!-- skip-test -->
```python
import mdapy as mp

cell = mp.io.read("POSCAR")  # the unit cell of phonopy, its atoms in its order
dft = read_force_constants("FORCE_CONSTANTS", cell, 4)  # phonopy --dim 4 4 4
bands = dft.phonons().band_structure("auto")
thermal = dft.phonons().thermal([300.0, 600.0])
```

| Name | Type | Default | Unit | Meaning |
|---|---|---|---|---|
| `cell` | `System` | | | the cell of the crystal, periodic in every direction, or of a [sheet](sheets.md); for `phonons()` each atom needs an `element` or a `mass` |
| `supercell` | int32, (3,) int32 or (3, 3) int32 | | | repeats of the cell along its vectors, or the integer matrix whose rows times the vectors of the cell are the vectors of the supercell, as in [`phonons`](phonons.md) |
| `second` | (n, nS, 3, 3) float64 | | eV/Å² | the force constants between each atom of the cell and each atom of the supercell, which holds S cells, each with the atoms of the cell in their order, the cell itself first: the `force_constants` of `Phonons` |

`read_force_constants(path, cell, supercell)` takes the file, the unit cell of
phonopy's calculation, its atoms in the order phonopy had them, and the supercell.
phonopy's supercell matrix gives the vectors of the supercell as its columns, so the
matrix of MDAPY is its transpose; for repeats along the vectors, as
`--dim 4 4 4`, the two are the same.

## Results

| Attribute or method | Type | Meaning |
|---|---|---|
| `cell` | `System` | the cell of the crystal |
| `supercell` | (3, 3) int64 | the matrix of the supercell; read-only |
| `second` | (n, nS, 3, 3) float64 | the second order force constants in eV/Å²; read-only |
| `phonons()` | `Phonons` | the [phonons](phonons.md#results) of the force constants, as `phonons` gives them |
| `write(path, *, full=False)` | | writes the file `FORCE_CONSTANTS` of phonopy |

**The file.** Its first line holds the number of rows and of columns; then come
blocks of four lines, one for each pair of an atom of a row and an atom of the
supercell, the row first: a line that starts with the number of the atom of the row,
from 1, and the 3 × 3 force constants of the pair in eV/Å², by rows. The atoms are
in phonopy's order, which MDAPY follows for any supercell matrix: atom by atom of the
cell, and for each the cells of the supercell, the first vector of the cell
fastest.

`write` writes the rows of the atoms of the cell, the compact form, which phonopy
reads when its primitive cell is the cell, as with `--pa P`; with `full=True`, the
rows of every atom of the supercell, which phonopy reads whatever its primitive
cell. `read_force_constants` reads both, and the compact files of phonopy whose
primitive cell is smaller than the cell, such as the cubic cell of fcc: the rows of
the atoms of the cell that the file leaves out are those of an atom in the file that
a translation of the crystal takes to them. As phonopy, it takes a header of one
number as the rows and the columns, and reads only the first number of the first
line of each block.

The force constants are in eV/Å², as phonopy writes them from VASP; those of other
calculators of phonopy, such as Quantum ESPRESSO in Ry/bohr², are converted first.

## Errors

`ForceConstants` raises `InvalidParameter` for a box that is neither periodic in
every direction nor that of a sheet, a cell without atoms, a supercell that is not
positive integers or a matrix of integers with a positive determinant, and force
constants that are not of the shape of the cell and the supercell or not finite;
`phonons()` for an atom with neither element nor mass. `read_force_constants` raises
`FileFormatError` for a file that is not in the format, or names an atom that the
supercell does not have, and `InvalidParameter` for a file of another number of
atoms than the supercell has, and for an atom of the cell whose force constants are
neither in the file nor those of an atom that a translation takes to it.

## See also

- [The format of `FORCE_CONSTANTS`](https://phonopy.github.io/phonopy/input-files.html#format-of-force-constants) in the documentation of phonopy.
