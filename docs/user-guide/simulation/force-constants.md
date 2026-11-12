<!-- needs: spglib -->

# Force constants

`mdapy.lattice_dynamics.force_constants(cell, frames, *, cutoff=None, symmetry=None)` fits the second
order force constants of a crystal to the forces on atoms displaced in a supercell,
from any calculator or from a calculation with DFT; `displaced_supercells` makes the
supercells, and a `ForceConstants` gives the [phonons](phonons.md) and passes to and
from [phonopy](https://phonopy.github.io/phonopy/) as its `FORCE_CONSTANTS` file.

```python
from mdapy.build import build_crystal
from mdapy.calculators import EAM
from mdapy.lattice_dynamics import (
    displaced_supercells,
    force_constants,
    primitive_cell,
)
from mdapy.simulation import minimize

potential = EAM.zhou(["Cu"])
crystal = minimize(
    build_crystal("fcc", 3.615, elements="Cu"), potential, cell="full", max_stress=1e-4
).system
cell = primitive_cell(crystal)

moved = displaced_supercells(cell, supercell=4)  # one supercell for fcc
frames = [
    s.with_columns(forces=potential.compute(s, ("forces",)).forces) for s in moved
]
constants = force_constants(cell, frames)
constants.parameters, constants.error  # 17 free force constants, their error
constants.phonons().frequencies([0.5, 0.5, 0.5])  # THz at L
```

Random displacements of every atom, each pattern by plus and minus, are the usual
choice for DFT, whose forces carry noise:

```python
moved = displaced_supercells(
    cell, supercell=4, method="random", displacement=0.03, count=2, seed=1
)
frames = [
    s.with_columns(forces=potential.compute(s, ("forces",)).forces) for s in moved
]
constants = force_constants(cell, frames)
```

With VASP, the supercells go out as POSCAR files and the forces come back in the
OUTCAR files, whose atoms VASP holds in the order of the POSCAR, grouped by element:

<!-- skip-test -->
```python
import mdapy as mp

moved = displaced_supercells(cell, supercell=4, method="random", count=2, seed=1)
for k, supercell in enumerate(moved):
    mp.io.write(f"POSCAR-{k:03d}", supercell)
# ... VASP computes each, in its own folder ...
frames = [mp.io.read(f"{k:03d}/OUTCAR", frame=-1) for k in range(len(moved))]
dft = force_constants(cell, frames)
dft.phonons().band_structure("auto")
dft.write("FORCE_CONSTANTS")  # for phonopy
```

| Name | Type | Default | Unit | Meaning |
|---|---|---|---|---|
| `cell` | `System` | | | the cell of the crystal, periodic in every direction, or of a [sheet](sheets.md), with atoms at rest (relax it first) |
| `frames` | sequence of `System` | | | supercells of the cell with atoms displaced, each with a `forces` column in eV/Å, its atoms in any order |
| `cutoff` | float64 or `None` | `None` | Å | fit only the force constants of pairs no farther apart than this, at most half the shortest vector of the supercell less 1e-3 Å; by default those of every pair of the supercell |
| `symmetry` | bool or `None` | `None` | | fit with the symmetry operations of the crystal, found by [spglib](../analysis/space-group.md): `True`, `False`, or by default whenever spglib is installed |

`displaced_supercells(cell, *, supercell, method="finite", displacement=0.01, count=1, seed=None, symmetry=None)`
makes the supercells:

| Name | Type | Default | Unit | Meaning |
|---|---|---|---|---|
| `cell` | `System` | | | the cell of the crystal, as for `force_constants` |
| `supercell` | int32, (3,) int32 or (3, 3) int32 | | | repeats of the cell along its vectors, or the integer matrix whose rows times the vectors of the cell are the vectors of the supercell, as in [`phonons`](phonons.md) |
| `method` | str | `"finite"` | | `"finite"`: one atom displaced in each supercell, the moves of `phonons`; `"random"`: every atom displaced along a random direction, each pattern by plus and minus |
| `displacement` | float64 | 0.01 | Å | how far each displaced atom moves |
| `count` | int32 | 1 | | the number of random patterns, each giving two supercells |
| `seed` | int64 or `None` | `None` | | the seed of the random directions; `None`, new ones each time |
| `symmetry` | bool or `None` | `None` | | for `"finite"`, move only the atoms and along the axes that symmetry does not relate to others, as in `phonons` |

The supercells have the columns of the cell, and in each cell of the supercell the
atoms of the cell in its order. The finite moves are those of `phonons`, one supercell
for fcc and bcc, 2 for hcp and 3 for L1<sub>2</sub> with the symmetry of the crystal,
6 for each atom of the cell without it. A random pattern moves every atom by
`displacement` along a direction of its own, and its supercell by minus the pattern
follows it.

`force_constants` matches the atoms of each frame to those of the supercell by
position, so that frames read back from files, whose atoms may be in another order
and wrapped into the box, give the same force constants; the box of each frame is that
of the supercell, and its supercell matrix comes from the box. The frames must
determine every free force constant: with the symmetry of fcc, one finite move or one
random pattern does; without symmetry, a cell of n atoms needs at least 3n random
patterns. Fewer raise `InvalidParameter`, which
tells how many of the free force constants the frames determine.

Frames of molecular dynamics serve too, as for
[`phonons_from_frames`](thermal-phonons.md#from-frames-of-any-dynamics): the force
constants that best give the forces at the temperature of the frames, with the cell at
their volume.

**Within a cutoff.** By default the force constants are those of the supercell with its
periodic boundaries: every pair of an atom of the cell and an atom of the supercell, under
the operations of the crystal that map the supercell onto itself. With `cutoff`, they are
those of pairs of the crystal no farther apart than `cutoff`, and zero beyond, under every
operation of the crystal, also in a supercell of less symmetry, such as 4 × 4 × 3 cells of
fcc. Each pair is then its own shortest image in the supercell, which is why the cutoff is
at most half the shortest vector of the supercell, and the phonons are those of these
pairs. A cutoff typed at the distance of a shell of neighbors holds the whole shell.

```python
near = force_constants(cell, frames, cutoff=5.0)
near.cutoff, near.parameters  # 5.0 Å, 9 free force constants
```

A cutoff leaves out the force constants of the pairs beyond it, which a many-body
potential such as EAM or NEP couples through the atoms between them, up to twice its
own cutoff. For copper with `EAM.zhou`, in a supercell of 6 × 6 × 6 primitive cells, the
frequencies differ from those of the supercell by up to 0.3 THz with a cutoff of 2.6 Å,
0.11 THz at 4.6 Å, 0.04 THz at 5.8 Å and 0.005 THz at 7.0 Å. `error` falls with the
cutoff only until the anharmonic forces dominate it, here from 4.6 Å, so compare the
frequencies of two cutoffs to choose one.

## Results

`force_constants` returns a `ForceConstants`.

| Attribute or method | Type | Meaning |
|---|---|---|
| `cell` | `System` | the cell of the crystal |
| `supercell` | (3, 3) int64 | the matrix of the supercell; read-only |
| `second` | (n, nS, 3, 3) float64 | the force constants $\Phi$ in eV/Å² between each atom of the cell and each atom of the supercell, which holds S cells, each with the atoms of the cell in their order, the cell itself first, as the `force_constants` of `Phonons`; read-only |
| `cutoff` | float64 or `None` | the cutoff in Å of a fit, beyond which the force constants are zero; `None` for every pair of the supercell, or not a fit |
| `parameters` | int64 or `None` | the number of free force constants that symmetry and the sum rule leave, of a fit; `None` otherwise |
| `error` | float64 or `None` | the relative error of the forces of a fit; `None` otherwise |
| `phonons()` | `Phonons` | the [phonons](phonons.md#results) of the force constants, as `phonons` gives them |
| `write(path, *, full=False)` | | writes the file `FORCE_CONSTANTS` of phonopy |

`error` is the size of the forces that the force constants leave out, over that of the
forces: those of the anharmonic terms and the noise of the calculator. With finite
moves of 0.01 Å it is small. With random patterns of 0.03 Å it is a few percent for
copper. Most of that is the cubic term, which a pattern and its opposite share and
which therefore drops out of the force constants, though not out of the error.

With the finite moves and the forces of a calculator, `force_constants` gives the force
constants of `phonons` where the operations of the crystal turn axes onto axes, as in
fcc and bcc. Otherwise the two differ by the error of the differences, of the order of
the square of the displacement: 1e-4 THz for hcp and L1<sub>2</sub> at 0.01 Å.

`ForceConstants(cell, supercell, second)` makes force constants from an array, such as
the `force_constants` of `Phonons`, and checks its shape against the cell and the
supercell.

## phonopy's FORCE_CONSTANTS

`write` writes the force constants as phonopy's `FORCE_CONSTANTS`, and
`read_force_constants(path, cell, supercell)` reads such a file:

```python
from mdapy.lattice_dynamics import read_force_constants

constants.write("FORCE_CONSTANTS")  # phonopy reads it
back = read_force_constants("FORCE_CONSTANTS", cell, 4)
back.phonons().frequencies([0.5, 0.5, 0.5])
```

| Name | Type | Default | Unit | Meaning |
|---|---|---|---|---|
| `path` | str or path-like | | | the file, in eV/Å² |
| `cell` | `System` | | | the unit cell of phonopy's calculation, its atoms in the order phonopy had them |
| `supercell` | int32, (3,) int32 or (3, 3) int32 | | | the supercell: the transpose of phonopy's supercell matrix, whose columns are the vectors of the supercell; the same for repeats along the vectors, as `--dim 4 4 4` |
| `full` | bool | `False` | | of `write`: the rows of every atom of the supercell rather than those of the atoms of the cell |

The first line of the file holds the number of rows and of columns. Then come blocks
of four lines, one for each pair of an atom of a row and an atom of the supercell, the
row first. Each block is a line that starts with the number of the atom of the row,
counted from 1, then the 3 × 3 force constants of the pair in eV/Å², by rows. The atoms
are in phonopy's order, which MDAPY follows for any supercell matrix: atom by atom of
the cell, and for each the cells of the supercell, the first vector of the cell fastest.

`write` writes the rows of the atoms of the cell, the compact form, which phonopy reads
when its primitive cell is the cell, as with `--pa P`. With `full=True` it writes the
rows of every atom of the supercell, which phonopy reads whatever its primitive cell.

`read_force_constants` reads both forms. It also reads compact files of phonopy whose
primitive cell is smaller than the cell, such as for the cubic cell of fcc: the rows the
file leaves out for atoms of the cell are those of an atom in the file that a
translation of the crystal takes to them. As phonopy does, it takes a header of one
number as both the rows and the columns, and reads only the first number of the first
line of each block. Force constants are in eV/Å², as phonopy writes them from VASP.
Those that other calculators of phonopy give, such as Quantum ESPRESSO in Ry/bohr², must
be converted first.

## Errors

`force_constants` raises `InvalidParameter` for:

- a box that is neither periodic in every direction nor that of a sheet;
- a cell without atoms, or no frames;
- a frame whose box is not that of a supercell of the cell, that has no forces, whose
  atoms are not one at each atom of the supercell, or whose atom has another element or
  type than its atom of the supercell;
- frames that do not determine every free force constant;
- a cutoff that is not positive, or longer than half the shortest vector of the supercell
  less 1e-3 Å, the longest it can be, which the message gives;
- `symmetry=True` without spglib;
- a fit too large for memory, over 2 GB, as a crystal of little symmetry with many atoms
  in its cell in a large supercell: [`phonons`](phonons.md) finds its force constants by
  finite differences instead.

`displaced_supercells` raises `InvalidParameter` for a supercell that is not positive
integers or a matrix of integers with a positive determinant, an unknown method, a
displacement that is not positive, and a count that is not a positive integer.

`ForceConstants` raises `InvalidParameter` for:

- a box that is neither periodic in every direction nor that of a sheet;
- a cell without atoms;
- a supercell that is not positive integers or a matrix of integers with a positive
  determinant;
- force constants that are not of the shape of the cell and the supercell, or not
  finite.

Its `phonons()` raises `InvalidParameter` for an atom with neither element nor mass.

`read_force_constants` raises `FileFormatError` for a file that is not in the format, or
that names an atom the supercell does not have. It raises `InvalidParameter` for a file
of another number of atoms than the supercell has, and for an atom of the cell whose
force constants are neither in the file nor those of an atom that a translation takes
to it.

## Definition

The forces on the atoms $i$ of the supercell, displaced by $\mathbf u_j$ from their
sites, are those of the harmonic energy:

$$
F_{i\alpha} = -\sum_{j\beta} \Phi_{i\alpha, j\beta}\, u_{j\beta} .
$$

The force constants are those of the supercell with its periodic boundaries, or, with a
cutoff, those of the pairs of the crystal no farther apart than it. They obey every
relation that is exact:

- the operations $(R, \mathbf t)$ of the crystal, those that map the supercell onto itself
  without a cutoff and all of them with one: $\Phi(g i, g j) = R\, \Phi(i, j)\, R^{\mathsf T}$;
- the exchange of the two atoms: $\Phi_{i\alpha, j\beta} = \Phi_{j\beta, i\alpha}$;
- the sum rule, so that moving the whole crystal costs no energy:
  $\sum_j \Phi_{i\alpha, j\beta} = 0$.

The free force constants are the components that these relations leave, `parameters`
of them. The fit makes the sum over the frames of
$|\mathbf F - \mathbf F_\text{fit}|^2$ as small as it can, and `error` is
$|\mathbf F - \mathbf F_\text{fit}| / |\mathbf F|$. The displacement of each atom is that
from the nearest atom of the supercell, through the periodic boundaries.

## See also

- [The format of `FORCE_CONSTANTS`](https://phonopy.github.io/phonopy/input-files.html#format-of-force-constants) in the documentation of phonopy.
