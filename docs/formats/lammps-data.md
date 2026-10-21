# LAMMPS data

The format `lammps-data` reads and writes the data files of LAMMPS
`read_data` and `write_data`, following LAMMPS stable_30Sep2026, for the
atom styles of molecular dynamics: `atomic`, `charge`, `bond`, `angle`,
`molecular` and `full`. Files with extension `.data` or `.lmp`, or names
starting with `data.`, are recognized.

```python
import numpy as np
import mdapy

water = mdapy.System(
    positions=[[1.0, 1.0, 1.0], [1.96, 1.0, 1.0], [0.76, 1.93, 1.0]],
    box=mdapy.Box(10.0),
    elements=["O", "H", "H"],
    topology=mdapy.Topology(bonds=[[0, 1], [0, 2]], angles=[[1, 0, 2]]),
).with_columns(charge=[-0.8476, 0.4238, 0.4238], molecule=[1, 1, 1])
mdapy.write("water.data", water)
same = mdapy.read("water.data")
same.table["element"].to_list()  # ['O', 'H', 'H'], from the Masses comments
```

`water.data` is

```text
LAMMPS data file via mdapy 2.0.0.dev0, timestep = 0, units = metal

3 atoms
2 atom types
2 bonds
1 bond types
1 angles
1 angle types

0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 10.0 zlo zhi

Masses

1 15.999 # O
2 1.008 # H

Atoms # full

1 1 1 -0.8476 1 1 1
2 1 2 0.4238 1.96 1 1
3 1 2 0.4238 0.76 1.93 1

Bonds

1 1 1 2
2 1 1 3

Angles

1 1 2 1 3
```

## Options

| Name | Type | Default | Meaning |
|---|---|---|---|
| `atom_style` | str | from the file | Reading: atom style of the Atoms section, when the file does not name it. Writing: the atom style to write. |
| `units` | str | `"metal"` when writing | `"metal"` or `"real"`. Reading: overrides the units named in the first line (`units = ...`, written by `write_data`). |
| `elements` | list of str | | Reading: the element of each atom type, in type order. Writing: the order of the elements as atom types, for a system without a `type` column. |
| `triclinic` | str | `"auto"` | Writing only: how to write the box, see [Box](#box). |

## Atoms

Each atom style has these columns, which become the columns of the same
name:

| Atom style | Atoms line |
|---|---|
| `atomic` | `id type x y z` |
| `charge` | `id type charge x y z` |
| `bond`, `angle`, `molecular` | `id molecule type x y z` |
| `full` | `id molecule type charge x y z` |

Three integers at the end of every line are the image flags, the `image`
column. The atom type may be a number or a type label of the
`Atom Type Labels` section; the `type` column holds the number and, when the
file has labels, `type_label` holds the label.

The atom style is taken from the comment of the section keyword, as in
`Atoms # full`. Without it, lines of 5 values are `atomic` (8 with image
flags) and lines of 7 values `full` (10). Lines of 6 values fit both `charge`
and `molecular`: they are read as `molecular` if the header declares bonds,
angles, dihedrals or impropers, and otherwise need `atom_style=`.

Other atom styles, such as `sphere` or `ellipsoid`, and the sections of
finite-size particles (Ellipsoids, Lines, Triangles, Bodies) raise
[`UnsupportedFormat`](../reference/api.md#mdapy.errors.UnsupportedFormat).

When writing, the style is `atom_style=`, otherwise the style the file was
read with (`info["lammps"]["atom_style"]`), otherwise the simplest style that
holds the system: `full` with a topology or `molecule` column and a `charge`
column, `molecular` with only the first, `charge` with only the second, and
`atomic` otherwise. A missing `molecule` column is written as molecule ID 0,
which LAMMPS reads as "not in a molecule". IDs are the `id` column, or 1 to N.

## Types, masses and elements

Data files have no elements. When reading, every atom type gets an element
from the first of these that names one for every type that has atoms:

1. the `elements=` option;
2. the comment after each mass in the Masses section, as in `1 63.546 # Cu`;
3. the atom type labels, when they are element symbols, with the same case;
4. the masses, each matching exactly one stable element within 0.01 amu.

Otherwise the system has no `element` column. The masses of the Masses
section become the `mass` column.

When writing, atom types are the `type` column. Without it, each element is a
type, in the order of `elements=` or else in the order of first appearance.
The Masses section takes, for each type, the `mass` column, otherwise the
masses read from a file (`info["lammps"]["masses"]`), otherwise the standard
atomic weight of the element; a type whose atoms are all one element gets
the comment `# symbol`. Types without atoms keep the masses and labels read
from the file.

## Box

A data file has no periodicity: the box is periodic in all directions.
A box given by `xlo xhi`, `ylo yhi`, `zlo zhi` and `xy xz yz` becomes a box
with rows (xhi − xlo, 0, 0), (xy, yhi − ylo, 0), (xz, yz, zhi − zlo) and
origin (xlo, ylo, zlo). A general triclinic box, given by `avec`, `bvec`,
`cvec` and `abc origin`, keeps its edge vectors and origin unchanged.
Missing bounds are −0.5 and 0.5, as in LAMMPS.

When writing, `triclinic` selects the form:

| `triclinic` | Box written |
|---|---|
| `"auto"` | Bounds and tilt factors when **a** lies along x and **b** in the xy plane; otherwise the general form. Positions are never rotated. |
| `"general"` | Always the general form, `avec bvec cvec abc origin`. |
| `"restricted"` | Always bounds and tilt factors; the box, positions and velocities are rotated as by [`Box.to_lammps`](../guide/box.md). |

LAMMPS needs right-handed edge vectors, so for a left-handed box **b** and
**c**, and the second and third image flags, are swapped; the lattice and
the positions are unchanged.

## Topology, labels and coefficients

Bonds, Angles, Dihedrals and Impropers become the [topology](../guide/system.md#topology),
with atom IDs turned into rows; type labels of the Bond, Angle, Dihedral and
Improper Type Labels sections become `type_label` columns of its tables.
Bond IDs are not kept: written terms are numbered from 1.

The other sections and header values are kept in `info["lammps"]` and are
written back unchanged:

| Key | Content |
|---|---|
| `atom_style`, `units` | as read |
| `atom_types`, `bond_types`, ... | the type counts of the header |
| `masses` | mass of every atom type |
| `type_labels` | labels of every type, by kind: `atom`, `bond`, ... |
| `coeffs` | each coefficient section (`Pair Coeffs`, `PairIJ Coeffs`, `Bond Coeffs`, the class2 sections, ...), as its style comment and its lines |
| `extra` | the `extra ... per atom` header values |

The timestep of the first line becomes `info["timestep"]`. Coefficients are
not converted between units, so a file read in one unit system cannot be
written with coefficients in the other.

## Units

| `units` | Velocities in the file | Everything else |
|---|---|---|
| `metal` | Å/ps | as in mdapy |
| `real` | Å/fs, converted to and from Å/ps | as in mdapy (g/mol equals amu; coefficients are not converted) |

The conversion moves the decimal point, so it adds no rounding. Other unit
systems raise `UnsupportedFormat`; `units="metal"` reads such a file with
its numbers unchanged.

## Validation

The tests in `tests/io/test_lammps_data.py` check that:

- a hand-written `full` file in real units gives the expected IDs, molecules,
  types, labels, charges, positions, image flags, velocities, masses,
  elements, topology, coefficients and timestep;
- each atom style reads its columns, with and without image flags, and the
  style is inferred, or reported as ambiguous, from the number of values;
- header comments and defaults, tilt factors, general triclinic boxes, type
  labels in every place where LAMMPS accepts them, and all 15 coefficient
  sections are read;
- elements are guessed in the documented order and only when every type is
  named;
- writing and reading gives back the same table, box, information and
  topology, including types without atoms, image flags and `extra` values;
  boxes of each form, left-handed boxes and the `restricted` rotation keep
  the lattice, distances and speeds;
- 35 malformed files raise `FileFormatError` with the right line, and 16
  systems that cannot be written raise the documented error and leave no
  file.

`tests/io/test_lammps_data_lammps.py` runs where the LAMMPS Python module is
installed: LAMMPS stable_30Sep2026 reads files written by mdapy in the
`atomic`, `charge` and `full` styles, in metal and real units and with
restricted and general triclinic boxes, writes them back with `write_data`,
and mdapy reads the same atoms, types, labels, elements, masses, topology,
unwrapped fractional coordinates (to 10<sup>−15</sup>) and velocities
(exactly in metal units, within one unit in the last place in real units,
where LAMMPS stores Å/fs).

## References

- LAMMPS documentation of `read_data` and `write_data`:
  <https://docs.lammps.org/read_data.html>,
  <https://docs.lammps.org/write_data.html>
- A. P. Thompson et al., LAMMPS - a flexible simulation tool for
  particle-based materials modeling at the atomic, meso, and continuum
  scales, *Comput. Phys. Commun.* **271**, 108171 (2022).
  doi:10.1016/j.cpc.2021.108171
