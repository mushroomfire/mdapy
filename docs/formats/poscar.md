# VASP POSCAR

The format `poscar` reads and writes the POSCAR and CONTCAR files of VASP,
one structure per file. Files named `POSCAR*` or `CONTCAR*`, or with
extension `.vasp` or `.poscar`, are recognized.

```python
import mdapy

si = mdapy.System(
    positions=[[0.0, 0.0, 0.0], [1.3575, 1.3575, 1.3575]],
    box=mdapy.Box([[0.0, 2.715, 2.715], [2.715, 0.0, 2.715], [2.715, 2.715, 0.0]]),
    elements=["Si", "Si"],
).with_columns(movable=[[False, False, False], [True, True, True]])
mdapy.write("POSCAR", si)
same = mdapy.read("POSCAR")
```

writes

```text
written by mdapy
1.0
0.0 2.715 2.715
2.715 0.0 2.715
2.715 2.715 0.0
Si
2
Selective dynamics
Direct
0 0 0 F F F
0.24999999999999997 0.25 0.25 T T T
```

## Options

| Name | Type | Default | Meaning |
|---|---|---|---|
| `elements` | list of str | | Reading only: the elements of the species, for files without a line of element symbols (VASP 4). |
| `direct` | bool | `True` | Writing only: write fractional (Direct) coordinates; `False` writes Cartesian coordinates. |

## Reading

| Line | Meaning | In mdapy |
|---|---|---|
| 1 | comment | `info["comment"]` |
| 2 | scaling factor: one number, a negative number giving the cell volume in Å<sup>3</sup>, or three numbers scaling x, y and z | applied to the box and Cartesian coordinates |
| 3–5 | lattice vectors | `box.matrix`; the box is periodic in all directions |
| 6 | element symbols; POTCAR labels such as `Na_pv` or `Na_pv/6a2f546d` name the element before `_` | `element` |
| 7 | number of atoms of each element | |
| 8 | optional `Selective dynamics` (first letter S) | |
| next | `Direct`, or `Cartesian` (first letter C or K) | |
| N lines | coordinates, and with selective dynamics three flags `T` (may move) or `F` | `positions`, `movable` |

Words after the coordinates and flags, such as site labels, and text after
`!` are ignored. A VASP 4 file, which has no line of element symbols, needs
`elements=`.

A CONTCAR may continue with velocities: the line after the coordinates gives
their coordinates (`Direct` if it starts with D, Cartesian otherwise; VASP
leaves it blank and ASE writes `Cartesian`), followed by one line per atom in
Å/fs, which become `velocities` in Å/ps. The lattice velocities and vectors
of a variable-cell run (a line starting with L) and the predictor-corrector
block of molecular dynamics are skipped.

## Writing

VASP needs the atoms of each element together. Atoms are grouped by element
in the order of first appearance, keeping their order within an element, so
a system whose elements are already grouped keeps its order. A POSCAR has no
box origin: coordinates are written relative to the origin of the box, and
the box read back has its origin at zero. Fractional coordinates are
computed from the Cartesian positions, so they can differ from round numbers
in the last digit, as in the example above. The comment line is
`info["comment"]` or "written by mdapy". The scaling factor is 1. With a
`movable` column (three booleans per atom) the file has selective dynamics,
and with `velocities` a velocity block in Å/fs after a blank line, as VASP
writes it.

## Validation

The tests in `tests/io/test_poscar.py` check that:

- a CONTCAR with POTCAR labels, selective dynamics, site labels and comments,
  and velocities gives the expected box, elements, positions, flags and
  velocities (scaled exactly in decimal);
- the three kinds of scaling factor, both coordinate modes, VASP 4 files with
  `elements=`, and each kind of block after the coordinates are read as
  documented;
- writing groups elements in the order of first appearance and reads back to
  the same elements, positions (to 10<sup>−14</sup> Å), flags and velocities;
- 15 malformed files raise `FileFormatError` with the right line, and
  systems that cannot be written raise the documented error and leave no
  file.

`tests/io/test_poscar_ase.py` runs where ASE is installed: files written by
ASE read to the cell, positions, fixed atoms and velocities that ASE reads,
and files written by mdapy read in ASE to the values that were written.

## References

- POSCAR in the VASP wiki: <https://www.vasp.at/wiki/index.php/POSCAR>
- CONTCAR in the VASP wiki: <https://www.vasp.at/wiki/index.php/CONTCAR>
