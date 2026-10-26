# LAMMPS dump

The format `lammps-dump` reads and writes the text dump files of the LAMMPS
`dump atom` and `dump custom` commands, following LAMMPS stable_30Sep2026,
with any number of frames. Files with extension `.dump` or `.lammpstrj`, or
names starting with `dump.`, are recognized.

```python
import numpy as np
import mdapy as mp

rng = np.random.default_rng(0)
system = mp.System(
    positions=rng.random((4, 3)) * 5,
    box=mp.Box(5.0),
    elements=["Cu", "Cu", "Zr", "Zr"],
).with_columns(velocities=rng.standard_normal((4, 3)))
frames = (
    system.with_columns(positions=system["positions"] + 0.1 * k).with_info(
        timestep=100 * k
    )
    for k in range(3)
)
mp.write("run.dump", frames)
last = mp.read("run.dump", frame=-1)
last.info["timestep"]  # 200
```

## Options

| Name | Type | Default | Meaning |
|---|---|---|---|
| `units` | str | from the file, else `"metal"` | `"metal"` or `"real"`. Reading: overrides `ITEM: UNITS`. Writing: the units of the file. |
| `elements` | list of str | | Reading: the element of each atom type, in type order, for dumps without an `element` column. Writing: the order of the elements as atom types, for a system without a `type` column. |
| `triclinic` | str | `"auto"` | Writing only: `"auto"`, `"general"` or `"restricted"`, as for [LAMMPS data](lammps-data.md#box). |
| `columns` | list of str | all columns | Writing only: the columns to write besides `id`, `type`, `element`, `type_label` and the positions. |

## Frames

Each frame has these items, as LAMMPS writes them:

| Item | Meaning | In mdapy |
|---|---|---|
| `ITEM: UNITS` | unit system; LAMMPS writes it only in the first frame, and it applies to all | |
| `ITEM: TIME` | elapsed time, in ps (metal) or fs (real) | `info["time"]`, ps |
| `ITEM: TIMESTEP` | timestep | `info["timestep"]` |
| `ITEM: NUMBER OF ATOMS` | | number of rows |
| `ITEM: BOX BOUNDS` | the box and its boundary flags | `box` |
| `ITEM: ATOMS` | the names of the columns | the columns |

A boundary flag `pp` makes a direction periodic; `f`, `s` and `m` flags do
not. A triclinic box (`xy xz yz`) is written by LAMMPS as its bounding box,
from which the box is recovered as described in the LAMMPS documentation of
triclinic boxes; a general triclinic box (`abc origin`) keeps its edge
vectors and origin. The same forms are written. Dumps of other styles (local,
cfg, yaml, xyz) and binary dumps raise
[`UnsupportedFormat`](../api.md#mdapy.errors.UnsupportedFormat);
xyz dumps can be read as [extended XYZ](extxyz.md).

## Columns

| Dump columns | mdapy column | Conversion |
|---|---|---|
| `id` | `id` | |
| `type` | `type` | |
| `element` | `element` | |
| `typelabel` | `type_label` | |
| `mol` | `molecule` | |
| `q` | `charge` | |
| `mass` | `mass` | |
| `x y z`, else `xs ys zs`, else `xu yu zu`, else `xsu ysu zsu` | `positions` | scaled coordinates become Cartesian |
| `ix iy iz` | `image` | |
| `vx vy vz` | `velocities` | real: Å/fs to Å/ps, in decimal |
| `fx fy fz` | `forces` | real: kcal/mol/Å to eV/Å |
| `name[1]` ... `name[k]` | `name`, an array column | |

Vectors need all three components. Other columns, including the coordinate
sets not used for the positions, keep their names and values; they are read
as integers when their name starts with `i_` and as floats otherwise. Rows
keep the order of the file.

When writing, `id` (or 1 to N), `type` (from the `type` column, or from the
elements as for [LAMMPS data](lammps-data.md#types-masses-and-elements)),
`element` and `typelabel` if present, and `x y z` come first, followed by
`mol`, `q`, `mass`, `ix iy iz`, `vx vy vz` and `fx fy fz` for the columns
that exist, and then every other numeric column under its own name, array
columns as `name[1]` to `name[k]`. Booleans are written as 0 and 1. The
first frame starts with `ITEM: UNITS`; `ITEM: TIME` is written when `info`
has `time`. Other `info` values are not written.

## See also

- LAMMPS documentation of `dump`, `dump_modify` and `read_dump`:
  <https://docs.lammps.org/dump.html>,
  <https://docs.lammps.org/dump_modify.html>,
  <https://docs.lammps.org/read_dump.html>
- Triclinic boxes in LAMMPS: <https://docs.lammps.org/Howto_triclinic.html>
