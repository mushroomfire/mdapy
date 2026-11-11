# LAMMPS

MDAPY reads and writes the data and dump files of LAMMPS, and computes
energies and forces with any interaction of LAMMPS through its Python
module.

```python
from mdapy.build import build_crystal
from mdapy.io import read, write

crystal = build_crystal("fcc", 3.615, elements="Cu", repeat=(4, 4, 4))
write("copper.data", crystal)  # atom_style atomic, Cu as type 1
read("copper.data").table["element"]  # Cu, from the comment after its mass
```

A LAMMPS input then starts from the file, and its dumps come back as a
trajectory:

```text
units           metal
read_data       copper.data
pair_style      eam/alloy
pair_coeff      * * Cu.eam.alloy Cu
dump            1 all custom 100 dump.lammpstrj id type x y z vx vy vz
```

<!-- skip-test -->
```python
from mdapy.io import read_trajectory

with read_trajectory("dump.lammpstrj", elements=["Cu"]) as frames:
    last = frames[-1]  # velocities in Å/ps
```

## What goes between them

| LAMMPS | MDAPY | Page |
|---|---|---|
| data files (`read_data`, `write_data`) | `mdapy.io.read`, `mdapy.io.write`: atom styles, velocities, topology, type labels, coefficients | [LAMMPS data](../../reference/formats/lammps-data.md) |
| text dumps (`dump atom`, `dump custom`) | `mdapy.io.read_trajectory`, `mdapy.io.write` | [LAMMPS dump](../../reference/formats/lammps-dump.md) |
| the box: bounds, tilt factors, general triclinic boxes | `Box.from_lammps`, `Box.to_lammps` | [Simulation box](../../basics/box.md) |
| any `pair_style` | the calculator `LAMMPS` | [LAMMPS calculator](../simulation/lammps.md) |
| potentials in the eam/alloy format | `EAM` reads and writes them | [EAM](../simulation/eam.md) |
| `replicate` | `mdapy.transform.replicate` | [Periodic copies and images](../build/transform.md) |

MDAPY works in LAMMPS metal units; the data and dump files of real units are
converted when read and written.

## See also

- [LAMMPS documentation](https://docs.lammps.org/).
