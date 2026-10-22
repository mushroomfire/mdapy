# Systems and topology

A `System` is one atomic configuration: a table with one row per atom, a
[simulation box](box.md), global information such as the energy, and for
molecules an optional `Topology` of bonds, angles, dihedrals and impropers.
A system never changes; every operation returns a new one.

```python
import numpy as np
import mdapy

water = mdapy.System(
    positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
    box=mdapy.Box(10.0),
    elements=["O", "H", "H"],
    info={"energy": -14.2},
    topology=mdapy.Topology(bonds=[[0, 1], [0, 2]], angles=[[1, 0, 2]]),
)
water.table  # polars DataFrame with columns positions, element
water.positions  # (3, 3) NumPy array, read-only, no copy
charged = water.with_columns(charge=[-0.8, 0.4, 0.4])
hydrogens = water.select(water.table["element"] == "H")
```

## Creating a system

| Name | Type | Default | Unit | Meaning |
|---|---|---|---|---|
| `positions` | (N, 3) array | | Å | Positions. Give either `positions` or `table`. |
| `box` | `Box` or lattice vectors | non-periodic box around the atoms | Å | Simulation box. |
| `table` | polars DataFrame | | | Per-atom table with a `positions` column. |
| `elements` | str or N str | | | Chemical symbols; one symbol applies to all atoms. |
| `types` | int or N int | | | Atom types, from 1; one value applies to all atoms. |
| `info` | mapping | `{}` | | Global information, for example `energy` (eV), `virial` (eV), `timestep`. |
| `topology` | `Topology` | `None` | | Bonds, angles, dihedrals, impropers. |

Standard columns are converted to their types when a system is created (see
the [Glossary](../reference/glossary.md#per-atom-properties)): `positions`,
`velocities` and `forces` to arrays of 3 floats, `element` to the element
type, and integer columns such as `type`, `id` and `image` to integers.
Integer columns accept floats only if they are whole numbers, so `1.0` is
read as type 1 but `1.5` raises an error. Other columns are kept as they are.
Inconsistent input raises
[`InvalidParameter`](../reference/api.md#mdapy.errors.InvalidParameter).

## Reading data

| Name | Meaning |
|---|---|
| `table` | Per-atom properties as a polars DataFrame. |
| `box` | The `Box`. |
| `info` | Global information, read-only. |
| `topology` | The `Topology`, or `None`. |
| `positions` | (N, 3) positions in Å: a read-only NumPy view of the table, without a copy. |
| `len(system)` | Number of atoms. |

Everything else is read from the table with polars, for example
`system.table["element"]` or `system.table.filter(pl.col("type") == 2)`.

## Changing a system

Each method returns a new system. Columns that are not changed are shared
between the old and the new system rather than copied. Arrays given to a
system, by its constructor or by these methods, are copied, so changing
them afterwards does not change the system.

| Method | Meaning |
|---|---|
| `with_columns(*exprs, **named)` | Add or replace columns: polars expressions or Series, NumPy arrays, or one value for all atoms. |
| `with_table(table)` | Replace the table. It must have the same atoms in the same order, because the topology refers to rows by position. |
| `with_positions(positions)` | Replace the positions. |
| `with_box(box, scale_positions=False)` | Replace the box. With `scale_positions=True` the atoms keep their fractional coordinates and deform with the box. |
| `with_info(**values)` | Add or replace global values; a value of `None` removes the entry. |
| `with_topology(topology)` | Replace the topology, or remove it with `None`. |
| `select(atoms)` | Keep some atoms, in the given order: a boolean mask or integer indices. |

```python
moved = water.with_positions(water.positions + [1.0, 0.0, 0.0])
bigger = water.with_box(mdapy.Box(20.0), scale_positions=True)
labelled = water.with_info(energy=None, label="monomer")
```

## Topology

A `Topology` holds four tables, `bonds`, `angles`, `dihedrals` and
`impropers`, with 2, 3, 4 and 4 atoms per term. Each table has the columns:

| Column | Type | Meaning |
|---|---|---|
| `atoms` | array of k int64 | Atom indices: 0-based rows of the system's table. |
| `type` | int32 | Type of the term, from 1. |
| `type_label` | str, optional | Name of the type, for example `"C-H"`. |

Other columns are kept. Atom order follows LAMMPS: the central atom of an
angle is the second one, dihedral atoms follow the chain, and the meaning of
the improper order depends on the improper style.

```python
import polars as pl

t = mdapy.Topology(
    bonds=pl.DataFrame(
        {"atoms": [[0, 1], [0, 2]], "type": [1, 1], "type_label": ["O-H", "O-H"]}
    ),
    angles=[[1, 0, 2]],  # an (M, 3) array: all types are 1
)
t.bonds
```

`select` keeps the topology consistent: terms whose atoms are all selected
are renumbered, and terms that lose an atom are dropped.

```python
oxygen = water.select([0])
len(oxygen.topology)  # 0: both bonds and the angle lost a hydrogen
```
