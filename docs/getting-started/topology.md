# Topology

A `Topology` holds the bonds, angles, dihedrals and impropers of a molecular
system. A system has one or none; it comes from LAMMPS data files and other
formats that store bonds, or is given when a system is made.

```python
import polars as pl

import mdapy as mp

water = mp.System(
    positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
    box=mp.Box(10.0),
    elements=["O", "H", "H"],
    topology=mp.Topology(bonds=[[0, 1], [0, 2]], angles=[[1, 0, 2]]),
)
water.topology.bonds  # a polars DataFrame of two bonds
len(water.topology)  # 3 terms in all
```

## Terms

A topology has four tables, `bonds`, `angles`, `dihedrals` and `impropers`,
with 2, 3, 4 and 4 atoms per term. Each is given as an (M, k) array of atom
indices, all of type 1, or as a polars DataFrame with the columns:

| Column | Type | Meaning |
|---|---|---|
| `atoms` | array of k int64 | Atom indices: rows of the system's table, from 0. |
| `type` | int32 | Type of the term, from 1; 1 when not given. |
| `type_label` | str, optional | Name of the type, for example `"O-H"`. |

Other columns are kept. Atom order follows LAMMPS: the central atom of an
angle is the second one, dihedral atoms follow the chain, and the meaning of
the improper order depends on the improper style.

```python
labelled = mp.Topology(
    bonds=pl.DataFrame(
        {"atoms": [[0, 1], [0, 2]], "type": [1, 1], "type_label": ["O-H", "O-H"]}
    ),
    angles=[[1, 0, 2]],
)
labelled.bonds["type_label"].to_list()  # ['O-H', 'O-H']
```

| Name | Meaning |
|---|---|
| `bonds`, `angles`, `dihedrals`, `impropers` | each table, a copy that shares its columns |
| `tables()` | the four tables by kind |
| `len(topology)` | the number of terms of every kind |
| `max_atom_index()` | the largest atom index of any term, or -1 |
| `remap(mapping)` | a topology with atoms renumbered: `mapping[old]` is the new index, or -1 for an atom removed |

## Topology follows the atoms

A topology refers to atoms by their row in the table, so it changes with
them. [`filter`, `gather` and `sort`](selecting.md) renumber the terms whose
atoms are all kept, and drop the terms that lose an atom:

```python
oxygen = water.filter(element="O")
len(oxygen.topology)  # 0: both bonds and the angle lost a hydrogen
reordered = water.gather([2, 1, 0])
reordered.topology.bonds["atoms"].to_list()  # [[2, 1], [2, 0]]
```

Like a system, a topology never changes: `remap` and the methods of systems
return new ones.

## Errors

`Topology` raises
[`InvalidParameter`](../reference/api.md#mdapy.errors.InvalidParameter) for
a table of the wrong shape or types, negative atom indices, a term that
repeats an atom, missing values, or a type below 1; a system raises it for a
topology that refers to atoms it does not have.
