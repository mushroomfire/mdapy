# Systems

A `System` is one atomic configuration: a table with one row per atom, a
[simulation box](box.md), global information such as the energy, and for
molecules an optional [topology](topology.md) of bonds, angles, dihedrals
and impropers. Every analysis, builder, reader and calculator of mdapy takes
and returns systems.

```python
import numpy as np
import polars as pl

import mdapy as mp

water = mp.System(
    positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
    box=mp.Box(10.0),
    elements=["O", "H", "H"],
    info={"energy": -14.2},
)
print(water)
# System: 3 atoms (H2 O)
#   box:      10 × 10 × 10 Å, orthogonal, periodic x y z
#   columns:  positions (3), element
#   info:     energy = -14.2
water["positions"]  # (3, 3) NumPy array of the positions, read-only
water.table  # the per-atom table, a polars DataFrame
```

A system never changes. Methods such as `with_columns`, `filter` and `sort`
return a new system and leave the old one as it is; see
[Changing a system](changing.md) and
[Selecting and ordering atoms](selecting.md).

## Creating a system

| Name | Type | Default | Unit | Meaning |
|---|---|---|---|---|
| `positions` | (N, 3) array | | Å | Positions. Give either `positions` or `table`. |
| `box` | `Box` or lattice vectors | non-periodic box around the atoms | Å | Simulation box; a number or three numbers give an orthogonal periodic box. |
| `table` | polars DataFrame | | | Per-atom table with a `positions` column of 3 floats per atom. |
| `elements` | str or N str | | | Chemical symbols; one symbol applies to all atoms. |
| `types` | int or N int | | | Atom types, from 1; one value applies to all atoms. |
| `info` | mapping | `{}` | | Global information, for example `energy` (eV), `virial` (eV), `timestep`. |
| `topology` | `Topology` | `None` | | Bonds, angles, dihedrals and impropers, see [Topology](topology.md). |

From positions, as above, or from a table that already holds the columns:

```python
table = pl.DataFrame(
    {
        "positions": pl.Series(
            [[0.0, 0.0, 0.0], [1.8, 1.8, 0.0]], dtype=pl.Array(pl.Float64, 3)
        ),
        "element": ["Cu", "Ni"],
        "velocities": [[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]],
    }
)
pair = mp.System(table=table, box=mp.Box(3.6))
```

Most systems come from files with [`mp.read`](io.md) or from the
[structure builders](../build/crystals.md), which return them the same way.

The arrays given to a system are copied, so changing them afterwards does
not change the system. A box that is not given encloses the atoms and is not
periodic.

## Columns

The table holds one row per atom and any number of columns. Columns with a
standard name have a fixed type, which they are converted to when a system
is made:

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `positions` | 3 floats | Å | positions, required |
| `velocities` | 3 floats | Å/ps | velocities |
| `forces` | 3 floats | eV/Å | forces |
| `element` | element | | chemical symbol |
| `type` | int32 | | atom type, from 1 |
| `id` | int64 | | atom identifier |
| `molecule` | int64 | | molecule identifier |
| `mass` | float64 | amu | mass |
| `charge` | float64 | e | charge |
| `image` | 3 int32 | | periodic image flags |

Integer columns accept floats only if they are whole numbers, so `1.0` is
read as type 1 but `1.5` raises an error. Standard columns must not have
missing values, and positions must be finite. Every other column is kept as
it is: numbers, booleans, text, and arrays of numbers such as the
`orientation` of [polyhedral template matching](../analysis/ptm.md). The
[Glossary](../reference/glossary.md#per-atom-properties) lists the columns
that mdapy itself writes.

The `element` column holds chemical symbols; underneath they are atomic
numbers, so elements compare as text and sort by atomic number:

```python
water.table["element"].to_list()  # ['O', 'H', 'H']
water.table["element"].to_physical().to_list()  # [8, 1, 1]
```

## Reading data

| Name | Meaning |
|---|---|
| `system["name"]` | The values of a column, a component or an expression, one per atom, as a read-only NumPy array; `system["positions"]` are the (N, 3) positions in Å, without a copy. |
| `table` | The per-atom table as a polars DataFrame. |
| `box` | The [`Box`](box.md). |
| `info` | Global information, a read-only mapping. |
| `topology` | The [`Topology`](topology.md), or `None`. |
| `len(system)` | The number of atoms. |

### Values by name

`system["name"]` gives the values of each atom as a NumPy array. The name
is a column, a component of an array column written `column.component`, or
an expression over the columns, written as in the
[conditions of `filter`](selecting.md#conditions-as-text):

```python
from mdapy.build import build_crystal

alloy = build_crystal("fcc", 3.6, elements="Cu", repeat=(3, 3, 3))
alloy = alloy.with_columns(
    element=np.where(np.arange(len(alloy)) % 4 == 0, "Ni", "Cu"),
    charge=np.linspace(-1.0, 1.0, len(alloy)),
)
alloy["charge"]  # (108,) float64
alloy["positions.z"].max()  # the largest z
alloy["positions"]  # (108, 3)
alloy["sqrt(positions.x**2 + positions.y**2)"]  # distance from the z axis
alloy["element"]  # ['Ni', 'Cu', 'Cu', ...]
```

| Name | Example | Gives |
|---|---|---|
| a column | `"charge"`, `"element"`, `"positions"` | its values, (N,) or (N, k) for an array column |
| a component | `"positions.z"`, `"velocities.x"`, `"group.1"` | one component, (N,) |
| an expression | `"positions.x**2 + positions.y**2"`, `"charge > 0"` | the value of each atom, (N,) |

Components are named as in conditions: `x`, `y`, `z` for 3 values, `x`, `y`,
`z`, `w` for 4, `xx` to `zz` by rows for 9, and the index from 0 for any
number. A column or a component is read where it is, without a copy, for
numbers without missing values; an expression is computed. The arrays are
read-only, so they cannot change the system.

Every column is read this way, the positions too: a system has no
attribute for a column, so that each is read and changed the same way, by
`system["name"]` and [`with_columns`](changing.md). A system is indexed only
by names: atoms are chosen with [`filter`, `gather` and `sort`](selecting.md),
and a system is not a sequence of atoms to loop over.

### The table

Everything else is read from the table, a polars DataFrame on which every
polars operation works, for example counts and averages by element:

```python
alloy.table.group_by("element").agg(pl.len(), pl.col("charge").mean())
alloy.table["positions"].arr.get(2).max()  # the largest z, as polars writes it
```

The table is a copy of the system's own: changing it does not change the
system. To make a changed system, see [Changing a system](changing.md).

## Printing a system

`print(system)` gives a summary of a few lines:

```text
System: 108 atoms (Cu81 Ni27)
  box:      10.8 × 10.8 × 10.8 Å, orthogonal, periodic x y z
  columns:  positions (3), element, charge
  info:     energy = -3.54, timestep = 1000
  topology: 2 bonds, 1 angle
```

- the number of atoms and the elements, the most frequent first;
- the box: its lengths for an orthogonal box, with its origin when that is
  not zero, or for a triclinic box the lengths of **a**, **b**, **c** and the
  angles between them; and its periodic directions;
- the columns, with the number of components of array columns in
  parentheses;
- the global information, if any: numbers and short text with their values,
  arrays with their shape;
- the topology, if any: the number of terms of each kind.

## Errors

The constructor raises
[`InvalidParameter`](../reference/api.md#mdapy.errors.InvalidParameter) for
inconsistent input: neither or both of `positions` and `table`, positions not
of shape (N, 3) or not finite, unknown chemical symbols, types below 1,
missing values in standard columns, values that do not fit the type of a
standard column, or topology indices beyond the number of atoms.
