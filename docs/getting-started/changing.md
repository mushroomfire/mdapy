# Changing a system

A system never changes. To change something, make a new system with one of
the methods below; the old one stays as it was. This page shows the methods,
then what a new system shares with the old one and what it copies, and why
nothing can change a system in place.

```python
import copy

import numpy as np
import polars as pl

import mdapy as mp
from mdapy.build import build_crystal

crystal = build_crystal("fcc", 3.615, elements="Cu", repeat=(10, 10, 10))
charged = crystal.with_columns(charge=0.0)
moved = crystal.with_columns(positions=crystal["positions"] + [0.5, 0.0, 0.0])
strained = crystal.with_box(crystal.box.matrix * 1.01, scale_positions=True)
labelled = crystal.with_info(energy=-3.54, label="perfect")
"charge" in crystal.table.columns  # False: crystal is as it was
```

## The methods

| Method | Makes a system with |
|---|---|
| `with_columns(*exprs, **named)` | columns added or replaced, the positions too: polars expressions or Series, NumPy arrays, one value for all atoms, or the results of analyses |
| `with_box(box, scale_positions=False)` | another box; with `scale_positions=True` the atoms keep their fractional coordinates and deform with the box |
| `with_info(values, /, **updates)` | other global information: `values` replaces all of it, `{}` clears it; then `updates` adds or replaces entries, and `None` removes one |
| `with_topology(topology)` | another topology, or none with `None` |
| `with_table(table)` | another table of the same atoms in the same order |
| `select(*columns)`, `drop(*columns)`, `rename(names)` | only some columns, in a given order; without some columns; columns renamed |
| `filter`, `gather`, `sort` | some of the atoms, or all in another order; see [Selecting and ordering atoms](selecting.md) |

The names and the meaning of `with_columns`, `select`, `drop`, `rename`,
`filter` and `sort` are those of polars: the first four work on columns,
the last two on rows, that is atoms.

`with_columns` takes polars expressions too, computed on the table:

```python
doubled = charged.with_columns(
    pl.col("charge") + 1.0, height=pl.col("positions").arr.get(2)
)
```

and the result of an analysis, whose per-atom values it adds as columns:

```python
from mdapy.identify import CommonNeighborAnalysis

with_structure = crystal.with_columns(CommonNeighborAnalysis().compute(crystal))
with_structure.table["structure"].dtype  # Int8
```

`with_table` cannot tell whether the atoms of the new table are in the same
order: the topology refers to atoms by their row, so to reorder atoms use
`sort` or `gather`, which renumber the topology, and to remove some use
`filter`.

### Columns

`select` keeps the columns given, in that order, and computes those given
as polars expressions; `drop` removes columns; `rename` gives columns new
names. The positions are always kept: leaving them out, dropping them or
renaming them raises an error.

```python
analysed = with_structure.with_columns(charge=0.0, csp=0.0)
written = analysed.select("element", "positions", "structure")  # for a file
lean = analysed.drop("csp", "charge")
standard = analysed.with_columns(q=0.1).drop("charge").rename({"q": "charge"})
```

A column renamed to a standard name, such as `charge` or `type`, is checked
as one: `type` must hold whole numbers from 1.

### Global information

`with_info` adds, replaces or removes entries, or, given a mapping, replaces
all of them:

```python
labelled.with_info(energy=-3.60)  # energy replaced, label kept
labelled.with_info(label=None)  # label removed
labelled.with_info({})  # nothing left
labelled.with_info({"source": "relaxed"}, step=10)  # source and step only
```

`filter`, `gather` and `sort` keep the global information as it is. After
`filter` some of it may no longer hold, the energy of the whole system for
example; clear or replace it when it does not:

```python
upper = labelled.filter("positions.z > 18").with_info({})
```

## What is shared and what is copied

The data of a system lives in its columns. A new system is a new table
whose columns are those of the old system, except the ones that change:

- columns that do not change are **shared**: both systems point to the same
  memory, and no data is copied;
- columns that change are **new**: only they take new memory;
- arrays given to a system, by its constructor or by these methods, are
  **copied** once, so changing them afterwards does not change the system.

```python
np.shares_memory(charged["positions"], crystal["positions"])  # True: shared
np.shares_memory(moved["positions"], crystal["positions"])  # False: new positions
np.shares_memory(strained["positions"], crystal["positions"])  # False: scaled positions
np.shares_memory(labelled["positions"], crystal["positions"])  # True: only info changed
```

| Operation | New memory |
|---|---|
| `with_info`, `with_topology`, `with_box`, `select`, `drop`, `rename` | none, but for columns `select` computes |
| `with_box(box, scale_positions=True)` | the positions |
| `with_columns(...)` | the columns added or replaced |
| `filter`, `gather`, `sort` | the rows kept, of every column, since they are no longer a block of the old table |
| `system.table`, `system["positions"]` | none |

So frames made by moving the atoms of one system hold one copy of the
elements, the types and every other column that does not change, and a copy
of the positions per frame:

```python
rng = np.random.default_rng(0)
x = crystal["positions"]
frames = [
    crystal.with_columns(positions=x + rng.normal(0.0, 0.05, x.shape))
    for _ in range(10)
]
```

When a new system is made, mdapy checks only the columns that change: a new
box or new global values cost no work on the table, new positions are
checked to be finite, and a new `type` column to be at least 1. A new table
given to `with_table` is checked whole.

## What del does

`del system` removes a name, not data. A column stays in memory as long as
any system, or any array read from one, uses it, and is freed with the last
of them. Systems made from another do not depend on it: each holds its own
references to the columns it shares.

```python
crystal2 = build_crystal("fcc", 3.615, elements="Cu", repeat=(10, 10, 10))
charged2 = crystal2.with_columns(charge=0.0)  # shares the positions
del crystal2
len(charged2)  # 4000: charged2 is whole, the positions are still in use
```

The memory of a column is freed when nothing uses it any more: deleting a
system frees the columns only it had, and none that a system made from it,
or one it was made from, still shares. A [neighbor list](neighbor.md) keeps
the system it was built for, and an array from `system["name"]` keeps the
column it views.

## Shallow or deep copies

A new system is a shallow copy of the old one in memory, since it points to
the same columns; but it behaves as a deep copy, since neither system can
change those columns. Changing a system always means making a new one, so
what is shared is never written to.

There is therefore no copy method. `copy.copy` and `copy.deepcopy` return the
system itself, as they do for a Python tuple, instead of spending memory on a
copy that could never differ:

```python
copy.deepcopy(crystal) is crystal  # True
```

To change values with NumPy, copy the array, change the copy and make a new
system from it:

```python
x = crystal["positions"].copy()  # a writable copy
x[:, 0] += 0.1
shifted = crystal.with_columns(positions=x)
```

## Other processes

Systems, boxes, topologies and triangle meshes pickle, so they go to other
processes, such as the workers of a `ProcessPoolExecutor`, and come back as
unchangeable as they left:

```python
from concurrent.futures import ProcessPoolExecutor

import mdapy
from mdapy.identify import CommonNeighborAnalysis


def fcc_atoms(system):
    return int((CommonNeighborAnalysis().compute(system).structure == 1).sum())


if __name__ == "__main__":
    frames = list(mdapy.read_trajectory("melt.dump"))
    with ProcessPoolExecutor() as pool:
        print(list(pool.map(fcc_atoms, frames)))
```

Pickle is no file format, though: a pickle of one version of mdapy may not
load in another. Save systems with [mdapy.write](io.md).

## Why nothing changes a system in place

Every way into a system gives something that cannot change it:

- `system["name"]` gives read-only NumPy arrays; those of columns and
  components are views of the table, which cannot be made writable:

  ```python
  try:
      crystal["positions"][0, 0] = 1.0
  except ValueError as error:
      print(error)  # assignment destination is read-only
  ```

- `table` is a copy that shares the columns. polars columns are copy on
  write: changing a value of a column, or adding, replacing or removing
  columns of the copy, makes new memory for the copy only.

  ```python
  table = crystal.table
  table.insert_column(0, pl.Series("tag", np.zeros(len(crystal))))
  "tag" in crystal.table.columns  # False
  ```

- `info` is a read-only mapping of read-only copies of the values given; the
  `Box` and the `Topology` cannot change either.
- A system takes no new attributes and lets none be replaced:

  ```python
  try:
      crystal.positions_backup = None
  except AttributeError as error:
      print(error)  # a System cannot be changed; ...
  ```

The internals of a system have names that start with two underscores and are
not part of its interface: code that reaches them on purpose knows what it
does, and nothing in mdapy relies on it.

This is what lets mdapy share without copying. A [neighbor list](neighbor.md)
built for a system serves every system with the same positions and box, such
as one that only has more columns; an analysis given a reference
configuration, such as [atomic strain](../analysis/atomic-strain.md), keeps
it for every configuration it compares; systems can be read from several
threads at once. None of this would hold if a system could change after it
was made.
