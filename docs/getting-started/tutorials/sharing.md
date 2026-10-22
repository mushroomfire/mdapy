# What a System shares and copies

A `System` never changes. This page shows what that costs: which
operations copy data, which share it, and how to work with large systems
and many frames without copying more than needed.

## Arrays given are copied once

A system copies the arrays it is given, so that changing them afterwards
does not change it.

```python
import numpy as np
import polars as pl

import mdapy
from mdapy.build import build_crystal

positions = np.zeros((2, 3))
system = mdapy.System(positions, mdapy.Box(10.0))
positions[0] = 5.0
system.positions[0]  # still [0, 0, 0]
```

The same holds for `with_columns` and `with_positions`. The values of
`info` are copied and made read-only too.

## Reading shares

`positions` is a read-only NumPy view of the table: reading it is free,
however often, and nothing can change the system through it.

```python
crystal = build_crystal("fcc", 3.615, elements="Cu", repeat=(10, 10, 10))
x = crystal.positions
np.shares_memory(x, crystal.positions)  # True: the same buffer every time
x.flags.writeable  # False
try:
    x[0, 0] = 1.0
except ValueError as error:
    print(error)  # assignment destination is read-only
```

The table is a copy that shares the columns: polars methods that change a
DataFrame in place change the copy only.

```python
table = crystal.table
table.insert_column(0, pl.Series("tag", np.zeros(len(crystal))))
"tag" in crystal.table.columns  # False
```

To move atoms, change a copy and make a new system:

```python
moved = x.copy()
moved[:, 0] += 0.1
shifted = crystal.with_positions(moved)
```

## New systems share what did not change

`with_*` methods make a new system whose unchanged columns are the same
buffers as in the old one. Adding a column does not copy the positions;
new positions do not copy the elements.

```python
labelled = crystal.with_columns(charge=0.0)
np.shares_memory(labelled.positions, crystal.positions)  # True
np.shares_memory(shifted.positions, crystal.positions)  # False: new positions
```

So a trajectory made by moving the atoms of one system holds one copy of
the elements, the types and every other column that does not change, and a
copy of the positions per frame:

```python
rng = np.random.default_rng(0)
frames = [crystal.with_positions(x + rng.normal(0.0, 0.05, x.shape)) for _ in range(10)]
```

`select` copies the rows it keeps, in their new order, since they are no
longer a block of the old table.

```python
half = crystal.select(np.arange(0, len(crystal), 2))
np.shares_memory(half.positions, crystal.positions)  # False
```

## Results

An analysis returns new arrays and leaves the system as it is. Attach the
values when they are needed with `with_columns`, which takes the result
itself:

```python
from mdapy.identify import CommonNeighborAnalysis

result = CommonNeighborAnalysis().compute(crystal)
with_structure = crystal.with_columns(result)  # a structure column
with_structure.table["structure"].dtype  # Int8
```

A [neighbor list](../neighbor.md) serves every system with the positions
and the box it was built for, so it still serves after `with_columns`, which
shares the positions; analyses refuse it once the atoms have moved or the
box has changed. See [Sharing neighbor lists](neighbor-lists.md).
