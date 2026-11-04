# 6. Your own files

Now use what you learned on your own simulations. The usual job is: a folder of
LAMMPS dumps, an analysis of every frame, and the result in a file for OVITO or
a figure. This page makes a few dumps first, so that it runs anywhere; replace
them with yours.

```python
import numpy as np

from mdapy.build import build_crystal
from mdapy.io import write

rng = np.random.default_rng(0)
crystal = build_crystal("fcc", 3.615, elements="Cu", repeat=(10, 10, 10))
crystal = crystal.with_columns(id=np.arange(1, len(crystal) + 1))
for step in (0, 1000, 2000, 10000):
    shaken = crystal.with_columns(
        positions=crystal["positions"]
        + rng.normal(0, 0.1 + step / 25000, (len(crystal), 3))
    )
    write(f"dump.{step}.lammpstrj", shaken.with_info(timestep=step))
```

## One frame

`mdapy.io.read` finds the format from the name of the file. Print what you got, and
you know the columns you have:

```python
from mdapy.io import read

frame = read("dump.1000.lammpstrj")
print(frame)
```

```text
System: 4000 atoms (Cu4000)
  box:      36.15 × 36.15 × 36.15 Å, orthogonal, periodic x y z
  columns:  id, type, element, positions (3)
  info:     timestep = 1000
```

If a dump names no elements, its atoms have types. Name them with
`with_columns`, or read a [data file for the topology](../../basics/trajectory.md#a-topology-for-the-frames).

## All the frames

A pattern with `*` takes the files in the order of the numbers in their names,
so `dump.10000` comes after `dump.2000`. Write the analysis of one frame as a
function, and `map` runs it on each frame when that frame is used:

```python
from mdapy.identify import CommonNeighborAnalysis
from mdapy.io import read_trajectory


def analyzed(frame):
    """The frame with the structure of each atom."""
    return frame.with_columns(CommonNeighborAnalysis().compute(frame))


with read_trajectory("dump.*.lammpstrj") as trajectory:
    print(len(trajectory))
    for frame in trajectory.map(analyzed):
        print(frame.info["timestep"], (frame["structure"] == 1).mean())
```

```text
4
0 0.99875
1000 0.904
2000 0.5
10000 0.0
```

The column `structure` holds an index into the names of the analysis, and 1 is
fcc. Only one frame is in memory at a time, however long the trajectory.

## Write the result

The same function, with `mdapy.io.write`. OVITO opens the file with the column
`structure` in it:

```python
with read_trajectory("dump.*.lammpstrj") as trajectory:
    write("analyzed.xyz", trajectory.map(analyzed))
```

## A table of the results

To collect a number from each frame, build a list and make a polars table, which
writes CSV or anything else:

```python
import polars as pl

rows = []
with read_trajectory("dump.*.lammpstrj") as trajectory:
    for frame in trajectory.map(analyzed):
        rows.append(
            {
                "timestep": frame.info["timestep"],
                "fcc": int((frame["structure"] == 1).sum()),
                "z_max": frame["positions.z"].max(),
            }
        )
table = pl.DataFrame(rows)
table.write_csv("results.csv")
print(table)
```

```text
shape: (4, 3)
┌──────────┬──────┬───────────┐
│ timestep ┆ fcc  ┆ z_max     │
│ ---      ┆ ---  ┆ ---       │
│ i64      ┆ i64  ┆ f64       │
╞══════════╪══════╪═══════════╡
│ 0        ┆ 3995 ┆ 34.595473 │
│ 1000     ┆ 3616 ┆ 34.749221 │
│ 2000     ┆ 2000 ┆ 34.774301 │
│ 10000    ┆ 0    ┆ 35.635328 │
└──────────┴──────┴───────────┘
```

## Large systems

Three things help with millions of atoms:

- `mp.set_num_threads(8)` chooses the number of threads. MDAPY uses all of them
  by default, which is rarely what you want on a shared machine.
- Analyses that need the same neighbors can share one list, which saves most of
  the search: [Sharing neighbor lists](../../examples/neighbor-lists.md).
- `render(..., device="gpu")` draws on the graphics card, usually ten times
  faster: [Rendering](../../user-guide/visualization/rendering.md).

## Where next

You have now seen the whole shape of MDAPY. The [cheat sheet](cheat-sheet.md)
sums it up, and the rest of this site is reference: pick the tab of what you
need, and the [tutorials](../../examples/defect-energies.md) work through real
studies.
