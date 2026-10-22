# mdapy files

The format `mdapy` (extension `.mdapy`) stores one or more systems exactly as
they are in memory: every column with its type, the box, the information and
the topology. It is a ZIP archive of [Parquet](https://parquet.apache.org)
tables, so its contents can also be read with any Parquet reader, without
mdapy.

```python
import numpy as np
import mdapy

rng = np.random.default_rng(0)
system = mdapy.System(
    positions=rng.random((4, 3)) * 5,
    box=mdapy.Box(5.0),
    elements=["Cu", "Cu", "Zr", "Zr"],
)
frames = [system.with_info(energy=-3.5 - 0.1 * k, step=k) for k in range(3)]
mdapy.write("run.mdapy", frames)
last = mdapy.read("run.mdapy", frame=-1)
assert last.table.equals(frames[-1].table)
```

## Contents

| Member | Content |
|---|---|
| `manifest.json` | `format` (`"mdapy"`), `version` (1), the mdapy version that wrote it, the time, the number of frames and the units. |
| `frames.parquet` | One row per frame: `frame`, `atoms` (number of atoms), `member` (its table of atoms), `topology` (its topology, or null), `box_matrix`, `box_origin`, `box_pbc`, `info` (all information, as JSON), and typed columns `info.<key>`. |
| `atoms/00000000.parquet`, ... | The table of each frame. Its Parquet metadata also holds the box, the information and the topology of the frame. |
| `topology/00000000/bonds.parquet`, ... | The bonds, angles, dihedrals and impropers of a topology, written once for consecutive frames that share it. |

The typed columns `info.<key>` repeat the information that every frame has
with the same simple type: a number, boolean or text, or a numeric array of
one shape. They make quantities of a whole trajectory readable without its
atoms:

```python
import io
import zipfile

import polars as pl

with zipfile.ZipFile("run.mdapy") as archive:
    frames_table = pl.read_parquet(io.BytesIO(archive.read("frames.parquet")))
frames_table["info.energy"].to_list()  # [-3.5, -3.6, -3.7]
```

The members are stored without further compression; the Parquet tables are
compressed with zstd. Members have a fixed date, so writing the same systems
gives the same members.

## What is kept

- **Columns:** every column with its type, including the element type,
  arrays of any shape, small integer and float types, and text.
- **Information:** numbers, booleans, text, `None`, NumPy arrays with their
  type and shape (including 0-dimensional and empty arrays), nested
  mappings and sequences, and NaN and infinities. NumPy scalars become
  Python numbers. Other values raise
  [`InvalidParameter`](../api.md#mdapy.errors.InvalidParameter).
- **Box** and **topology**, with the type labels and extra columns of its
  tables.

## Files that were not closed

The archive is completed when writing ends. If the writing program stops
earlier, for example when a simulation is killed, the archive has no
directory, `frames.parquet` or manifest. mdapy then reads the complete
members of atoms in order and takes each frame's box, information and
topology from their metadata, so every frame written completely is read.
This works for uncompressed files.

A file written by a newer mdapy with a higher `version` raises
[`UnsupportedFormat`](../api.md#mdapy.errors.UnsupportedFormat).
