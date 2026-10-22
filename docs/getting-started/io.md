# Reading and writing files

`mdapy.read` reads one frame of a file into a [`System`](system.md), and
`mdapy.write` writes one or more systems to a file. To go through the frames
of a file one at a time, see [Trajectories](trajectory.md). The format is found from
the file name, and compressed files are handled transparently.

```python
import mdapy

water = mdapy.System(
    positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
    box=mdapy.Box(10.0),
    elements=["O", "H", "H"],
    info={"energy": -14.2},
)
mdapy.write("water.xyz", water)
same = mdapy.read("water.xyz")
assert same.table.equals(water.table.select(same.table.columns))

mdapy.write("frames.xyz.gz", [water, water.with_positions(water.positions + 1.0)])
last = mdapy.read("frames.xyz.gz", frame=-1)
mdapy.formats()  # the formats mdapy can read and write
```

## Reading

`mdapy.read(path, format=None, *, frame=0, **options)`

| Name | Type | Default | Meaning |
|---|---|---|---|
| `path` | str or path | | The file. |
| `format` | str | from the name | Format name; see [Formats](#formats). |
| `frame` | int | `0` | Frame to read from a file that holds several; negative values count from the end. |
| `**options` | | | Options of the format, listed on its page. |

It returns a `System`. Compressed files are recognized from their first
bytes, whatever their name: gzip, bzip2, xz, and zstd with Python 3.14 or
newer.

Reading frame `k` skips the frames before it without reading their atoms;
`frame=-1` goes through the whole file. Of an uncompressed file only the
parts that are read are loaded; a compressed file is decompressed into
memory first.

## Writing

`mdapy.write(path, systems, format=None, **options)`

| Name | Type | Default | Meaning |
|---|---|---|---|
| `path` | str or path | | The file; it is replaced if it exists. |
| `systems` | `System` or iterable of `System` | | One system, or several for a format that holds several frames. |
| `format` | str | from the name | Format name. |
| `**options` | | | Options of the format, listed on its page. |

An iterator of systems, such as a generator, is written one system at a
time, so a long trajectory need not fit in memory. A name ending in `.gz`,
`.bz2`, `.xz` or `.zst` compresses the file. If writing fails, the
incomplete file is removed.

## Formats

| Name | Extensions | Read | Write | Frames | Page |
|---|---|---|---|---|---|
| `extxyz` | `.xyz`, `.extxyz` | yes | yes | several | [Extended XYZ](../reference/formats/extxyz.md) |
| `lammps-data` | `.data`, `.lmp`, names `data.*` | yes | yes | one | [LAMMPS data](../reference/formats/lammps-data.md) |
| `lammps-dump` | `.dump`, `.lammpstrj`, names `dump.*` | yes | yes | several | [LAMMPS dump](../reference/formats/lammps-dump.md) |
| `poscar` | `.vasp`, `.poscar`, names `POSCAR*`, `CONTCAR*` | yes | yes | one | [VASP POSCAR](../reference/formats/poscar.md) |
| `mdapy` | `.mdapy` | yes | yes | several | [mdapy files](../reference/formats/mdapy.md) |

`mdapy.detect_format(path)` returns the format of a file name. It ignores a
compression suffix and compares the extension without regard to case; name
patterns such as `data.*` count only when no extension matches. It
raises [`UnsupportedFormat`](../reference/api.md#mdapy.errors.UnsupportedFormat)
when no format, or more than one, fits the name; pass `format=` in that case.
`mdapy.formats()` returns this table as a polars DataFrame.

## Numbers in text files

- **Reading** gives the double closest to the decimal number in the file,
  exactly as Python's `float()` does. A decimal point is always `.`, even if
  the program has set a locale that uses a comma, as GUI toolkits do.
- **Writing** uses the shortest text that reads back to the same double, so
  a file written by mdapy reads back to exactly the same numbers.
- **Unit conversions by a power of ten**, such as velocities in Å/fs in a
  file and Å/ps in mdapy, add no rounding and also round-trip exactly.

## Errors

| Exception | When |
|---|---|
| [`FileFormatError`](../reference/api.md#mdapy.errors.FileFormatError) | The file does not follow its format. The message starts with `file:line:`, and the exception has `path` and `line` attributes. |
| [`UnsupportedFormat`](../reference/api.md#mdapy.errors.UnsupportedFormat) | Unknown format name, a format that cannot be read or written, or zstd before Python 3.14. |
| [`InvalidParameter`](../reference/api.md#mdapy.errors.InvalidParameter) | The frame does not exist, or a system cannot be written in the format. |
| `FileNotFoundError` | The file does not exist. |
