# Reading and writing files

`mdapy.read` reads one frame of a file into a [`System`](system.md), and
`mdapy.write` writes one or more systems to a file. The format is found from
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

To find frame `k`, mdapy scans the frame boundaries before it, without
reading their atoms; reading `frame=-1` scans the whole file. Uncompressed
files are memory-mapped, so only the parts that are read are loaded.
Compressed files are decompressed into memory first.

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
| `extxyz` | `.xyz`, `.extxyz` | yes | yes | several | [Extended XYZ](../formats/extxyz.md) |
| `lammps-data` | `.data`, `.lmp`, names `data.*` | yes | yes | one | [LAMMPS data](../formats/lammps-data.md) |
| `lammps-dump` | `.dump`, `.lammpstrj`, names `dump.*` | yes | yes | several | [LAMMPS dump](../formats/lammps-dump.md) |
| `poscar` | `.vasp`, `.poscar`, names `POSCAR*`, `CONTCAR*` | yes | yes | one | [VASP POSCAR](../formats/poscar.md) |

`mdapy.detect_format(path)` returns the format of a file name. It ignores a
compression suffix and compares the extension without regard to case; name
patterns such as `data.*` count only when no extension matches. It
raises [`UnsupportedFormat`](../reference/api.md#mdapy.errors.UnsupportedFormat)
when no format, or more than one, fits the name; pass `format=` in that case.
`mdapy.formats()` returns this table as a polars DataFrame.

## Numbers in text files

Text formats are read and written by compiled code, in parallel over lines.

- **Reading** gives the double closest to the decimal number in the file,
  exactly as Python's `float()` does. A decimal point is always `.`, even if
  the program has set a locale that uses a comma, as GUI toolkits do.
- **Writing** uses the shortest text that reads back to the same double, so
  a file written by mdapy reads back to exactly the same numbers.
- **Unit conversions by a power of ten**, such as velocities in Å/fs in a
  file and Å/ps in mdapy, are applied to the decimal digits instead of by a
  multiplication, so they add no rounding and also round-trip exactly.

## Errors

| Exception | When |
|---|---|
| [`FileFormatError`](../reference/api.md#mdapy.errors.FileFormatError) | The file does not follow its format. The message starts with `file:line:`, and the exception has `path` and `line` attributes. |
| [`UnsupportedFormat`](../reference/api.md#mdapy.errors.UnsupportedFormat) | Unknown format name, a format that cannot be read or written, or zstd before Python 3.14. |
| [`InvalidParameter`](../reference/api.md#mdapy.errors.InvalidParameter) | The frame does not exist, or a system cannot be written in the format. |
| `FileNotFoundError` | The file does not exist. |

## Validation

The tests in `tests/io/test_text.py` and `tests/io/test_registry.py` check
that:

- parsed numbers equal Python's `float()` bit for bit for 200,000 random
  doubles in shortest form, numbers printed with six `printf` formats, 50,000
  random decimal strings of up to 50 digits, 19-digit numbers at every power
  of ten from 10<sup>-360</sup> to 10<sup>330</sup>, numbers exactly halfway
  between two doubles, and numbers around the smallest normal double;
- the table of powers of five used by the parser equals its definition,
  recomputed with Python integers;
- numbers scaled by a power of ten equal the decimal number scaled exactly,
  and formatted numbers read back to the same bits;
- results, including the order of strings and which bad line is reported,
  are the same for 1, 3 and 8 threads;
- numbers are read correctly while the process uses a locale with a decimal
  comma;
- files compressed with each method read back unchanged, compression is
  recognized without the suffix, and failed writes leave no file.

## Performance

On an Apple M5 laptop with 10 threads, writing a 184 MB extended XYZ file of
one million atoms with velocities and forces takes 0.15 s, and reading it
takes 0.11 s. Compression is single-threaded: the same file with `.gz` takes
5.6 s to write and 0.6 s to read.

## References

- W. D. Clinger, How to read floating point numbers accurately,
  *Proc. ACM SIGPLAN Conf. Programming Language Design and Implementation*,
  92–101 (1990). doi:10.1145/93542.93557
- D. Lemire, Number parsing at a gigabyte per second,
  *Softw. Pract. Exp.* **51**, 1700–1727 (2021). doi:10.1002/spe.2984
- N. Mushtak and D. Lemire, Fast number parsing without fallback,
  *Softw. Pract. Exp.* **53**, 1467–1471 (2023). doi:10.1002/spe.3198
