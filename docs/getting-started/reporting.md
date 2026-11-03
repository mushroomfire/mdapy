# Reporting a problem

Problems and questions go to the
[issue tracker](https://github.com/mushroomfire/mdapy/issues). `show_versions`
prints what a report needs to say about the installation and the machine: paste
its output into the issue.

```python
import mdapy as mp

mp.show_versions()
```

```text
MDAPY          2.0.0
Atom indices   32 bits
Python         3.13.16 | packaged by Anaconda, Inc. | (main, Oct  1 2026, 20:19:24) [Clang 20.1.8 ]
Platform       macOS-27.0.1-arm64-arm-64bit-Mach-O

CPU            Apple M5
Logical CPUs   10, 10 usable
MDAPY threads  10
Memory         24.0 GB
GPU            Apple M5 (Metal, ray tracing)
FFmpeg         not found

numpy          2.5.4 (accelerate)
polars         2.0.0

ase            3.29.0
freud-analysis 3.6.1
lammps         2026.9.30
matplotlib     3.11.2
ovito          3.17.1
pymatgen-core  2026.10.16
seekpath       2.2.2
spglib         not installed
```

## Parameters

| Name | Type | Default | Meaning |
|---|---|---|---|
| `file` | text stream | the standard output | Where to print the report. |

```python
import mdapy as mp

with open("report.txt", "w") as report:
    mp.show_versions(file=report)
```

## What it tells

| Line | Meaning |
|---|---|
| Atom indices | 32 bits, or 64 in a build for more atoms ([64-bit atom indices](installation.md#64-bit-atom-indices)). |
| Logical CPUs | Those of the machine, and how many of them this process may use: fewer in a job of a cluster or under `taskset`. |
| MDAPY threads | How many threads MDAPY is using now ([Parallelism](parallelism.md)). |
| GPU | The one that traces pictures and how, or `none found` ([Pictures on the GPU](installation.md#pictures-on-the-gpu)). |
| FFmpeg | Whether [movies](../visualization/movies.md) can be written. |
| numpy | Its version and, in brackets, the linear algebra library as numpy names it. On conda-forge that is only `blas`, whichever library conda installed. |
| numpy, polars | What MDAPY requires. |
| The packages below them | What the code of MDAPY uses when they are installed, each for a feature of its own. |

The versions of the packages are read from their installed files, so nothing is
imported, and the report takes less than a second.

## What to write

The [bug report form](https://github.com/mushroomfire/mdapy/issues/new/choose) of the
issue tracker asks for what lets a problem be reproduced:

- what happened, and what you expected;
- the shortest script that shows the problem and runs as it is, with a structure
  from [`build_crystal`](../reference/api.md#mdapy.build.build_crystal) if the problem
  allows it, or else the smallest file that shows it;
- the full message of the error, all of the traceback;
- the output of `show_versions`.
