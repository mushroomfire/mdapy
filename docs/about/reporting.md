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
| Atom indices | 32 bits, or 64 in a build for more atoms ([64-bit atom indices](../getting-started/installation.md#64-bit-atom-indices)). |
| glibc | The C library in use, on Linux only ([An old Linux](../getting-started/installation.md#an-old-linux)). |
| Logical CPUs | Those of the machine, and how many of them this process may use: fewer in a job of a cluster or under `taskset`. |
| MDAPY threads | How many threads MDAPY is using now ([Parallelism](../basics/parallelism.md)). |
| GPU | The one that traces pictures and how, or `none found` ([Pictures on the GPU](../getting-started/installation.md#pictures-on-the-gpu)). |
| FFmpeg | Whether [movies](../user-guide/visualization/movies.md) can be written. |
| numpy | Its version and, in brackets, the linear algebra library as numpy names it. On conda-forge that is only `blas`, whichever library conda installed. |
| numpy, polars | What MDAPY requires. |
| The packages below them | What the code of MDAPY uses when they are installed, each for a feature of its own. |

The versions of the packages are read from their installed files, so nothing is
imported, and the report takes less than a second.

## Which form

A [new issue](https://github.com/mushroomfire/mdapy/issues/new/choose) starts from one of
five forms; each puts its label on the issue. Choose by what you have:

| Form | Label | When | For example |
|---|---|---|---|
| Bug report | `bug` | MDAPY gives a wrong result, raises an error it should not, or crashes | a structure type that differs from OVITO's; a file that reads with other programs but not with `mdapy.io.read`; a segmentation fault |
| Feature request | `feature request` | something MDAPY does not do yet, or an option of something it does | an analysis of OVITO that MDAPY lacks; a file format; a column more in a result |
| Performance | `performance` | it works, but takes too long or too much memory | an analysis that takes minutes for a million atoms; a run out of memory |
| Documentation | `documentation` | a page is wrong, missing or unclear, or an example does not run | a parameter with no explanation; a picture whose code is not on its page |
| Question | `question` | how to do something, or anything you are not sure where to put | how to compute a property per grain; which analysis suits a liquid |

When unsure between a bug and a question, ask a question: a question that turns out to be a
bug is moved. Search the [issues](https://github.com/mushroomfire/mdapy/issues?q=is%3Aissue)
first, closed ones too: the answer may be there. One problem per issue.

## What to write

The form for a bug asks for what lets a problem be reproduced:

- what happened, and what you expected;
- the shortest script that shows the problem and runs as it is, with a structure
  from [`build_crystal`](../reference/api/build.md#mdapy.build.build_crystal) if the problem
  allows it, or else the smallest file that shows it;
- the full message of the error, all of the traceback;
- the output of `show_versions`.

A feature request asks what it is for, how you would call it, and how its results can be
checked: MDAPY takes a feature when an analytic result, mature software or the paper
that defines the method confirms it. A report on performance asks for the time or memory
measured, for how many atoms and threads, and a script that measures it.
