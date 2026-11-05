# Reading and writing files

MDAPY, [OVITO](https://www.ovito.org) and [ASE](https://wiki.fysik.dtu.dk/ase/)
read and write LAMMPS dumps, LAMMPS data files and extended XYZ of the same
systems, from 4 000 to 2 million atoms. A program is timed only where it
reads the positions that MDAPY wrote, or writes a file that MDAPY reads back
as the system.

## Results

An Apple M5 (10 cores: 4 performance, 6 efficiency), 24 GB, macOS 27, with an
editor and a browser open. Python 3.13, MDAPY 2.0.0a2 in development built with
Apple clang 21, OVITO 3.17.1 and ASE 3.29.0 from pip. The
[analyses](comparison.md) run on another machine, so their times do not compare
with these.

![Time of OVITO and of ASE over the time of MDAPY for each case and size](images/files.png)

### What they show

- **OVITO**: from 32 000 atoms on, MDAPY reads the three formats 3.2 to
  7.6 times faster and writes them 8.6 to 45 times faster. At 4 000 atoms
  MDAPY reads 1.0 to 1.3 times faster, in about 0.4 ms, and writes 2.7 to
  6.0 times faster.
- **ASE**: MDAPY reads 5.2 to 62 times faster and writes 8.4 to 126 times
  faster. ASE reads a data file of 2 million atoms in 2.9 s and 4.0 GB of
  memory.
- **2 million atoms** take MDAPY 0.02 to 0.05 s to read in each format, and
  0.02 to 0.04 s to write with every digit.

### Times and memory

Every case at every size: the shortest of three times, and the memory each
call took beyond the system, the peak resident memory of the process minus
its peak before the timings.

#### OVITO

| Case | Atoms | MDAPY 2.0.0a2 (s) | OVITO 3.17.1 (s) | OVITO / MDAPY | MDAPY (MB) | OVITO (MB) |
|---|---:|---:|---:|---:|---:|---:|
| Read a LAMMPS dump | 4,000 | 0.0004 | 0.0004 | 1.00 | 0 | 1 |
| Read a LAMMPS dump | 32,000 | 0.0009 | 0.0028 | 3.18 | 4 | 9 |
| Read a LAMMPS dump | 256,000 | 0.0034 | 0.022 | 6.39 | 51 | 47 |
| Read a LAMMPS dump | 2,048,000 | 0.023 | 0.175 | 7.58 | 213 | 127 |
| Write a LAMMPS dump | 4,000 | 0.0003 | 0.0018 | 5.97 | 0 | 0 |
| Write a LAMMPS dump | 32,000 | 0.0010 | 0.013 | 13.52 | 4 | 0 |
| Write a LAMMPS dump | 256,000 | 0.0044 | 0.106 | 24.12 | 24 | 0 |
| Write a LAMMPS dump | 2,048,000 | 0.022 | 0.835 | 37.28 | 0 | 0 |
| Read a LAMMPS data file | 4,000 | 0.0004 | 0.0006 | 1.33 | 0 | 1 |
| Read a LAMMPS data file | 32,000 | 0.0012 | 0.0042 | 3.59 | 7 | 11 |
| Read a LAMMPS data file | 256,000 | 0.0065 | 0.033 | 5.09 | 53 | 31 |
| Read a LAMMPS data file | 2,048,000 | 0.051 | 0.260 | 5.11 | 278 | 85 |
| Write a LAMMPS data file | 4,000 | 0.0007 | 0.0018 | 2.74 | 2 | 0 |
| Write a LAMMPS data file | 32,000 | 0.0015 | 0.013 | 8.63 | 10 | 0 |
| Write a LAMMPS data file | 256,000 | 0.0068 | 0.107 | 15.89 | 40 | 0 |
| Write a LAMMPS data file | 2,048,000 | 0.039 | 0.846 | 21.88 | 83 | 0 |
| Read extended XYZ | 4,000 | 0.0004 | 0.0005 | 1.22 | 0 | 1 |
| Read extended XYZ | 32,000 | 0.0010 | 0.0030 | 3.18 | 4 | 9 |
| Read extended XYZ | 256,000 | 0.0036 | 0.024 | 6.57 | 34 | 22 |
| Read extended XYZ | 2,048,000 | 0.028 | 0.193 | 6.93 | 122 | 0 |
| Write extended XYZ | 4,000 | 0.0003 | 0.0019 | 5.71 | 0 | 0 |
| Write extended XYZ | 32,000 | 0.0010 | 0.014 | 14.37 | 4 | 0 |
| Write extended XYZ | 256,000 | 0.0040 | 0.115 | 28.91 | 18 | 0 |
| Write extended XYZ | 2,048,000 | 0.020 | 0.919 | 44.97 | 0 | 0 |

#### ASE

| Case | Atoms | MDAPY 2.0.0a2 (s) | ASE 3.29.0 (s) | ASE / MDAPY | MDAPY (MB) | ASE (MB) |
|---|---:|---:|---:|---:|---:|---:|
| Read a LAMMPS dump | 4,000 | 0.0004 | 0.0021 | 5.19 | 0 | 1 |
| Read a LAMMPS dump | 32,000 | 0.0009 | 0.017 | 19.40 | 4 | 12 |
| Read a LAMMPS dump | 256,000 | 0.0034 | 0.134 | 39.03 | 51 | 101 |
| Read a LAMMPS dump | 2,048,000 | 0.023 | 1.061 | 45.85 | 213 | 397 |
| Read a LAMMPS data file | 4,000 | 0.0004 | 0.0053 | 12.07 | 0 | 16 |
| Read a LAMMPS data file | 32,000 | 0.0012 | 0.042 | 36.58 | 7 | 75 |
| Read a LAMMPS data file | 256,000 | 0.0065 | 0.347 | 53.62 | 53 | 568 |
| Read a LAMMPS data file | 2,048,000 | 0.051 | 2.883 | 56.59 | 278 | 4019 |
| Write a LAMMPS data file | 4,000 | 0.0007 | 0.0056 | 8.44 | 2 | 0 |
| Write a LAMMPS data file | 32,000 | 0.0015 | 0.043 | 27.78 | 10 | 4 |
| Write a LAMMPS data file | 256,000 | 0.0068 | 0.346 | 51.18 | 40 | 42 |
| Write a LAMMPS data file | 2,048,000 | 0.039 | 2.832 | 73.26 | 83 | 52 |
| Read extended XYZ | 4,000 | 0.0004 | 0.0034 | 9.11 | 0 | 1 |
| Read extended XYZ | 32,000 | 0.0010 | 0.027 | 28.51 | 4 | 13 |
| Read extended XYZ | 256,000 | 0.0036 | 0.215 | 59.98 | 34 | 96 |
| Read extended XYZ | 2,048,000 | 0.028 | 1.734 | 62.11 | 122 | 501 |
| Write extended XYZ | 4,000 | 0.0003 | 0.0050 | 14.90 | 0 | 0 |
| Write extended XYZ | 32,000 | 0.0010 | 0.039 | 38.73 | 4 | 3 |
| Write extended XYZ | 256,000 | 0.0040 | 0.328 | 82.11 | 18 | 9 |
| Write extended XYZ | 2,048,000 | 0.020 | 2.571 | 125.83 | 0 | 0 |

## Cases

| Case | MDAPY | OVITO | ASE |
|---|---|---|---|
| Read a LAMMPS dump | `mdapy.io.read(path)` | `import_file(path).compute()` | `ase.io.read(path, format="lammps-dump-text")` |
| Write a LAMMPS dump | `mdapy.io.write(path, system)` | `export_file(data, path, "lammps/dump", columns=[...], precision=17)` | ASE writes no dumps |
| Read a LAMMPS data file | `mdapy.io.read(path)` | `import_file(path, atom_style="atomic").compute()` | `ase.io.read(path, format="lammps-data", atom_style="atomic", units="metal")` |
| Write a LAMMPS data file | `mdapy.io.write(path, system)` | `export_file(data, path, "lammps/data", atom_style="atomic", precision=17)` | `ase.io.write(path, atoms, format="lammps-data", atom_style="atomic", masses=True)` |
| Read extended XYZ | `mdapy.io.read(path)` | `import_file(path).compute()` | `ase.io.read(path, format="extxyz")` |
| Write extended XYZ | `mdapy.io.write(path, system)` | `export_file(data, path, "xyz", columns=[...], precision=17)` | `ase.io.write(path, atoms, format="extxyz")` |

The system is fcc copper with its atoms displaced at random by 0.1 Å along
each axis. The dump holds the columns `id type x y z`; the data file the
atom style atomic, with the mass of copper; extended XYZ the species and the
positions.

MDAPY writes every number with as many digits as it takes to read it back
exactly, at most 17. OVITO is asked for 17 significant digits. ASE writes
the positions of a data file in full and those of extended XYZ with 8
decimals, so its extended XYZ is about a third smaller.

## How it is measured

- Each program runs in a process of its own. MDAPY and OVITO have 10
  threads (`MDAPY_NUM_THREADS`, `OVITO_THREAD_COUNT`); ASE reads and writes
  in Python on one.
- Before the timings, a separate process checks each program. Reading: the
  positions it reads are those MDAPY reads within 1e-12 Å, and the types and
  elements are the same. Writing: MDAPY reads back the file the program
  wrote, and finds the positions within 1e-12 Å (1e-8 Å for the extended
  XYZ of ASE) and the same types and elements. A program that differs is
  not timed.
- The file to read is written by MDAPY before the timings, so it is in the
  file cache of the operating system: the times are those of reading and
  parsing, not of the disk. Writing goes to the disk of the machine as the
  operating system takes it.
- Each process first reads or writes a system of 500 atoms, so that loading
  libraries and starting threads are not timed, then three times the
  system; the shortest time counts.
- Making the system, and converting it into OVITO's data collection or
  ASE's atoms for writing, are not timed.

## Running it

The comparison is `benchmarks/comparison.py` in the source repository, the
group `files` of its cases. It needs OVITO (`pip install ovito`) and ASE
(`pip install ase`).

```bash
python benchmarks/comparison.py run --group files --sizes 4000 32000 256000 2048000 \
    --threads 10 --memory 8 --timeout 1800 --output files.json   # as above
python benchmarks/comparison.py table files.json
python benchmarks/comparison.py plot files.json files.png
```
