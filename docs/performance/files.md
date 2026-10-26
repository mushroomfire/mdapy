# Reading and writing files

mdapy, [OVITO](https://www.ovito.org) and [ASE](https://wiki.fysik.dtu.dk/ase/)
read and write LAMMPS dumps, LAMMPS data files and extended XYZ of the same
systems, from 4 000 to 2 million atoms. A program is timed only where it
reads the positions that mdapy wrote, or writes a file that mdapy reads back
as the system.

## Results

A node of the Triton cluster of Aalto University, alone for the run: two
Intel Xeon Gold 6148 (40 cores, two NUMA nodes), 191 GB, Linux. Python 3.13,
mdapy 2.0.0.dev0 built with GCC 12.3, OVITO 3.17.0 and ASE 3.29.0 from pip.

![Time of OVITO and of ASE over the time of mdapy for each case and size](images/files.png)

### What they show

- **OVITO**: from 32 000 atoms on, mdapy reads the three formats 2.5 to
  5.4 times faster and writes them 3.9 to 10.7 times faster. At 4 000 atoms
  OVITO reads faster, in 1.1 to 1.4 ms against 1.3 to 2.2 ms, and mdapy
  still writes 1.6 to 3.5 times faster.
- **ASE**: mdapy reads 4 to 55 times faster and writes 7 to 60 times faster.
  ASE reads a data file of 2 million atoms in 7.8 s and 4 GB of memory.
- **2 million atoms** take mdapy 0.09 to 0.17 s to read in each format, and
  0.12 to 0.18 s to write with every digit.

### Times and memory

Every case at every size: the shortest of three times, and the memory each
call took beyond the system, the peak resident memory of the process minus
its peak before the timings.

#### OVITO

| Case | Atoms | mdapy 2.0.0.dev0 (s) | OVITO 3.17.0 (s) | OVITO / mdapy | mdapy (MB) | OVITO (MB) |
|---|---:|---:|---:|---:|---:|---:|
| Read a LAMMPS dump | 4,000 | 0.001 | 0.001 | 0.82 | 0 | 1 |
| Read a LAMMPS dump | 32,000 | 0.003 | 0.007 | 2.83 | 2 | 5 |
| Read a LAMMPS dump | 256,000 | 0.013 | 0.058 | 4.54 | 2 | 8 |
| Read a LAMMPS dump | 2,048,000 | 0.105 | 0.473 | 4.51 | 147 | 101 |
| Write a LAMMPS dump | 4,000 | 0.001 | 0.003 | 3.52 | 0 | 0 |
| Write a LAMMPS dump | 32,000 | 0.003 | 0.020 | 7.53 | 2 | 0 |
| Write a LAMMPS dump | 256,000 | 0.021 | 0.148 | 7.20 | 20 | 0 |
| Write a LAMMPS dump | 2,048,000 | 0.133 | 1.216 | 9.14 | 0 | 0 |
| Read a LAMMPS data file | 4,000 | 0.002 | 0.001 | 0.64 | 0 | 0 |
| Read a LAMMPS data file | 32,000 | 0.004 | 0.010 | 2.51 | 2 | 4 |
| Read a LAMMPS data file | 256,000 | 0.023 | 0.081 | 3.54 | 10 | 4 |
| Read a LAMMPS data file | 2,048,000 | 0.167 | 0.673 | 4.03 | 247 | 77 |
| Write a LAMMPS data file | 4,000 | 0.002 | 0.003 | 1.57 | 0 | 0 |
| Write a LAMMPS data file | 32,000 | 0.005 | 0.020 | 3.87 | 2 | 0 |
| Write a LAMMPS data file | 256,000 | 0.032 | 0.155 | 4.87 | 25 | 0 |
| Write a LAMMPS data file | 2,048,000 | 0.179 | 1.231 | 6.89 | 22 | 0 |
| Read extended XYZ | 4,000 | 0.002 | 0.001 | 0.71 | 0 | 0 |
| Read extended XYZ | 32,000 | 0.003 | 0.007 | 2.45 | 2 | 2 |
| Read extended XYZ | 256,000 | 0.012 | 0.057 | 4.91 | 0 | 0 |
| Read extended XYZ | 2,048,000 | 0.088 | 0.474 | 5.38 | 44 | 0 |
| Write extended XYZ | 4,000 | 0.001 | 0.003 | 2.87 | 1 | 0 |
| Write extended XYZ | 32,000 | 0.003 | 0.020 | 7.51 | 1 | 0 |
| Write extended XYZ | 256,000 | 0.019 | 0.164 | 8.54 | 10 | 0 |
| Write extended XYZ | 2,048,000 | 0.122 | 1.305 | 10.73 | 0 | 0 |

#### ASE

| Case | Atoms | mdapy 2.0.0.dev0 (s) | ASE 3.29.0 (s) | ASE / mdapy | mdapy (MB) | ASE (MB) |
|---|---:|---:|---:|---:|---:|---:|
| Read a LAMMPS dump | 4,000 | 0.001 | 0.005 | 4.18 | 0 | 0 |
| Read a LAMMPS dump | 32,000 | 0.003 | 0.042 | 15.80 | 2 | 4 |
| Read a LAMMPS dump | 256,000 | 0.013 | 0.355 | 27.97 | 2 | 40 |
| Read a LAMMPS dump | 2,048,000 | 0.105 | 2.872 | 27.36 | 147 | 383 |
| Read a LAMMPS data file | 4,000 | 0.002 | 0.014 | 6.34 | 0 | 7 |
| Read a LAMMPS data file | 32,000 | 0.004 | 0.124 | 30.80 | 2 | 60 |
| Read a LAMMPS data file | 256,000 | 0.023 | 1.054 | 45.97 | 10 | 483 |
| Read a LAMMPS data file | 2,048,000 | 0.167 | 7.800 | 46.66 | 247 | 4040 |
| Write a LAMMPS data file | 4,000 | 0.002 | 0.014 | 7.13 | 0 | 0 |
| Write a LAMMPS data file | 32,000 | 0.005 | 0.105 | 20.70 | 2 | 1 |
| Write a LAMMPS data file | 256,000 | 0.032 | 0.859 | 27.02 | 25 | 32 |
| Write a LAMMPS data file | 2,048,000 | 0.179 | 6.927 | 38.78 | 22 | 264 |
| Read extended XYZ | 4,000 | 0.002 | 0.009 | 5.07 | 0 | 1 |
| Read extended XYZ | 32,000 | 0.003 | 0.071 | 23.13 | 2 | 7 |
| Read extended XYZ | 256,000 | 0.012 | 0.597 | 51.37 | 0 | 51 |
| Read extended XYZ | 2,048,000 | 0.088 | 4.871 | 55.27 | 44 | 449 |
| Write extended XYZ | 4,000 | 0.001 | 0.014 | 13.93 | 1 | 0 |
| Write extended XYZ | 32,000 | 0.003 | 0.106 | 39.39 | 1 | 0 |
| Write extended XYZ | 256,000 | 0.019 | 0.886 | 46.24 | 10 | 3 |
| Write extended XYZ | 2,048,000 | 0.122 | 7.266 | 59.74 | 0 | 31 |

## Cases

| Case | mdapy | OVITO | ASE |
|---|---|---|---|
| Read a LAMMPS dump | `mp.read(path)` | `import_file(path).compute()` | `ase.io.read(path, format="lammps-dump-text")` |
| Write a LAMMPS dump | `mp.write(path, system)` | `export_file(data, path, "lammps/dump", columns=[...], precision=17)` | ASE writes no dumps |
| Read a LAMMPS data file | `mp.read(path)` | `import_file(path, atom_style="atomic").compute()` | `ase.io.read(path, format="lammps-data", atom_style="atomic", units="metal")` |
| Write a LAMMPS data file | `mp.write(path, system)` | `export_file(data, path, "lammps/data", atom_style="atomic", precision=17)` | `ase.io.write(path, atoms, format="lammps-data", atom_style="atomic", masses=True)` |
| Read extended XYZ | `mp.read(path)` | `import_file(path).compute()` | `ase.io.read(path, format="extxyz")` |
| Write extended XYZ | `mp.write(path, system)` | `export_file(data, path, "xyz", columns=[...], precision=17)` | `ase.io.write(path, atoms, format="extxyz")` |

The system is fcc copper with its atoms displaced at random by 0.1 Å along
each axis. The dump holds the columns `id type x y z`; the data file the
atom style atomic, with the mass of copper; extended XYZ the species and the
positions.

mdapy writes every number with as many digits as it takes to read it back
exactly, at most 17. OVITO is asked for 17 significant digits. ASE writes
the positions of a data file in full and those of extended XYZ with 8
decimals, so its extended XYZ is about a third smaller.

## How it is measured

- Each program runs in a process of its own. mdapy and OVITO have 40
  threads (`MDAPY_NUM_THREADS`, `OVITO_THREAD_COUNT`); ASE reads and writes
  in Python on one.
- Before the timings, a separate process checks each program. Reading: the
  positions it reads are those mdapy reads within 1e-12 Å, and the types and
  elements are the same. Writing: mdapy reads back the file the program
  wrote, and finds the positions within 1e-12 Å (1e-8 Å for the extended
  XYZ of ASE) and the same types and elements. A program that differs is
  not timed.
- The file to read is written by mdapy before the timings, so it is in the
  file cache of the operating system: the times are those of reading and
  parsing, not of the disk. Writing goes to the local disk of the node as
  the operating system takes it.
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
    --threads 40 --memory 170 --timeout 900 --output files.json   # as above
python benchmarks/comparison.py table files.json
python benchmarks/comparison.py plot files.json files.png
```
