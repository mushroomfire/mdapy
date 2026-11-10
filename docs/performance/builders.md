# Building structures

MDAPY and [Atomsk](https://atomsk.univ-lille.fr) build fcc copper as a
single crystal and as a polycrystal of 20 grains, and write it as a LAMMPS
data file, from 4 000 to nearly 10 million atoms. A program is timed only where it
builds the structure MDAPY builds.

## Results

A node of the Triton cluster of Aalto University, alone for the run: two
Intel Xeon Gold 6148 (40 cores, two NUMA nodes), 191 GB, Linux. Python 3.13,
MDAPY 2.0.0b1 built with GCC 12.3, and Atomsk Beta 0.13.1, a
statically linked build of its release. The [analyses](comparison.md) ran on
another node, with Intel Xeon Gold 6230, so their times do not compare with these.

![Time of Atomsk over the time of MDAPY for each case and size](images/builders.png)

*How many times faster MDAPY is: a row for each case, the largest gain on
top, and a point for each size, larger and darker for more atoms; the number
on the right is the speed-up at the largest size.*

### What they show

- **Single crystal**: MDAPY builds and writes it 30 to 75 times faster
  than Atomsk, 0.34 s against 25.5 s at nearly 10 million atoms; building it is
  a repetition of the cell, and most of the time is writing the file.
- **Polycrystal**: MDAPY is 44 to 101 times faster, 0.48 s against 48.3 s at
  nearly 10 million atoms; most of the time is writing the file here too.

### Times and memory

Every case at every size: the shortest of three times, and the memory each
build took, the peak resident memory of the process minus its peak before
the timings; that of Atomsk, a process of its own, is not counted.

#### Atomsk

| Case | Atoms | MDAPY 2.0.0b1 (s) | Atomsk Beta 0.13.1 (s) | Atomsk / MDAPY | MDAPY (MB) | Atomsk (MB) |
|---|---:|---:|---:|---:|---:|---:|
| fcc single crystal, written as a LAMMPS data file | 4,000 | 0.0014 | 0.044 | 30.3 | 0 | 0 |
| fcc single crystal, written as a LAMMPS data file | 32,000 | 0.0031 | 0.151 | 48.4 | 3 | 0 |
| fcc single crystal, written as a LAMMPS data file | 256,000 | 0.010 | 0.710 | 68.0 | 40 | 0 |
| fcc single crystal, written as a LAMMPS data file | 2,048,000 | 0.077 | 5.317 | 68.9 | 143 | 0 |
| fcc single crystal, written as a LAMMPS data file | 9,841,500 | 0.343 | 25.539 | 74.5 | 413 | 0 |
| Polycrystal of 20 grains, written as a LAMMPS data file | 3,981 | 0.0059 | 0.585 | 100.0 | 0 | 0 |
| Polycrystal of 20 grains, written as a LAMMPS data file | 31,995 | 0.010 | 0.450 | 44.3 | 2 | 0 |
| Polycrystal of 20 grains, written as a LAMMPS data file | 255,941 | 0.017 | 1.540 | 88.8 | 39 | 0 |
| Polycrystal of 20 grains, written as a LAMMPS data file | 2,047,788 | 0.109 | 10.208 | 93.5 | 64 | 0 |
| Polycrystal of 20 grains, written as a LAMMPS data file | 9,841,309 | 0.478 | 48.292 | 100.9 | 311 | 0 |

## Cases

| Case | MDAPY | Atomsk |
|---|---|---|
| Single crystal | `mdapy.io.write(path, build_crystal("fcc", 3.615, elements="Cu", repeat=(n, n, n)))` | `atomsk --create fcc 3.615 Cu -duplicate n n n path.lmp` |
| Polycrystal | `mdapy.io.write(path, build_polycrystal(unit, box, seeds, orientations=rotations, min_distance=0.01))` | `atomsk --polycrystal unit.xsf parameters.txt path.lmp` |

Atomsk writes every structure it builds to a file, so both programs are
timed building the structure and writing it as a LAMMPS data file.

The polycrystal fills a cubic periodic box with 20 grains, the Voronoi cells
of 20 seeds at random places, each of fcc copper in a random orientation.
Both programs have the same seeds and orientations: Atomsk as a node line
with three angles of rotation, about x, then y, then z, and MDAPY as the
rotation matrices of those angles. Atomsk removes no atom where grains meet,
so MDAPY is asked to remove only atoms at the same place (`min_distance`
0.01 Å) rather than those closer than 0.7 times the nearest distance, as it
does by default.

## How it is measured

- MDAPY runs in a process of its own with 40 threads. Atomsk runs as a
  command from that process, with `OMP_NUM_THREADS` 40; the time of a build
  by Atomsk includes starting the program.
- Before the timings, a separate process builds the structure with both
  programs and reads both files with MDAPY. Single crystal: the same box and
  the same atoms, in any order, within 1e-9 Å (Atomsk writes 12 decimals).
  Polycrystal: the same box, as many atoms within 0.5 %, and in each grain
  the same lattice. The atoms differ: each program places the lattice of a
  grain in its own way, MDAPY with a lattice site at the seed, Atomsk with
  the center of a block of lattice at a center of the cell, so that in a
  grain the atoms of Atomsk are those of MDAPY moved by one vector, a
  different one for each grain. The check finds that vector the same for
  every atom of a grain away from its faces, within 1e-6 Å.
- The times of MDAPY were measured after those of Atomsk, on another node of
  the same kind.
- Each process first builds a structure of 500 atoms, then three times the
  structure; the shortest time counts.

## Running it

The comparison is `benchmarks/comparison.py` in the source repository, the
group `builders` of its cases. It runs the command `atomsk`, or the program
the environment variable `ATOMSK` names.

```bash
ATOMSK=/path/to/atomsk python benchmarks/comparison.py run --group builders \
    --sizes 4000 32000 256000 2048000 9841500 --threads 40 --memory 170 --timeout 1800 \
    --output builders.json   # as above
python benchmarks/comparison.py table builders.json
python benchmarks/comparison.py plot builders.json builders.png
```
