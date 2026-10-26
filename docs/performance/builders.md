# Building structures

mdapy and [Atomsk](https://atomsk.univ-lille.fr) build fcc copper as a
single crystal and as a polycrystal of 20 grains, and write it as a LAMMPS
data file, from 4 000 to 2 million atoms. A program is timed only where it
builds the structure mdapy builds.

## Results

A node of the Triton cluster of Aalto University, alone for the run: two
Intel Xeon Gold 6148 (40 cores, two NUMA nodes), 191 GB, Linux. Python 3.13,
mdapy 2.0.0.dev0 built with GCC 12.3, and Atomsk Beta 0.13.1, a statically linked build of its release.

![Time of Atomsk over the time of mdapy for each case and size](images/builders.png)

### What they show

- **Single crystal**: mdapy builds and writes it 4.5 to 17 times faster
  than Atomsk, 1.2 s against 5.5 s at 2 million atoms.
- **Polycrystal**: mdapy is 2.4 to 6.4 times faster, 4.4 s against 10.4 s at
  2 million atoms.
- Writing the data file takes mdapy 0.18 s of these times at 2 million
  atoms.

### Times and memory

Every case at every size: the shortest of three times, and the memory each
build took, the peak resident memory of the process minus its peak before
the timings; that of Atomsk, a process of its own, is not counted.

#### Atomsk

| Case | Atoms | mdapy 2.0.0.dev0 (s) | Atomsk Beta 0.13.1 (s) | Atomsk / mdapy | mdapy (MB) | Atomsk (MB) |
|---|---:|---:|---:|---:|---:|---:|
| fcc single crystal, written as a LAMMPS data file | 4,000 | 0.004 | 0.069 | 17.17 | 0 | – |
| fcc single crystal, written as a LAMMPS data file | 32,000 | 0.018 | 0.142 | 8.02 | 1 | – |
| fcc single crystal, written as a LAMMPS data file | 256,000 | 0.142 | 0.705 | 4.97 | 29 | – |
| fcc single crystal, written as a LAMMPS data file | 2,048,000 | 1.234 | 5.492 | 4.45 | 90 | – |
| Polycrystal of 20 grains, written as a LAMMPS data file | 3,981 | 0.033 | 0.212 | 6.44 | 0 | – |
| Polycrystal of 20 grains, written as a LAMMPS data file | 31,995 | 0.095 | 0.372 | 3.93 | 4 | – |
| Polycrystal of 20 grains, written as a LAMMPS data file | 255,941 | 0.506 | 1.442 | 2.85 | 58 | – |
| Polycrystal of 20 grains, written as a LAMMPS data file | 2,047,788 | 4.387 | 10.387 | 2.37 | 117 | – |

## Cases

| Case | mdapy | Atomsk |
|---|---|---|
| Single crystal | `mp.write(path, build_crystal("fcc", 3.615, elements="Cu", repeat=(n, n, n)))` | `atomsk --create fcc 3.615 Cu -duplicate n n n path.lmp` |
| Polycrystal | `mp.write(path, build_polycrystal(unit, box, seeds, orientations=rotations, min_distance=0.01))` | `atomsk --polycrystal unit.xsf parameters.txt path.lmp` |

Atomsk writes every structure it builds to a file, so both programs are
timed building the structure and writing it as a LAMMPS data file.

The polycrystal fills a cubic periodic box with 20 grains, the Voronoi cells
of 20 seeds at random places, each of fcc copper in a random orientation.
Both programs have the same seeds and orientations: Atomsk as a node line
with three angles of rotation, about x, then y, then z, and mdapy as the
rotation matrices of those angles. Atomsk removes no atom where grains meet,
so mdapy is asked to remove only atoms at the same place (`min_distance`
0.01 Å) rather than those closer than 0.7 times the nearest distance, as it
does by default.

## How it is measured

- mdapy runs in a process of its own with 40 threads. Atomsk runs as a
  command from that process, with `OMP_NUM_THREADS` 40; the time of a build
  by Atomsk includes starting the program.
- Before the timings, a separate process builds the structure with both
  programs and reads both files with mdapy. Single crystal: the same box and
  the same atoms, in any order, within 1e-9 Å (Atomsk writes 12 decimals).
  Polycrystal: the same box, as many atoms within 0.5 %, and in each grain
  the same lattice. The atoms differ: each program places the lattice of a
  grain in its own way, mdapy with a lattice site at the seed, Atomsk with
  the center of a block of lattice at a center of the cell, so that in a
  grain the atoms of Atomsk are those of mdapy moved by one vector, a
  different one for each grain. The check finds that vector the same for
  every atom of a grain away from its faces, within 1e-6 Å.
- Each process first builds a structure of 500 atoms, then three times the
  structure; the shortest time counts.

## Running it

The comparison is `benchmarks/comparison.py` in the source repository, the
group `builders` of its cases. It runs the command `atomsk`, or the program
the environment variable `ATOMSK` names.

```bash
ATOMSK=/path/to/atomsk python benchmarks/comparison.py run --group builders \
    --sizes 4000 32000 256000 2048000 --threads 40 --memory 170 --timeout 900 \
    --output builders.json   # as above
python benchmarks/comparison.py table builders.json
python benchmarks/comparison.py plot builders.json builders.png
```
