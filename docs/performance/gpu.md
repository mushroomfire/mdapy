# Molecular dynamics on the GPU

MDAPY and LAMMPS run the same molecular dynamics at constant energy on one NVIDIA GPU:
copper with the EAM potential of Zhou et al. and argon with Lennard-Jones, from 16 000
to 2 million atoms, MDAPY with its [potentials on the GPU](../user-guide/simulation/gpu.md)
and LAMMPS with its KOKKOS package, in mixed and in double precision, and with its GPU
package in mixed precision.

## Results

A workstation: an NVIDIA RTX A2000 12GB (Ampere, driver 580.196), an Intel Core
i5-12600K, Ubuntu 24.04. CUDA 13.5 and GCC 13.3; MDAPY 2.2.0 in development, built with
`-Ccmake.define.MDAPY_CUDA=ON`; LAMMPS stable_30Sep2026 with KOKKOS for this GPU
(`Kokkos_ARCH_AMPERE86`), once with `KOKKOS_PREC=mixed` and once in double precision,
and with the GPU package (`GPU_PREC=mixed`, `GPU_ARCH=sm_86`).

![Atom-steps per second of MDAPY and LAMMPS on one GPU against the number of atoms](images/gpu.png)

*How many atom-steps each runs in a second: higher is faster. Solid lines in mixed
precision, dashed in double; for KOKKOS the faster of its two neighbor lists at each size.*

### What they show

- **Mixed precision**: MDAPY is the fastest at every size, 1.1 to 1.6 times faster than
  KOKKOS for copper and 1.3 to 1.4 times for argon, and 1.5 to 2.8 times faster than
  the GPU package, which moves the atoms on the processor and copies them to the GPU
  at every step.
- **Double precision**: MDAPY is about as fast as KOKKOS for the smaller systems and
  1.15 to 1.35 times slower for the largest. This GPU computes in double precision a
  small part as fast as in single.

### Times

The time of a step in ms; for KOKKOS the faster of its two neighbor lists.

#### Copper, EAM of Zhou et al.

| Atoms | MDAPY, mixed | KOKKOS, mixed | GPU package, mixed | MDAPY, double | KOKKOS, double |
|---:|---:|---:|---:|---:|---:|
| 32,000 | 0.63 | 1.03 | 1.34 | 3.78 | 4.40 |
| 256,000 | 4.28 | 6.17 | 10.22 | 23.24 | 21.03 |
| 864,000 | 16.73 | 18.86 | 35.56 | 92.70 | 68.30 |
| 2,048,000 | 41.49 | 45.01 | 84.12 | 234.88 | 173.87 |

#### Argon, Lennard-Jones cut at 8.5 Å

| Atoms | MDAPY, mixed | KOKKOS, mixed | GPU package, mixed | MDAPY, double | KOKKOS, double |
|---:|---:|---:|---:|---:|---:|
| 16,384 | 0.30 | 0.39 | 0.44 | 0.78 | 0.78 |
| 131,072 | 1.85 | 2.44 | 3.53 | 4.74 | 4.41 |
| 442,368 | 5.22 | 6.96 | 13.06 | 13.12 | 11.85 |
| 1,048,576 | 11.40 | 15.48 | 31.41 | 30.74 | 26.72 |

## How it was measured

- Each run builds the fcc crystal, gives it the velocities of 600 K (copper, time step
  2 fs) or 60 K (argon, 5 fs), runs 50 steps at constant energy and times the next
  300, with a skin of 1 Å and the neighbor list checked at every step.
- LAMMPS reports the time of its loop. The time of MDAPY is that of `run`, which also
  computes the forces and records the state at its start and its end.
- KOKKOS ran with full neighbor lists and Newton's third law off, its default on GPUs,
  and with half lists and the law on; the faster counts. The half lists were faster but
  for the smallest systems.
- This GPU slows down as it warms: each run started after 45 s of rest.

## Running it

The comparison is `benchmarks/gpu.py` in the source repository. It runs the LAMMPS
executables that the environment variables name and leaves out those not given.

```bash
LMP_KOKKOS_MIXED=/path/to/kokkos-mixed/lmp LMP_KOKKOS_DOUBLE=/path/to/kokkos/lmp \
    LMP_GPU=/path/to/gpu/lmp python benchmarks/gpu.py run --rest 45 --output gpu.json
python benchmarks/gpu.py table gpu.json
python benchmarks/gpu.py plot gpu.json gpu.png
```
