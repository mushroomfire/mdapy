# Molecular dynamics on many processes

MDAPY and LAMMPS run the same molecular dynamics at constant energy of copper with the
EAM of Zhou et al., split among 1 to 40 processes of one thread each with MPI, for
256 000 and 2 048 000 atoms: MDAPY's own dynamics [on many
processes](../user-guide/simulation/processes.md), and LAMMPS through MDAPY's
[LAMMPS calculator](../user-guide/potentials/lammps.md). Both read the same
`eam/alloy` file, written by `EAM.zhou(["Cu"]).write(path)`; each run takes 100 steps
of 2 fs from 300 K after 20 steps, and the time of a step is the time of the 100 over
100.

## Results

A node of the Triton cluster of Aalto University: two Intel Xeon Gold 6148, 40 cores,
OpenMPI 4.1.6 and GCC 12.3.0, the processes started by `srun --mpi=pmix`. MDAPY 2.3 in
development, built with `pip install .`; LAMMPS stable_30Sep2026 with MPI and the
MANYBODY package, with its defaults: `newton on`, a skin of 2 Å. MDAPY's skin is 1 Å.

![Atom-steps per second of MDAPY and LAMMPS on 1 to 40 processes](images/mpi.png)

*How many atom-steps each runs in a second: higher is faster.*

### What they show

- On 4 to 40 processes MDAPY is as fast as LAMMPS within 2 to 11 % for 2 048 000 atoms,
  and within 8 to 25 % for 256 000, whose 6 400 atoms a process on 40 leave the
  communication and the steps around the forces a larger part.
- On one process MDAPY computes each pair of atoms from both its atoms, so that a run
  repeats to the last bit for any number of threads, and is 1.7 times as slow on one
  thread. On many processes of one thread, each process computes each pair once, as
  LAMMPS does.

### Times

The time of a step in ms.

| Atoms | Processes | MDAPY | LAMMPS | MDAPY / LAMMPS |
|---|---:|---:|---:|---:|
| 256 000 | 1 | 777.8 | 465.8 | 1.67 |
| 256 000 | 4 | 130.2 | 121.1 | 1.08 |
| 256 000 | 16 | 39.2 | 33.9 | 1.15 |
| 256 000 | 40 | 20.3 | 16.2 | 1.25 |
| 2 048 000 | 1 | 6341.7 | 3730.0 | 1.70 |
| 2 048 000 | 4 | 994.5 | 973.8 | 1.02 |
| 2 048 000 | 16 | 280.6 | 272.0 | 1.03 |
| 2 048 000 | 40 | 145.2 | 130.3 | 1.11 |
