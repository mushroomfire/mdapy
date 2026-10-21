# mdapy

mdapy is a Python package for analysing and modelling atomistic simulations:
reading and writing structures and trajectories, neighbor search, structure
identification, property analysis, structure building, and energy
minimization with interatomic potentials.

!!! warning "mdapy 2.0 is under development"
    This site documents mdapy 2.0, a complete rewrite on the `v2` branch that is
    not compatible with mdapy 1.x. Features are added one at a time, each with
    its own tests and documentation page. For the stable release use
    [mdapy 1.0.8](https://pypi.org/project/mdapy/1.0.8/).

## Design in brief

- **Two dependencies.** NumPy 2 and polars 2. Everything else is optional.
- **One unit system.** Å, eV, ps, amu, K, and GPa for stress. See
  [Units and conventions](reference/units.md).
- **Consistent names.** Every parameter and property follows the
  [Glossary](reference/glossary.md).
- **No OpenMP.** Compiled kernels use mdapy's own thread pool, so mdapy runs
  safely in the same process as PyTorch, NumPy and LAMMPS. Results do not
  depend on the number of threads. See [Parallelism](guide/parallelism.md).
- **Tested and documented feature by feature.** Each feature has its own
  page describing what it computes, how it is validated, and which papers it
  follows.

## Status

| Area | State |
|---|---|
| Thread control, units, errors | available |
| Structures (`System`, `Box`), file I/O, neighbor lists | next |
| Calculators (NEP, EAM), energy minimization | planned |
| Structure identification and analysis | planned |

See the [release notes](release-notes.md) for what each development version
contains.
