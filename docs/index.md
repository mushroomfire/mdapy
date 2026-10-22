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

## What mdapy does

Files become a `System`; its neighbors feed the analyses and the
calculators; calculators drive minimization and the elastic constants.
Dashed parts are planned.

```mermaid
flowchart LR
    files["Files<br>extended XYZ · LAMMPS · POSCAR · mdapy"] <--> system["System<br>box · topology · trajectory"]
    build["Build and transform<br>crystals · polycrystals · replicate"] -.-> system
    system --> neighbors["Neighbor lists"]
    neighbors -.-> analysis["Analysis<br>structure · order · defects · dynamics"]
    neighbors --> calculators["Calculators<br>EAM · NEP · LAMMPS"]
    calculators --> simulation["Simulation<br>energy minimization"]
    calculators --> elasticity["Elasticity and phonons<br>elastic constants"]
    classDef planned stroke-dasharray: 5 5
    class build,analysis planned
```

<div class="grid cards" markdown>

-   **Getting started**

    ---

    [Installation](getting-started/installation.md) ·
    [Quick tour](getting-started/quick-tour.md)

    Basics: [Systems and topology](getting-started/system.md) ·
    [Simulation box](getting-started/box.md) ·
    [Reading and writing files](getting-started/io.md) ·
    [Trajectories](getting-started/trajectory.md) ·
    [Neighbor lists](getting-started/neighbor.md) ·
    [Parallelism](getting-started/parallelism.md)

-   **Analysis** (planned)

    ---

    Structure identification · Local order and distributions · Defects and
    deformation · Dynamics

-   **Modelling**

    ---

    Build and transform (planned)

    Calculators: [the interface](modelling/calculators.md) ·
    [EAM](modelling/eam.md) · [NEP](modelling/nep.md) ·
    [LAMMPS](modelling/lammps.md)

    Simulation: [Energy minimization](modelling/minimize.md)

    Elasticity and phonons: [Elastic constants](modelling/elastic-constants.md)

-   **Reference**

    ---

    File formats: [Extended XYZ](reference/formats/extxyz.md) ·
    [LAMMPS data](reference/formats/lammps-data.md) ·
    [LAMMPS dump](reference/formats/lammps-dump.md) ·
    [VASP POSCAR](reference/formats/poscar.md) ·
    [mdapy files](reference/formats/mdapy.md)

    [Units and conventions](reference/units.md) ·
    [Elements](reference/elements.md) · [Glossary](reference/glossary.md) ·
    [References](reference/references.md) · [Python API](reference/api.md)

</div>

See the [release notes](development/release-notes.md) for what each
development version contains.

## Design in brief

- **Two dependencies.** NumPy and polars. Everything else is optional.
- **One unit system.** Å, eV, ps, amu, K, and GPa for stress. See
  [Units and conventions](reference/units.md).
- **Consistent names.** Every parameter and property follows the
  [Glossary](reference/glossary.md).
- **No OpenMP.** Compiled kernels use mdapy's own thread pool, so mdapy runs
  safely in the same process as PyTorch, NumPy and LAMMPS. Results do not
  depend on the number of threads. See [Parallelism](getting-started/parallelism.md).
- **Tested and documented feature by feature.** Each feature is tested
  against independent references and has its own page describing how to use
  it and, for analyses, what it computes and which papers it follows.

## Citing mdapy

If mdapy contributes to a publication, please cite
[Wu and Shao (2023)](reference/references.md#wu2023), and the papers that
the pages of the features you use ask to cite.
