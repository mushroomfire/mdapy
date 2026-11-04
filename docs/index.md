---
title: MDAPY
---

# ![MDAPY](assets/mdapy-logo.png#only-light){ width="420" } ![MDAPY](assets/mdapy-logo-dark.png#only-dark){ width="420" } { .logo }

**MDAPY**, short for *Molecular Dynamics Analysis in Python*, builds, analyzes,
simulates and visualizes atomistic systems.

!!! note "MDAPY 2.0 is a complete rewrite"
    It is not compatible with MDAPY 1.x. It brings many more features, and we recommend
    upgrading. The code of 1.x stays on the branch [`1.x`](https://github.com/mushroomfire/mdapy/tree/1.x).

If you like MDAPY, please give it a ⭐ [star](https://github.com/mushroomfire/mdapy).

## 💡 Why MDAPY

- **Light.** [NumPy](https://numpy.org) and [polars](https://pola.rs) are its only
  requirements. Packages such as [matplotlib](https://matplotlib.org) or
  [ASE](https://wiki.fysik.dtu.dk/ase/) are optional, and it carries no other
  third-party code.
- **Robust.** Tests cover more than 99 % of the code and compare each feature with analytic
  results and with mature software. Results are the same for any number of threads, and damaged files or
  impossible inputs give a clear error, not a crash.
- **Fast.** The heavy parts run in C++, on a parallel engine of MDAPY's own. See the
  [benchmarks](performance/comparison.md).
- **Predictable.** Names say what the code does: `minimize`, `elastic_constants`,
  `CommonNeighborAnalysis`. A `System` never changes in place, every analysis has the form
  `compute(system)` and returns NumPy arrays, each concept has one name
  ([Glossary](reference/glossary.md)), and one unit system, Å, eV, ps, amu, K, and GPa for
  stress, runs through everything ([Units and conventions](reference/units.md)).
- **Integrated.** One package and one `System`, from the first structure to the last
  picture. It converts systems to and from [ASE](https://wiki.fysik.dtu.dk/ase/),
  [OVITO](https://www.ovito.org), [pymatgen](https://pymatgen.org) and
  [freud](https://freud.readthedocs.io), reads and writes the files of
  [LAMMPS](https://www.lammps.org), [VASP](https://www.vasp.at) and
  [GPUMD](https://gpumd.org), takes any ASE calculator or LAMMPS interaction as a
  calculator ([Interoperability](user-guide/interop/ase.md)), shares a process with
  [PyTorch](https://pytorch.org) ([Parallelism](basics/parallelism.md)), and
  works as one step of
  [high-throughput calculations](examples/high-throughput.md).
- **Documented.** Every feature has its own page on how to use it and, for analyses, what
  it computes and which papers it follows, and
  [lessons](getting-started/learn/table.md) lead from the first structure on.

## 🧰 What it does

Structures come from files or are built, live in a `System`, and its neighbors feed the
analyses and the calculators. Dashed parts are planned.

```mermaid
flowchart TB
    files[("<b>Files</b><br/>extended XYZ · LAMMPS · POSCAR<br/>OUTCAR · CIF · PDB · MDAPY")]
    build["<b>Structure builder</b><br/>crystals · alloys · defects<br/>dislocations · polycrystals"]
    system(["<b>System</b><br/>box · topology · trajectory"])
    neighbors(["<b>Neighbor lists</b>"])
    visual["<b>Visualization</b><br/>rendering · movies · plots"]
    mlip["<b>MLIP</b><br/>VASP training sets · descriptors<br/>choosing frames · errors · uncertainty"]
    analysis["<b>Analysis</b><br/>structures · local order · defects · dynamics"]
    calculators["<b>Calculators</b><br/>native EAM · NEP<br/>interfaces to LAMMPS · ASE · VASP"]
    minimization["<b>Relaxation and dynamics</b><br/>minimization · molecular dynamics<br/>defect energies"]
    lattice["<b>Lattice dynamics</b><br/>elastic constants · phonons<br/>sheets · finite temperature"]
    thermodynamics["<b>Thermodynamics</b><br/>free energy · melting points<br/>stacking faults with temperature"]
    planned["phase diagrams"]
    packages["<b>Interoperability</b><br/>ASE · OVITO · pymatgen · freud<br/>LAMMPS · VASP · GPUMD"]

    files <--> system
    build --> system
    packages <--> system
    packages <--> calculators
    system --> neighbors
    system --> visual
    files --> mlip
    calculators --> mlip
    neighbors --> analysis
    neighbors --> calculators
    calculators --> minimization
    minimization --> lattice
    minimization --> thermodynamics
    thermodynamics -.-> planned

    classDef core fill:#2563eb40,stroke:#2563eb,stroke-width:4px
    classDef data fill:#3b82f61f,stroke:#3b82f6,stroke-width:2px
    classDef analysis fill:#ea580c26,stroke:#ea580c,stroke-width:2px
    classDef simulation fill:#9333ea26,stroke:#9333ea,stroke-width:2px
    classDef planned fill:none,stroke:#9333ea,stroke-width:2px,stroke-dasharray:5 5
    classDef build fill:#16a34a26,stroke:#16a34a,stroke-width:2px
    classDef visual fill:#db277726,stroke:#db2777,stroke-width:2px
    classDef mlip fill:#0d948826,stroke:#0d9488,stroke-width:2px
    class system,neighbors core
    class files,packages data
    class analysis analysis
    class calculators,minimization,lattice,thermodynamics simulation
    class planned planned
    class build build
    class visual visual
    class mlip mlip
```

## 📦 Install

```bash
pip install mdapy
```

[Installation](getting-started/installation.md) describes the other ways to install it, what
those need, and the optional packages.

## 🧪 Tests

More than 99 % of the code is tested. [Contributing](about/contributing.md) says how to
run the tests and what a complete run needs.

## 📖 Citing MDAPY

If MDAPY contributes to a publication, please cite
[Wu and Shao (2023)](reference/references.md#wu2023), and the papers that
the pages of the features you use ask to cite.

See the [release notes](about/release-notes.md) for what each
development version contains.
