---
title: MDAPY
---

# ![MDAPY](assets/mdapy-logo.png#only-light){ width="420" } ![MDAPY](assets/mdapy-logo-dark.png#only-dark){ width="420" } { .logo }

**MDAPY**, short for *Molecular Dynamics Analysis in Python*, builds, analyzes,
simulates and visualizes atomistic systems. It reads and writes structures and trajectories in
the common formats, finds neighbors, identifies crystal structures and
defects and measures their properties, builds crystals and polycrystals, and
computes energies, forces, relaxed structures and elastic constants with
interatomic potentials. The heavy work runs in compiled C++ on all cores of
the machine.

!!! note "MDAPY 2.0 is a complete rewrite"
    It is not compatible with MDAPY 1.x. It brings many more features, and we recommend
    upgrading. The code of 1.x stays on the branch [`1.x`](https://github.com/mushroomfire/mdapy/tree/1.x).

## What MDAPY does

Structures come from files or are built, live in a `System`, and its
neighbors feed the analyses and the calculators. Dashed parts are planned.

```mermaid
flowchart TB
    files[("<b>Files</b><br/>extended XYZ · LAMMPS · POSCAR · MDAPY")]
    build["<b>Structure builder</b><br/>crystals · alloys · defects<br/>polycrystals · packing"]
    system(["<b>System</b><br/>box · topology · trajectory"])
    neighbors(["<b>Neighbor lists</b>"])
    visual["<b>Visualization</b><br/>rendering · plots"]
    mlip["<b>MLIP</b><br/>VASP training sets · descriptors<br/>choosing frames · perturbation"]
    analysis["<b>Analysis</b><br/>structures · local order · defects · dynamics"]
    calculators["<b>Calculators</b><br/>EAM · NEP · LAMMPS"]
    minimization["<b>Minimization</b><br/>and dynamics"]
    lattice["<b>Lattice dynamics</b><br/>elastic constants<br/>phonons"]
    thermodynamics["<b>Thermodynamics</b><br/>free energy"]
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
    calculators --> lattice
    minimization --> thermodynamics

    classDef core fill:#2563eb40,stroke:#2563eb,stroke-width:4px
    classDef data fill:#3b82f61f,stroke:#3b82f6,stroke-width:2px
    classDef analysis fill:#ea580c26,stroke:#ea580c,stroke-width:2px
    classDef simulation fill:#9333ea26,stroke:#9333ea,stroke-width:2px
    classDef build fill:#16a34a26,stroke:#16a34a,stroke-width:2px
    classDef visual fill:#db277726,stroke:#db2777,stroke-width:2px
    classDef mlip fill:#0d948826,stroke:#0d9488,stroke-width:2px
    class system,neighbors core
    class files,packages data
    class analysis analysis
    class calculators,minimization,lattice,thermodynamics simulation
    class build build
    class visual visual
    class mlip mlip
```

## Design

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
  calculator ([Interoperability](interop/ase.md)), shares a process with
  [PyTorch](https://pytorch.org) ([Parallelism](getting-started/parallelism.md)), and
  works as one step of
  [high-throughput calculations](getting-started/tutorials/high-throughput.md).
- **Documented.** Every feature has its own page on how to use it and, for analyses, what
  it computes and which papers it follows, and
  [lessons](getting-started/learn/table.md) lead from the first structure on.

## Features

<div class="grid cards" markdown>

-   :material-rocket-launch-outline:{ .lg .middle } **Getting started**

    ---

    Install MDAPY, take the quick tour, learn the basics and work through
    the tutorials.

    [Installation](getting-started/installation.md) ·
    [Quick tour](getting-started/quick-tour.md)

    Learn MDAPY in six short lessons: [a crystal is a table](getting-started/learn/table.md) ·
    [asking the atoms questions](getting-started/learn/analyses.md) ·
    [from numbers to figures](getting-started/learn/figures.md) ·
    [energies and relaxation](getting-started/learn/energy.md) ·
    [a nanoparticle melts](getting-started/learn/trajectories.md) ·
    [your own files](getting-started/learn/your-files.md) ·
    [cheat sheet](getting-started/learn/cheat-sheet.md)

    System: [Systems](getting-started/system.md) ·
    [selecting and ordering atoms](getting-started/selecting.md) ·
    [expanding a selection](getting-started/expand-selection.md) ·
    [overlapping atoms](getting-started/overlapping.md) ·
    [changing a system](getting-started/changing.md) ·
    [topology](getting-started/topology.md) ·
    [simulation box](getting-started/box.md)

    [Reading and writing files](getting-started/io.md) ·
    [Trajectories](getting-started/trajectory.md) ·
    [Neighbor lists](getting-started/neighbor.md) ·
    [Parallelism](getting-started/parallelism.md)

    Tutorials: [fcc and bcc grains](getting-started/tutorials/two-phase-polycrystal.md) ·
    [columnar grains](getting-started/tutorials/columnar-polycrystal.md) ·
    [defect energies of copper](getting-started/tutorials/defect-energies.md) ·
    [screening alloys in many processes](getting-started/tutorials/high-throughput.md) ·
    [a movie that flies through a crystal](getting-started/tutorials/movie.md) ·
    [sharing neighbor lists](getting-started/tutorials/neighbor-lists.md)

-   :material-chart-bell-curve-cumulative:{ .lg .middle } **Analysis**

    ---

    Identify structures and measure order, defects and dynamics.

    [Overview](analysis/overview.md)

    Structure identification: [common neighbor analysis](analysis/cna.md) ·
    [Ackland–Jones](analysis/ackland-jones.md) ·
    [polyhedral template matching](analysis/ptm.md) ·
    [planar faults](analysis/planar-faults.md) ·
    [grain segmentation](analysis/grain-segmentation.md) ·
    [diamond](analysis/diamond.md) · [CHILL+](analysis/chill-plus.md) ·
    [centro-symmetry](analysis/csp.md) ·
    [common neighbor parameter](analysis/cnp.md) ·
    [space group](analysis/space-group.md)

    Local order and distributions: [radial distribution](analysis/rdf.md) ·
    [coordination](analysis/coordination.md) ·
    [structure factor](analysis/structure-factor.md) ·
    [bond angles](analysis/bond-angles.md) ·
    [Warren–Cowley](analysis/warren-cowley.md) ·
    [Steinhardt](analysis/steinhardt.md) ·
    [entropy fingerprint](analysis/entropy.md) ·
    [Voronoi](analysis/voronoi.md) ·
    [spatial binning](analysis/spatial-binning.md)

    Defects and deformation: [clusters](analysis/clusters.md) ·
    [surface mesh](analysis/surface-mesh.md) ·
    [dislocations](analysis/dislocations.md) (experimental) ·
    [chemical species](analysis/species.md) ·
    [Wigner–Seitz](analysis/wigner-seitz.md) ·
    [atomic displacement](analysis/displacement.md) ·
    [atomic strain](analysis/atomic-strain.md)

    Dynamics: [mean squared displacement](analysis/msd.md) ·
    [Lindemann index](analysis/lindemann.md) ·
    [Time averages](analysis/time-average.md) ·
    [Trajectory lines](analysis/trajectory-lines.md) ·
    [atomic temperature](analysis/temperature.md)

-   :material-cube-outline:{ .lg .middle } **Structure builder**

    ---

    Build crystals, alloys, defects and polycrystals; packed molecules are
    planned.

    [Crystals](build/crystals.md) · [random alloys](build/alloys.md) ·
    [SQS](build/sqs.md) · [point defects and faults](build/defects.md) ·
    [surfaces and slip systems](build/slip-systems.md) ·
    [dislocations](build/dislocations.md) ·
    [polycrystals](build/polycrystals.md) ·
    [graphene on grain boundaries](build/graphene.md) ·
    [periodic copies and images](build/transform.md) ·
    [slices](build/slice.md) ·
    [affine transformation](build/affine.md)

-   :mdapy-simulation:{ .lg .middle } **Simulation**

    ---

    Energies and forces from potentials, minimization, lattice dynamics and free
    energies.

    Calculators: [the interface](simulation/calculators.md) ·
    [EAM](simulation/eam.md) · [NEP](simulation/nep.md) ·
    [LAMMPS](simulation/lammps.md) · [VASP](simulation/vasp.md)

    Minimization and dynamics: [energy minimization](simulation/minimize.md) ·
    [molecular dynamics](simulation/dynamics.md) ·
    [defect energies](simulation/defect-energies.md)

    Thermodynamics: [free energy](simulation/free-energy.md) ·
    [stacking faults at finite temperature](simulation/stacking-fault-temperature.md)

    Lattice dynamics: [elastic constants](simulation/elastic-constants.md) ·
    [phonons](simulation/phonons.md) ·
    [at finite temperature](simulation/thermal-phonons.md) ·
    [quasi-harmonic approximation](simulation/quasi-harmonic.md) ·
    [elastic constants at finite temperature](simulation/thermal-elastic.md) ·
    [from molecular dynamics](simulation/md-elastic.md) ·
    [sheets](simulation/sheets.md)

-   :material-sine-wave:{ .lg .middle } **MLIP**

    ---

    Training data for machine-learned potentials.

    [Training sets from VASP](mlip/vasp.md) ·
    [choosing frames](mlip/selection.md) ·
    [perturbed structures](mlip/perturbation.md)

-   :material-palette-outline:{ .lg .middle } **Visualization**

    ---

    Images of systems, colored by any property, and plots of results.

    [Rendering](visualization/rendering.md) ·
    [coordination polyhedra](visualization/polyhedra.md) · [plots](visualization/plots.md)

-   :material-puzzle-outline:{ .lg .middle } **Interoperability**

    ---

    Systems and calculators shared with other packages, and the files of
    simulation programs.

    Packages: [ASE](interop/ase.md) ·
    [ASE calculators](interop/ase-calculators.md) ·
    [OVITO](interop/ovito.md) · [pymatgen](interop/pymatgen.md) ·
    [freud](interop/freud.md)

    Programs: [LAMMPS](interop/lammps.md) · [VASP](interop/vasp.md) ·
    [GPUMD](interop/gpumd.md)

-   :material-speedometer:{ .lg .middle } **Performance**

    ---

    Timings of MDAPY and of other programs, from thousands to millions of
    atoms.

    [Analyses: OVITO and freud](performance/comparison.md) ·
    [reading and writing files: OVITO and ASE](performance/files.md) ·
    [building structures: Atomsk](performance/builders.md)

-   :material-book-open-variant:{ .lg .middle } **Reference**

    ---

    File formats, units, the glossary, the papers and the Python API.

    [CIF](reference/formats/cif.md) ·
    [Extended XYZ](reference/formats/extxyz.md) ·
    [LAMMPS data](reference/formats/lammps-data.md) ·
    [LAMMPS dump](reference/formats/lammps-dump.md) ·
    [PDB](reference/formats/pdb.md) ·
    [VASP OUTCAR](reference/formats/outcar.md) ·
    [VASP POSCAR](reference/formats/poscar.md) ·
    [MDAPY files](reference/formats/mdapy.md)

    [Units and conventions](reference/units.md) ·
    [Elements](reference/elements.md) · [Glossary](reference/glossary.md) ·
    [References](reference/references.md) · [Python API](reference/api.md)

</div>

See the [release notes](development/release-notes.md) for what each
development version contains.

## Citing MDAPY

If MDAPY contributes to a publication, please cite
[Wu and Shao (2023)](reference/references.md#wu2023), and the papers that
the pages of the features you use ask to cite.
