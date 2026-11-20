# ![MDAPY](assets/mdapy-logo.png#only-light){ width="420" } ![MDAPY](assets/mdapy-logo-dark.png#only-dark){ width="420" } { .logo }

**MDAPY**, short for *Molecular Dynamics Analysis in Python*, builds, simulates,
analyzes and visualizes atomistic systems in one Python package. It has its own
molecular dynamics engine, potentials and ray tracer, on the processor and on NVIDIA
GPUs: a study from the first structure to the last picture needs no other program.

!!! note "MDAPY 2.0 is a complete rewrite"
    It is not compatible with MDAPY 1.x. It brings many more features, and we recommend
    upgrading: [From MDAPY 1.x](getting-started/from-1x.md) gives the 2.0 way of each
    common call. The code of 1.x stays on the branch [`1.x`](https://github.com/mushroomfire/mdapy/tree/1.x).

If you like MDAPY, please give it a ⭐ [star](https://github.com/mushroomfire/mdapy).

## 💡 Why MDAPY

- **All in one.** One package and one `System`, from the first structure through the
  simulation and the analysis to the last picture. MDAPY has its own
  [molecular dynamics engine](user-guide/molecular-dynamics/index.md), potentials and
  ray tracer, on the processor and on NVIDIA GPUs, so that a study needs no other
  program.
- **Light.** [NumPy](https://numpy.org) and [polars](https://pola.rs) are its only
  requirements. Packages such as [matplotlib](https://matplotlib.org) or
  [ASE](https://wiki.fysik.dtu.dk/ase/) are optional, and it carries no other
  third-party code.
- **Robust.** Tests cover more than 99 % of the code and compare each feature with analytic
  results and with mature software. All of them run automatically on Linux, macOS and
  Windows. Results are the same for any number of threads, and damaged files or impossible
  inputs give a clear error, not a crash.
- **Fast.** The heavy parts run in C++, on a parallel engine of MDAPY's own, and molecular
  dynamics runs whole on an NVIDIA GPU. See the [benchmarks](performance/comparison.md).
- **Interoperable.** It converts systems to and from
  [ASE](https://wiki.fysik.dtu.dk/ase/), [OVITO](https://www.ovito.org),
  [pymatgen](https://pymatgen.org) and [freud](https://freud.readthedocs.io), reads and
  writes the files of [LAMMPS](https://www.lammps.org), [VASP](https://www.vasp.at) and
  [GPUMD](https://gpumd.org), takes any ASE calculator or LAMMPS interaction as a
  calculator ([Interoperability](user-guide/interop/ase.md)), runs its dynamics
  [in LAMMPS](user-guide/molecular-dynamics/in-lammps.md) when asked, shares a process
  with [PyTorch](https://pytorch.org) ([Parallelism](basics/parallelism.md)), and works as
  one step of [high-throughput calculations](examples/high-throughput.md).
- **Predictable.** Names say what the code does: `minimize`, `elastic_constants`,
  `CommonNeighborAnalysis`. A `System` never changes in place, every analysis has the form
  `compute(system)` and returns NumPy arrays, each concept has one name
  ([Glossary](reference/glossary.md)), and one unit system, Å, eV, ps, amu, K, and GPa for
  stress, runs through everything ([Units and conventions](reference/units.md)).
- **Stable.** From 2.0.0 on, code that works keeps working: the API changes only with
  3.0, a replaced name warns long before it goes, and a result changes only to fix a bug,
  named in the release notes ([Compatibility](about/compatibility.md)).
- **Documented.** Every feature has its own page on how to use it and, for analyses, what
  it computes and which papers it follows, and
  [lessons](getting-started/learn/table.md) lead from the first structure on.

## ⚛️ Molecular dynamics

MDAPY runs molecular dynamics with its own engine, on the processor or entirely on an
NVIDIA [GPU](user-guide/molecular-dynamics/gpu.md); the same calls run
[in LAMMPS](user-guide/molecular-dynamics/in-lammps.md) too. It is made for materials
and machine-learned potentials: metals, alloys, semiconductors, glasses and liquids,
with [EAM](user-guide/potentials/eam.md), [MEAM](user-guide/potentials/meam.md),
[ADP](user-guide/potentials/adp.md), [Tersoff](user-guide/potentials/tersoff.md),
[Stillinger–Weber](user-guide/potentials/stillinger-weber.md) or
[NEP](user-guide/potentials/nep.md), or any model with an
[ASE calculator](user-guide/interop/ase-calculators.md). Molecular force fields, with
bonds and angles, are not its aim. Its features serve mechanical and thermal
simulations:

- **Ensembles**: NVE, NVT, NPT and NPH, with Nosé–Hoover chains or Langevin dynamics
  ([Ensembles](user-guide/molecular-dynamics/ensembles.md)).
- **Controls**: held, moved and pushed groups of atoms, heat baths, adaptive time steps
  and switching between potentials
  ([Groups and controls](user-guide/molecular-dynamics/controls.md)).
- **Mechanical loading**: tension, compression and shear
  ([Deformation](user-guide/molecular-dynamics/deformation.md)), and
  [indentation](user-guide/molecular-dynamics/indentation.md).
- **Shock**: pistons, the Hugoniostat and the multiscale shock technique
  ([Choosing a method](user-guide/molecular-dynamics/shock.md)).
- **Heat**: thermal conductivity by [Green–Kubo](user-guide/properties/green-kubo.md),
  [HNEMD](user-guide/properties/hnemd.md) and [NEMD](user-guide/properties/nemd.md);
  [free energies](user-guide/properties/free-energy.md),
  [melting points](user-guide/properties/melting-point.md) and
  [phase diagrams](user-guide/properties/phase-diagram.md).

```python
from mdapy.build import build_crystal
from mdapy.calculators import EAM
from mdapy.simulation import Deform, MolecularDynamics

crystal = build_crystal("fcc", 3.615, elements="Cu", repeat=(6, 6, 6))
copper = EAM.zhou(["Cu"])  # EAM.zhou(["Cu"], device="cuda") runs on the GPU
with MolecularDynamics(crystal, copper, timestep=0.002) as md:
    md.velocities(300, seed=1)
    md.run(2500, ensemble="npt", temperature=300, pressure=0.0)
    md.run(2500, ensemble="nvt", temperature=300, controls=[Deform(xx=0.02)])
    stress = md.thermo["stress"]  # the stress as copper is stretched along x
```

[Molecular dynamics](user-guide/molecular-dynamics/index.md) shows what it can do and
where each feature runs.

## 🧰 What it does

Structures are read or built into a `System`; its neighbor lists and the potentials
feed the simulations and the analyses, and the results are drawn or fitted. Each box
opens its pages.

<div class="feature-map" markdown>

<div class="feature-stage" markdown>
<p class="feature-label">Input</p>
<div class="feature-cells" markdown>

[<b>Files</b><span>extended XYZ · LAMMPS · POSCAR · OUTCAR · CIF · PDB · MDAPY</span>](basics/io.md){ .feature-card }

[<b>Structure builder</b><span>crystals · alloys · defects · dislocations · polycrystals</span>](user-guide/build/crystals.md){ .feature-card }

[<b>Interoperability</b><span>ASE · OVITO · pymatgen · freud · LAMMPS · VASP · GPUMD</span>](user-guide/interop/ase.md){ .feature-card }

</div>
</div>

<div class="feature-stage feature-core" markdown>
<p class="feature-label">Core</p>
<div class="feature-cells" markdown>

[<b>System</b><span>box · topology · trajectory · any property of the atoms</span>](basics/system.md){ .feature-card }

[<b>Neighbor lists</b><span>built once, shared by analyses and calculators</span>](basics/neighbor.md){ .feature-card }

[<b>Potentials</b><span>native EAM · MEAM · Tersoff · NEP and more; LAMMPS · ASE · VASP</span>](user-guide/potentials/calculators.md){ .feature-card }

</div>
</div>

<div class="feature-stage" markdown>
<p class="feature-label">Simulate</p>
<div class="feature-cells" markdown>

[<b>Molecular dynamics</b><span>its own engine · processor and GPU · ensembles · controls</span>](user-guide/molecular-dynamics/index.md){ .feature-card }

[<b>Loading and shock</b><span>deformation · indentation · pistons · Hugoniostat · MSST</span>](user-guide/molecular-dynamics/deformation.md){ .feature-card }

</div>
</div>

<div class="feature-stage" markdown>
<p class="feature-label">Properties</p>
<div class="feature-cells" markdown>

[<b>Relaxation</b><span>minimization · defect energies</span>](user-guide/properties/minimize.md){ .feature-card }

[<b>Lattice dynamics</b><span>elastic constants · phonons · sheets · finite temperature</span>](user-guide/properties/elastic-constants.md){ .feature-card }

[<b>Thermal conductivity</b><span>phonons · Green–Kubo · HNEMD · NEMD</span>](user-guide/properties/thermal-conductivity.md){ .feature-card }

[<b>Thermodynamics</b><span>free energy · melting points · phase diagrams</span>](user-guide/properties/free-energy.md){ .feature-card }

</div>
</div>

<div class="feature-stage" markdown>
<p class="feature-label">Analyze</p>
<div class="feature-cells" markdown>

[<b>Analysis</b><span>structures · local order · defects · dislocations · dynamics · statistics</span>](user-guide/analysis/overview.md){ .feature-card }

[<b>MLIP</b><span>VASP training sets · descriptors · choosing frames · errors · uncertainty</span>](user-guide/mlip/vasp.md){ .feature-card }

</div>
</div>

<div class="feature-stage" markdown>
<p class="feature-label">Output</p>
<div class="feature-cells" markdown>

[<b>Visualization</b><span>rendering · movies · plots · colors</span>](user-guide/visualization/rendering.md){ .feature-card }

[<b>Fitting</b><span>splines · least squares · equations of state</span>](user-guide/fitting/interpolation.md){ .feature-card }

</div>
</div>

</div>

## 📦 Install

```bash
pip install mdapy
```

Code written for 1.x keeps working with `pip install "mdapy<2"`, or moves to 2.0 [as
this guide shows](getting-started/from-1x.md).

[Installation](getting-started/installation.md) describes the other ways to install it, what
those need, and the optional packages.

## 🧪 Tests

More than 99 % of the code is tested. [Contributing](about/contributing.md) says how to
run the tests and what a complete run needs.

## 📖 Citing MDAPY

If MDAPY contributes to a publication, please cite
[Wu and Shao (2023)](reference/references.md#wu2023), and the papers that
the pages of the features you use ask to cite.
[Papers citing MDAPY](about/citations.md) lists, by year, the papers that cite it.

See the [release notes](about/release-notes.md) for what each
release contains.
