# ![MDAPY](assets/mdapy-logo.png#only-light){ width="420" } ![MDAPY](assets/mdapy-logo-dark.png#only-dark){ width="420" } { .logo }

**MDAPY**, short for *Molecular Dynamics Analysis in Python*, builds, analyzes,
simulates and visualizes atomistic systems.

!!! note "MDAPY 2.0 is a complete rewrite"
    It is not compatible with MDAPY 1.x. It brings many more features, and we recommend
    upgrading: [From MDAPY 1.x](getting-started/from-1x.md) gives the 2.0 way of each
    common call. The code of 1.x stays on the branch [`1.x`](https://github.com/mushroomfire/mdapy/tree/1.x).

If you like MDAPY, please give it a ⭐ [star](https://github.com/mushroomfire/mdapy).

## 💡 Why MDAPY

- **Light.** [NumPy](https://numpy.org) and [polars](https://pola.rs) are its only
  requirements. Packages such as [matplotlib](https://matplotlib.org) or
  [ASE](https://wiki.fysik.dtu.dk/ase/) are optional, and it carries no other
  third-party code.
- **Robust.** Tests cover more than 99 % of the code and compare each feature with analytic
  results and with mature software. All of them run automatically on Linux, macOS and
  Windows. Results are the same for any number of threads, and damaged files or impossible
  inputs give a clear error, not a crash.
- **Fast.** The heavy parts run in C++, on a parallel engine of MDAPY's own. See the
  [benchmarks](performance/comparison.md).
- **Predictable.** Names say what the code does: `minimize`, `elastic_constants`,
  `CommonNeighborAnalysis`. A `System` never changes in place, every analysis has the form
  `compute(system)` and returns NumPy arrays, each concept has one name
  ([Glossary](reference/glossary.md)), and one unit system, Å, eV, ps, amu, K, and GPa for
  stress, runs through everything ([Units and conventions](reference/units.md)).
- **Stable.** From 2.0.0 on, code that works keeps working: the API changes only with
  3.0, a replaced name warns long before it goes, and a result changes only to fix a bug,
  named in the release notes ([Compatibility](about/compatibility.md)).
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

Structures are read or built into a `System`; its neighbor lists and the calculators
feed the analyses and the simulations, and the results are drawn or fitted. Each box
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

[<b>Calculators</b><span>native EAM · MEAM · Tersoff · NEP and more; LAMMPS · ASE · VASP</span>](user-guide/simulation/calculators.md){ .feature-card }

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
<p class="feature-label">Simulate</p>
<div class="feature-cells feature-chain" markdown>

[<b>Relaxation and dynamics</b><span>minimization · molecular dynamics · defect energies</span>](user-guide/simulation/minimize.md){ .feature-card }

<span class="feature-arrow" aria-hidden="true">→</span>

[<b>Lattice dynamics</b><span>elastic constants · phonons · sheets · finite temperature</span>](user-guide/simulation/elastic-constants.md){ .feature-card }

[<b>Thermodynamics</b><span>free energy · melting points · phase diagrams</span>](user-guide/simulation/free-energy.md){ .feature-card }

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
