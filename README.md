<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/mushroomfire/mdapy/master/docs/assets/mdapy-logo-dark.png">
    <img alt="MDAPY" src="https://raw.githubusercontent.com/mushroomfire/mdapy/master/docs/assets/mdapy-logo.png" width="420">
  </picture>
</p>

<p align="center">
  <b>Fast building, analysis, simulation and visualization of atomistic systems, in Python.</b>
</p>

<p align="center">
  <a href="https://github.com/mushroomfire/mdapy/actions/workflows/tests.yml"><img alt="Tests" src="https://github.com/mushroomfire/mdapy/actions/workflows/tests.yml/badge.svg?branch=master"></a>
  <img alt="Python 3.11 or newer" src="https://img.shields.io/badge/python-3.11%2B-3776AB">
  <img alt="Linux, macOS, Windows" src="https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-555555">
  <a href="https://github.com/mushroomfire/mdapy/blob/master/LICENSE"><img alt="License: MPL-2.0" src="https://img.shields.io/badge/license-MPL--2.0-555555"></a>
  <a href="https://doi.org/10.1016/j.cpc.2023.108764"><img alt="Paper: Comput. Phys. Commun. 290, 108764" src="https://img.shields.io/badge/paper-CPC%202023-555555"></a>
</p>

<p align="center">
  <a href="https://mushroomfire.github.io/mdapy/">Documentation</a> ·
  <a href="https://mushroomfire.github.io/mdapy/getting-started/quick-tour/">Quick tour</a> ·
  <a href="https://mushroomfire.github.io/mdapy/getting-started/learn/table/">Lessons</a> ·
  <a href="#-examples">Examples</a> ·
  <a href="#-install">Install</a> ·
  <a href="#-tests">Tests</a> ·
  <a href="#-citing-mdapy">Cite</a>
</p>

> **MDAPY 2.0 is a complete rewrite and is not compatible with MDAPY 1.x.** It brings many
> more features, and we recommend upgrading. The code of 1.x stays on the branch
> [`1.x`](https://github.com/mushroomfire/mdapy/tree/1.x).

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
  [benchmarks](https://mushroomfire.github.io/mdapy/performance/comparison/).
- **Predictable.** Names say what the code does: `minimize`, `elastic_constants`,
  `CommonNeighborAnalysis`. A `System` never changes in place, every analysis has the form
  `compute(system)` and returns NumPy arrays, and one unit system, Å, eV and ps, runs
  through everything.
- **Integrated.** One package and one `System`, from the first structure to the last
  picture. It converts systems to and from [ASE](https://wiki.fysik.dtu.dk/ase/),
  [OVITO](https://www.ovito.org), [pymatgen](https://pymatgen.org) and
  [freud](https://freud.readthedocs.io), takes any ASE calculator or
  [LAMMPS](https://www.lammps.org) interaction, shares a process with
  [PyTorch](https://pytorch.org), and works as one step of
  [high-throughput calculations](https://mushroomfire.github.io/mdapy/examples/high-throughput/).
- **Documented.** Every feature has its own page, with an example, its parameters, its
  results and its references, and [lessons](https://mushroomfire.github.io/mdapy/getting-started/learn/table/) lead
  from the first structure on.

## 🧰 What it does

| Area | What is in it |
|---|---|
| [**Build**](https://mushroomfire.github.io/mdapy/user-guide/build/crystals/) | Crystals, polycrystals, alloys, defects and dislocations |
| [**Analyze**](https://mushroomfire.github.io/mdapy/user-guide/analysis/overview/) | More than 40 analyses of structure, local order, defects and dynamics |
| [**Simulate**](https://mushroomfire.github.io/mdapy/user-guide/simulation/calculators/) | Native EAM and NEP, interfaces to [LAMMPS](https://www.lammps.org) and [ASE](https://wiki.fysik.dtu.dk/ase/); relaxation and dynamics; elastic constants, phonons, free energies, melting points and phase diagrams |
| [**MLIP**](https://mushroomfire.github.io/mdapy/user-guide/mlip/checks/) | Training sets from [VASP](https://www.vasp.at), descriptors, errors of a model, the frames worth computing |
| [**Visualize**](https://mushroomfire.github.io/mdapy/user-guide/visualization/rendering/) | Ray-traced pictures on the processor or the graphics card, movies and plots |
| [**Exchange**](https://mushroomfire.github.io/mdapy/basics/io/) | Eight file formats, and systems shared with [ASE](https://wiki.fysik.dtu.dk/ase/), [OVITO](https://www.ovito.org), [pymatgen](https://pymatgen.org) and [freud](https://freud.readthedocs.io) |

## 🎨 Examples

One copper polycrystal, from nothing to a movie.

<table>
<tr><th width="58%">Build</th><th>Grains, 81,734 atoms</th></tr>
<tr><td>

```python
import mdapy as mp
from mdapy.build import build_crystal, build_polycrystal
from mdapy.visualization import render

copper = build_crystal("fcc", 3.615, elements="Cu")
grains = build_polycrystal(copper, mp.Box(100.0), 8, seed=1)
render(grains, "grains.png", color="grain")
```

</td><td><img alt="A polycrystal colored by grain" src="https://raw.githubusercontent.com/mushroomfire/mdapy/master/docs/assets/readme/grains.png" width="100%"></td></tr>
<tr><th>Analyze</th><th>The atoms that are not fcc</th></tr>
<tr><td>

```python
from mdapy.identify import CommonNeighborAnalysis

structures = CommonNeighborAnalysis().compute(grains)
structures.counts()  # {'other': 21043, 'fcc': 60682, ...}
boundaries = structures.structure != 1  # not fcc
render(grains, "boundaries.png", atoms=boundaries, color=structures)
```

</td><td><img alt="The grain boundaries of the polycrystal" src="https://raw.githubusercontent.com/mushroomfire/mdapy/master/docs/assets/readme/boundaries.png" width="100%"></td></tr>
<tr><th>Simulate</th><th>Phonons of copper</th></tr>
<tr><td>

```python
from mdapy.calculators import EAM
from mdapy.lattice_dynamics import phonons, primitive_cell
from mdapy.simulation import minimize
from mdapy.visualization import plot

eam = EAM.zhou(["Cu"])
relaxed = minimize(copper, eam, cell="full", max_stress=1e-4)
cell = primitive_cell(relaxed.system)
bands = phonons(cell, eam).band_structure("fcc")
with plot.style():
    fig, ax = plot.figure()
    plot.band_structure(bands, ax=ax)
    plot.save(fig, "phonons", formats=("png",))
```

</td><td><img alt="The phonon dispersion of copper" src="https://raw.githubusercontent.com/mushroomfire/mdapy/master/docs/assets/readme/phonons.png" width="100%"></td></tr>
<tr><th>MLIP</th><th>The 20 frames most worth a DFT calculation</th></tr>
<tr><td>

```python
from mdapy import mlip

crystal = build_crystal("fcc", 3.615, elements="Cu", repeat=(3, 3, 3))
frames = mlip.perturb(crystal, 200, displacement=0.1, seed=1)
points = mlip.structure_descriptors(frames, mlip.SOAP(["Cu"]))
chosen = mlip.farthest_point_sampling(points, 20)

xy = mlip.principal_components(points).transform(points)
with plot.style():
    fig, ax = plot.figure()
    ax.scatter(*xy.T, s=4, color="0.7", label="all frames")
    ax.scatter(*xy[chosen].T, s=12, label="chosen")
    ax.set_xlabel("principal component 1")
    ax.set_ylabel("principal component 2")
    ax.legend()
    plot.save(fig, "frames", formats=("png",))
```

</td><td><img alt="Frames on the map of their descriptors, with those chosen" src="https://raw.githubusercontent.com/mushroomfire/mdapy/master/docs/assets/readme/frames.png" width="100%"></td></tr>
</table>

### Make a movie

<p align="center">
  <a href="https://mushroomfire.github.io/mdapy/examples/movie/"><img alt="A movie that flies through a copper polycrystal" src="https://raw.githubusercontent.com/mushroomfire/mdapy/master/docs/examples/images/movie.webp" width="480"></a>
</p>

See [A movie that flies through a crystal](https://mushroomfire.github.io/mdapy/examples/movie/)
for the whole script.

## 📦 Install

```bash
pip install --pre mdapy
```

The first release of 2.0 is a preview, and `--pre` asks `pip` for it: without it, `pip`
installs 1.x. MDAPY needs Python 3.11 or newer. [Installation](https://mushroomfire.github.io/mdapy/getting-started/installation/)
describes the other ways to install it, from a source archive or from git, what those need,
and the optional packages.

## 🧪 Tests

```bash
git clone https://github.com/mushroomfire/mdapy.git
cd mdapy
pip install -e ".[dev]"
pytest
```

Tests that need another program are skipped where it is missing.
[Contributing](https://mushroomfire.github.io/mdapy/about/contributing/#what-a-complete-run-needs) lists what a
complete run needs.

## 📖 Citing MDAPY

If MDAPY contributes to a publication, please cite:

> Yong-Chao Wu and Jian-Li Shao, *mdapy: A flexible and efficient analysis software
> for molecular dynamics simulations*, Computer Physics Communications **290**, 108764
> (2023). [doi:10.1016/j.cpc.2023.108764](https://doi.org/10.1016/j.cpc.2023.108764)

## 📄 License

Mozilla Public License 2.0, see [LICENSE](https://github.com/mushroomfire/mdapy/blob/master/LICENSE):
MDAPY may be used in any software, open or closed, and changes to its own files are
shared under the same license.

## 📜 Code history

MDAPY has been written three times.

| Series | Years | How it was built | Code |
|---|---|---|---|
| 0.x | 2022–2025 | Python, with [Taichi](https://github.com/taichi-dev/taichi) for the kernels | Branch [`mdapy_old`](https://github.com/mushroomfire/mdapy/tree/mdapy_old), last release 0.11.5 |
| 1.x | 2025–2026 | Rewritten with every kernel in C++, and with NumPy and polars as its only requirements | Branch [`1.x`](https://github.com/mushroomfire/mdapy/tree/1.x), last release 1.0.10, [documentation](https://mdapy.readthedocs.io) |
| 2.x | 2026– | Rewritten again from scratch, for the reasons below | This branch |

Taichi was no longer updated, which also limited the versions of Python that 0.x could run
on. 1.x was rewritten to be easy to update and maintain, and it needs only NumPy and polars,
no longer SciPy and Taichi.

2.x goes further: it is easier to maintain, more stable and more consistent, lighter without
external code, with many more features and a clearer organization. In particular:

- **OpenMP.** The [OpenMP](https://www.openmp.org) runtime of 1.x could clash with those of
  other packages. 2.x runs a thread pool of its own, so it shares a process with
  [PyTorch](https://pytorch.org) and [LAMMPS](https://www.lammps.org).
- **Third-party code.** 1.x carried ten third-party codes to build and keep working. 2.x
  carries none.
- **The API.** In 1.x an analysis was a method that changed the system. In 2.x a `System`
  never changes and every analysis is a class with one form.
- **Many small calculations.** The threads can be limited for each process, so thousands of
  small calculations side by side do not fight for cores.
- **Graphical interfaces.** Long computations report their progress, can be cancelled and
  leave Python running, so a graphical interface is easy to build on 2.x.
