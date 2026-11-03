<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/mushroomfire/mdapy/v2/docs/assets/mdapy-logo-dark.png">
    <img alt="MDAPY" src="https://raw.githubusercontent.com/mushroomfire/mdapy/v2/docs/assets/mdapy-logo.png" width="420">
  </picture>
</p>

<p align="center">
  <b>Fast building, analysis, simulation and visualization of atomistic systems, in Python.</b>
</p>

<p align="center">
  <a href="https://github.com/mushroomfire/mdapy/actions/workflows/tests.yml"><img alt="Tests" src="https://github.com/mushroomfire/mdapy/actions/workflows/tests.yml/badge.svg?branch=v2"></a>
  <img alt="Python 3.11 or newer" src="https://img.shields.io/badge/python-3.11%2B-3776AB">
  <img alt="Linux, macOS, Windows" src="https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-555555">
  <a href="https://github.com/mushroomfire/mdapy/blob/v2/LICENSE"><img alt="License: MPL-2.0" src="https://img.shields.io/badge/license-MPL--2.0-555555"></a>
  <a href="https://doi.org/10.1016/j.cpc.2023.108764"><img alt="Paper: Comput. Phys. Commun. 290, 108764" src="https://img.shields.io/badge/paper-CPC%202023-555555"></a>
</p>

<p align="center">
  <a href="https://github.com/mushroomfire/mdapy/blob/v2/docs/index.md">Documentation</a> ·
  <a href="https://github.com/mushroomfire/mdapy/blob/v2/docs/getting-started/quick-tour.md">Quick tour</a> ·
  <a href="https://github.com/mushroomfire/mdapy/blob/v2/docs/getting-started/learn/table.md">Lessons</a> ·
  <a href="#examples">Examples</a> ·
  <a href="#install">Install</a> ·
  <a href="#citing-mdapy">Cite</a>
</p>

> **MDAPY 2.0 is a complete rewrite and is not compatible with MDAPY 1.x.** It is
> under development, and a first preview, 2.0.0a1, comes next. The stable release
> is `pip install mdapy==1.0.10`, with its
> [documentation](https://mdapy.readthedocs.io).

## Why MDAPY

- **Light.** No third-party code inside: every kernel is MDAPY's own C++. NumPy and
  polars are all it needs, and everything else, such as matplotlib, ASE or LAMMPS, is
  optional.
- **Flexible.** Results are plain NumPy arrays and atoms are rows of a polars table, so
  each step mixes with your own code: your own columns and masks of atoms, your own
  [calculator](https://github.com/mushroomfire/mdapy/blob/v2/docs/simulation/calculators.md),
  your own spheres, cylinders and polygons in a picture.
- **Robust.** More than 2,800 tests compare each feature with analytic results,
  independent implementations and live runs of LAMMPS, OVITO, freud and phonopy.
  Results are the same for any number of threads, and damaged files or impossible
  inputs give a clear error, not a crash.
- **Fast.** The heavy work runs in compiled C++ on every core. From 256,000 atoms on it
  is faster than OVITO in every analysis they share, and from 32,000 atoms on faster
  than freud in every one:
  [every case at every size](https://github.com/mushroomfire/mdapy/blob/v2/docs/performance/comparison.md).
- **Predictable.** A `System` never changes in place, every analysis has the same form,
  `compute(system)`, each concept has one name, and one unit system, Å, eV and ps, runs
  through everything.
- **Integrated.** One package and one `System`, from the first structure to the last
  picture. It converts systems to and from ASE, OVITO, pymatgen and freud, takes any ASE
  calculator or LAMMPS interaction, shares a process with PyTorch, and works as one step
  of a [large screening](https://github.com/mushroomfire/mdapy/blob/v2/docs/getting-started/tutorials/high-throughput.md).

## What it does

| Area | What is in it |
|---|---|
| [**Build**](https://github.com/mushroomfire/mdapy/blob/v2/docs/build/crystals.md) | Crystals, polycrystals, alloys, defects and dislocations |
| [**Analyze**](https://github.com/mushroomfire/mdapy/blob/v2/docs/analysis/overview.md) | More than 40 analyses of structure, local order, defects and dynamics |
| [**Simulate**](https://github.com/mushroomfire/mdapy/blob/v2/docs/simulation/calculators.md) | EAM, NEP, LAMMPS and ASE calculators; relaxation, dynamics, elastic constants and phonons |
| [**MLIP**](https://github.com/mushroomfire/mdapy/blob/v2/docs/mlip/checks.md) | Training sets from VASP, descriptors, errors of a model, the frames worth computing |
| [**Visualize**](https://github.com/mushroomfire/mdapy/blob/v2/docs/visualization/rendering.md) | Ray-traced pictures on the processor or the graphics card, movies and plots |
| [**Exchange**](https://github.com/mushroomfire/mdapy/blob/v2/docs/getting-started/io.md) | Eight file formats, and systems shared with ASE, OVITO, pymatgen and freud |

## Examples

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

</td><td><img alt="A polycrystal colored by grain" src="https://raw.githubusercontent.com/mushroomfire/mdapy/v2/docs/assets/readme/grains.png" width="100%"></td></tr>
<tr><th>Analyze</th><th>The atoms that are not fcc</th></tr>
<tr><td>

```python
from mdapy.identify import CommonNeighborAnalysis

structures = CommonNeighborAnalysis().compute(grains)
structures.counts()  # {'other': 21043, 'fcc': 60682, ...}
boundaries = structures.structure != 1  # not fcc
render(
    grains, "boundaries.png", atoms=boundaries, color=structures
)
```

</td><td><img alt="The grain boundaries of the polycrystal" src="https://raw.githubusercontent.com/mushroomfire/mdapy/v2/docs/assets/readme/boundaries.png" width="100%"></td></tr>
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

</td><td><img alt="The phonon dispersion of copper" src="https://raw.githubusercontent.com/mushroomfire/mdapy/v2/docs/assets/readme/phonons.png" width="100%"></td></tr>
<tr><th>MLIP</th><th>The 20 frames most worth a DFT calculation</th></tr>
<tr><td>

```python
from mdapy import mlip

crystal = build_crystal(
    "fcc", 3.615, elements="Cu", repeat=(3, 3, 3)
)
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

</td><td><img alt="Frames on the map of their descriptors, with those chosen" src="https://raw.githubusercontent.com/mushroomfire/mdapy/v2/docs/assets/readme/frames.png" width="100%"></td></tr>
</table>

### Make a movie

<p align="center">
  <a href="https://github.com/mushroomfire/mdapy/blob/v2/docs/getting-started/tutorials/movie.md"><img alt="A movie that flies through a copper polycrystal" src="https://raw.githubusercontent.com/mushroomfire/mdapy/v2/docs/getting-started/tutorials/images/movie.webp" width="480"></a>
</p>

Give `render` a camera for each frame, and hand the pictures to `write_movie` one at a
time. This turns around the polycrystal:

```python
import numpy as np

from mdapy.visualization import write_movie

look = dict(style="matte", focus=(50.0, 50.0, 50.0), span=170.0, width=1080, height=1080)
pictures = (
    render(grains, color=structures, view=[np.cos(t), np.sin(t), -0.5], **look)
    for t in np.linspace(0, 2 * np.pi, 300)
)
write_movie("turn.mp4", pictures, fps=30)  # with FFmpeg on the PATH
```

The movie above uses the same pieces, with a camera that flies between two twin planes:
[A movie that flies through a crystal](https://github.com/mushroomfire/mdapy/blob/v2/docs/getting-started/tutorials/movie.md)
gives the whole script.

## Install

```bash
git clone -b v2 https://github.com/mushroomfire/mdapy.git
cd mdapy
pip install .
```

MDAPY needs Python 3.11 or newer and, to build, a C++17 compiler and CMake 3.18 or
newer. It will be on PyPI from the first preview on. See
[Installation](https://github.com/mushroomfire/mdapy/blob/v2/docs/getting-started/installation.md)
for the optional packages.

## Reporting a problem

Please [open an issue](https://github.com/mushroomfire/mdapy/issues) with the shortest
code that shows the problem and the output of `mdapy.show_versions()`. See
[Reporting a problem](https://github.com/mushroomfire/mdapy/blob/v2/docs/getting-started/reporting.md).

## Citing MDAPY

If MDAPY contributes to a publication, please cite:

> Yong-Chao Wu and Jian-Li Shao, *mdapy: A flexible and efficient analysis software
> for molecular dynamics simulations*, Computer Physics Communications **290**, 108764
> (2023). [doi:10.1016/j.cpc.2023.108764](https://doi.org/10.1016/j.cpc.2023.108764)

## License

Mozilla Public License 2.0, see [LICENSE](https://github.com/mushroomfire/mdapy/blob/v2/LICENSE):
MDAPY may be used in any software, open or closed, and changes to its own files are
shared under the same license.

## Code history

MDAPY has been written three times.

| Series | Years | How it was built | Code |
|---|---|---|---|
| 0.x | 2022–2025 | Python, with [Taichi](https://github.com/taichi-dev/taichi) for the kernels | Branch [`mdapy_old`](https://github.com/mushroomfire/mdapy/tree/mdapy_old), last release 0.11.5 |
| 1.x | 2025–2026 | Rewritten with every kernel in C++, for speed | Tag [`v1.0.10`](https://github.com/mushroomfire/mdapy/tree/v1.0.10), [documentation](https://mdapy.readthedocs.io) |
| 2.x | 2026– | Rewritten again from scratch, for the reasons below | This branch |

1.x was fast, but other programs found it hard to live with. 2.x was rewritten to solve that:

- **OpenMP.** The OpenMP runtime of 1.x could clash with those of other packages: on
  macOS its wheel crashed next to the one in a conda environment. 2.x runs a thread pool
  of its own and needs no OpenMP, so it shares a process with PyTorch and LAMMPS.
- **Third-party code.** 1.x carried ten third-party codes, from Voro++ and polyhedral
  template matching to Tachyon and OptiX, each to build and keep working. 2.x carries
  none, and its license allows use in any software.
- **The API.** In 1.x an analysis was a method that changed the system,
  `system.cal_centro_symmetry_parameter(N=12)`. In 2.x a `System` never changes, every
  analysis is a class with one form, and the names come from one glossary.
- **Many small calculations.** The threads follow the CPUs a process may use and can be
  limited for each process, so thousands of small calculations side by side do not
  fight for cores.
- **Programs built on it.** Long computations report their progress, can be cancelled
  and leave Python running, so a graphical program stays responsive while it calls
  MDAPY.
