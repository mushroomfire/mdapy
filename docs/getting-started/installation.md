# Installation

MDAPY needs Python 3.11 or newer, and on macOS, macOS 14 or newer. Its only
required dependencies are NumPy 2.3.1 or newer and polars 2.

## From PyPI

```bash
pip install mdapy
```

This is the usual way. PyPI holds built packages for Linux on x86-64 and on ARM,
macOS on Apple silicon and Windows on x86-64, so nothing has to be compiled. Elsewhere, `pip`
builds MDAPY from its source archive, which needs [the tools below](#what-a-build-needs).

These systems are the ones officially supported. On others, such as macOS on an Intel
processor, MDAPY built from source should work as well, but no package is built for them
and no test runs on them.

## From a source archive

Build the source archive on PyPI, instead of taking a built package:

```bash
pip install --no-binary mdapy mdapy
```

or a source archive you have downloaded:

```bash
pip install mdapy-*.tar.gz
```

## From git

For the newest code, directly:

```bash
pip install "git+https://github.com/mushroomfire/mdapy.git"
```

or from a clone, which also has the tests and the documentation:

```bash
git clone https://github.com/mushroomfire/mdapy.git
cd mdapy
pip install .
```

## What a build needs

Installing from a source archive or from git compiles MDAPY, which needs:

| Needed | Version | How to get it |
|---|---|---|
| A C++ compiler | C++17: GCC, Clang or MSVC | Debian, Ubuntu: `sudo apt install g++`; Fedora, RHEL: `sudo dnf install gcc-c++`; macOS: `xcode-select --install`; Windows: Visual Studio with "Desktop development with C++" |
| CMake | 3.18 or newer | `pip install cmake`, or the package of your system |
| scikit-build-core, nanobind | 0.10 or newer, 2.4 or newer | `pip` fetches them for the build |
| NumPy, polars | 2.3.1 or newer, 2 or newer | `pip` installs them with MDAPY |

No OpenMP library and no other third-party code is needed.

## An old Linux

The built packages on PyPI need glibc 2.27 or newer, about as those of NumPy do: Ubuntu 18.04,
RHEL 8 or newer. `mdapy.show_versions()` or `ldd --version` tells which glibc a machine has.

On an older system `pip` finds no built package that fits, and builds MDAPY and NumPy from their
source archives, which takes [the tools above](#what-a-build-needs). A message such as
`GLIBC_2.27 not found` means a package built for a newer system. On a cluster, a newer compiler
usually comes from a module, such as `module load gcc`. In a [conda](https://conda-forge.org)
environment, get a compiler and NumPy that run on the old system first, and `pip` then builds
MDAPY alone:

```bash
conda install -c conda-forge compilers numpy polars
pip install mdapy
```

## Pictures on the GPU

[Pictures](../user-guide/visualization/rendering.md#on-the-gpu) are traced on the GPU
with Metal on macOS and Direct3D 12 on Windows, in every build, and with Vulkan
on Linux when the build finds Vulkan's headers: install them before MDAPY, as
`libvulkan-dev` on Debian and Ubuntu or `vulkan-headers` on Fedora and RHEL. MDAPY opens the
Vulkan loader, `libvulkan.so.1`, only for a picture on the GPU; the loader
and the driver come with the graphics drivers. Without the headers, or
without a GPU, pictures are traced on the processor.

## 64-bit atom indices

Atom indices in results such as neighbor lists are 32-bit integers, enough
for 2,147,483,647 atoms and half the memory of 64-bit ones. For larger
systems, build MDAPY with 64-bit indices:

```bash
pip install . -Ccmake.define.MDAPY_INDEX64=ON
```

`mdapy.index_dtype` tells which build is installed:

```bash
python -c "import mdapy; print(mdapy.index_dtype)"   # int64
```

The table of a `System` holds a row per atom, and the default runtime of
polars at most 4,294,967,295 rows. Beyond that, install the 64-bit runtime
of polars too, which polars then takes by itself:

```bash
pip install "polars[rt64]"
python -c "import polars; print(polars.get_index_type())"   # UInt64
```

## Optional dependencies

MDAPY needs only NumPy and polars. Other packages, when installed, switch on
the parts of MDAPY that use them; without them the rest works, and these
parts raise an error that names the package to install.

| Package | Install | For |
|---|---|---|
| matplotlib | `pip install "mdapy[plot]"` | the plots of [`mdapy.visualization.plot`](../user-guide/visualization/plots.md) |
| spglib, seekpath | `pip install "mdapy[symmetry]"` | the [space group](../user-guide/analysis/space-group.md); phonons and elastic constants from fewer computations; `band_structure("auto")` |
| ASE | `pip install "mdapy[ase]"` | [`to_ase`, `from_ase`](../user-guide/interop/ase.md), and [calculators](../user-guide/interop/ase-calculators.md) between ASE and mdapy |
| freud | `pip install "mdapy[freud]"` | [`to_freud`, `from_freud`](../user-guide/interop/freud.md) |
| pymatgen | `pip install "mdapy[pymatgen]"` | [`to_pymatgen`, `from_pymatgen`](../user-guide/interop/pymatgen.md) |
| OVITO | `pip install ovito` | [`to_ovito`, `from_ovito`](../user-guide/interop/ovito.md) |
| LAMMPS | its Python module, built with LAMMPS | the [LAMMPS calculator](../user-guide/simulation/lammps.md), [molecular dynamics](../user-guide/simulation/dynamics.md) and [elastic constants from molecular dynamics](../user-guide/simulation/md-elastic.md) |

Extras combine: `pip install "mdapy[plot,symmetry,ase]"`. OVITO's package
from PyPI warns in a conda environment, where conda's own is preferred, so
it is not an extra; nor is LAMMPS, whose Python module comes with a build
of LAMMPS with the packages of styles each user needs.

## Tests and documentation

| Extra | Installs | For |
|---|---|---|
| `dev` | pytest, pytest-cov, pytest-xdist, coverage, ruff, matplotlib, ASE, spglib | running the tests and the linter |
| `docs` | MkDocs Material, mkdocstrings | building this site |

The tests that compare MDAPY with other programs run where these are
installed and are skipped otherwise; continuous integration installs them
all:

| Package | Version | Compared |
|---|---|---|
| OVITO | 3.17.1 | structure identification and analyses, atom by atom |
| freud | 3.6.1 | analyses that freud implements |
| LAMMPS | stable 30 Sep 2026, with MANYBODY, MOLECULE, EXTRA-COMPUTE and VORONOI | calculators, analyses, molecular dynamics; the dynamical matrix where PHONON is built in |
| phonopy | 4.8.6 | force constants, frequencies, densities of states, thermal properties |
| seekpath | 2.2.2 | the paths of `band_structure("auto")` |
| pymatgen | `pymatgen-core` | conversions |

## Development setup

An editable install rebuilds nothing on import; rerun the install after
changing C++ code.

```bash
pip install -e ".[dev,docs]"
pytest                      # run the test suite
pytest --cov                # with coverage (fails below 90 % in total)
ruff check . && ruff format --check .
mkdocs serve                # preview the documentation
```

The test suite and the coverage rules are described in
[Contributing](../about/contributing.md).
