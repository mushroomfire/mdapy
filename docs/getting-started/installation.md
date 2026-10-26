# Installation

mdapy needs Python 3.11 or newer, and on macOS, macOS 14 or newer. Its only
required dependencies are NumPy 2.3.1 or newer and polars 2.

## From source

mdapy 2.0 is not on PyPI yet. Build it from the `v2` branch:

```bash
git clone -b v2 https://github.com/mushroomfire/mdapy.git
cd mdapy
pip install .
```

Building needs a C++17 compiler and CMake 3.18 or newer; `pip` installs the
remaining build tools (scikit-build-core and nanobind) automatically. No
OpenMP library is needed.

## 64-bit atom indices

Atom indices in results such as neighbor lists are 32-bit integers, enough
for 2,147,483,647 atoms and half the memory of 64-bit ones. For larger
systems, build mdapy with 64-bit indices:

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

mdapy needs only NumPy and polars. Other packages, when installed, switch on
the parts of mdapy that use them; without them the rest works, and these
parts raise an error that names the package to install.

| Package | Install | For |
|---|---|---|
| matplotlib | `pip install "mdapy[plot]"` | the plots of [`mdapy.visualization.plot`](../visualization/plots.md) |
| spglib, seekpath | `pip install "mdapy[symmetry]"` | the [space group](../analysis/space-group.md); phonons and elastic constants from fewer computations; `band_structure("auto")` |
| ASE | `pip install "mdapy[ase]"` | [`to_ase`, `from_ase`](../interop/ase.md), and [calculators](../interop/ase-calculators.md) between ASE and mdapy |
| freud | `pip install "mdapy[freud]"` | [`to_freud`, `from_freud`](../interop/freud.md) |
| pymatgen | `pip install "mdapy[pymatgen]"` | [`to_pymatgen`, `from_pymatgen`](../interop/pymatgen.md) |
| OVITO | `pip install ovito` | [`to_ovito`, `from_ovito`](../interop/ovito.md) |
| LAMMPS | its Python module, built with LAMMPS | the [LAMMPS calculator](../simulation/lammps.md), [molecular dynamics](../simulation/dynamics.md) and [elastic constants from molecular dynamics](../simulation/md-elastic.md) |

Extras combine: `pip install "mdapy[plot,symmetry,ase]"`. OVITO's package
from PyPI warns in a conda environment, where conda's own is preferred, so
it is not an extra; nor is LAMMPS, whose Python module comes with a build
of LAMMPS with the packages of styles each user needs.

## Tests and documentation

| Extra | Installs | For |
|---|---|---|
| `dev` | pytest, pytest-cov, pytest-xdist, coverage, ruff, matplotlib, ASE, spglib | running the tests and the linter |
| `docs` | MkDocs Material, mkdocstrings | building this site |

The tests that compare mdapy with other programs run where these are
installed and are skipped otherwise; continuous integration installs them
all:

| Package | Version | Compared |
|---|---|---|
| OVITO | 3.17.0 | structure identification and analyses, atom by atom |
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
[Contributing](../development/contributing.md).
