# Installation

mdapy needs Python 3.10 or newer, and on macOS, macOS 14 or newer. Its only
required dependencies are NumPy 2 and polars 2.

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

## Optional extras

| Extra | Installs | Needed for |
|---|---|---|
| `plot` | matplotlib | plotting helpers |
| `ase` | ASE | conversion to and from ASE |
| `dev` | pytest, pytest-cov, pytest-xdist, coverage, ruff | running the tests |
| `docs` | MkDocs Material, mkdocstrings | building this site |

```bash
pip install ".[dev]"
```

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
