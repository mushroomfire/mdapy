# Contributing

MDAPY 2.0 grows one feature at a time. This page lists the rules every
change follows.

## Issues and pull requests

1. Open an [issue](https://github.com/mushroomfire/mdapy/issues/new/choose) first and agree
   on the change there. A small fix, such as a typo, needs none. An issue is a bug report, a
   feature request, a report on performance or on the documentation, or a question, and each
   has its label: `bug`, `feature request`, `performance`, `documentation` or `question`.
2. Open a pull request against `master` and fill in its template: the issue it closes, how
   the results were checked and the checklist.
3. A pull request is merged when CI passes on Linux, macOS and Windows and it meets the rules
   of this page.

By submitting a pull request you agree that your contribution is licensed under the
[Mozilla Public License 2.0](https://github.com/mushroomfire/mdapy/blob/master/LICENSE), the
license of MDAPY, and you confirm that you have the right to contribute it.

## When a feature is done

A feature is merged only when it has all of the following:

1. the implementation;
2. tests that establish its correctness (see below), meeting the coverage
   thresholds;
3. its own page, placed by the documentation layout and following the page
   template below;
4. entries in the [Glossary](../reference/glossary.md) for any new names.

The maintainers write the [release notes](release-notes.md) when they merge it.

## Releases

A release is a tag `v` and its version on master, such as `v2.0.0`; CI builds the
wheels, tests each of them and uploads them to PyPI. Releases follow the promises of
[Compatibility](compatibility.md): a change to the public API waits for 3.0, and a
replaced name keeps working and warns until then.

The site of the documentation is published from the tag of each release, so that it
describes the version `pip` installs, not the next one that master holds. A correction
of the documentation is published with the next release, or at once by running the
workflow `docs` on the tag by hand.

## Documentation layout

The tabs follow what the reader is doing, and the directories of `docs/` follow the tabs:

| Tab | Directory | Holds |
|---|---|---|
| Getting started | `getting-started/` | installation, the quick tour, the lessons and the cheat sheet in `getting-started/learn/` |
| Basics | `basics/` | what every user needs: systems, the box, selecting atoms, files, trajectories, neighbor lists, parallelism |
| Examples | `examples/` | whole calculations from start to end, each a page of its own |
| User guide | `user-guide/` | one directory per area, each page one feature: `build/`, `analysis/`, `simulation/`, `mlip/`, `visualization/`, `interop/` |
| Performance | `performance/` | timings of MDAPY and of other programs |
| Reference | `reference/` | the Python API in `reference/api/`, units and conventions, elements, the glossary, the references, file formats |
| About | `about/` | the release notes, compatibility, reporting a problem, this page |

Each feature has its own page, named after it (`eam.md`, `minimize.md`), in
the section of its tab that fits it; closely tied classes, such as an
analysis and its result, share the page. A feature that fits no section goes
to a section **Misc** of its tab until a section for it exists. A new page is
added to the navigation in `mkdocs.yml`.

## Feature pages

Each feature has a hand-written page that tells how to use it, with these
sections, in this order:

1. **Title and one sentence** saying what the feature does.
2. **Minimal example** that runs as shown. `tests/test_docs.py` runs every
   Python block of every page, in order, with shared variables. A block that
   is only an illustration is preceded by the line `<!-- skip-test -->`. A
   page whose examples need an optional package starts with a line such as
   `<!-- needs: ovito -->`; its examples are skipped where the package is
   not installed. Examples import mdapy as `import mdapy as mp` and write
   `mdapy.io.read`, `mp.System`; the text names the functions in full, as
   `mdapy.io.read`. Where a result can be seen, a short example after it draws
   it with mdapy's renderer or plots it with `mdapy.visualization.plot`
   into `images/name.png`; the tests run it in a folder with an empty
   `images`. The picture, made by running the page's examples and kept
   next to the page as a PNG of 256 colors, follows the block with a
   caption in italics that says what it shows.
3. **Parameters**, as a table: `| Name | Type | Default | Unit | Meaning |`.
4. **Results**: the attributes and every public method, as tables, with the
   conventions needed to read them.
5. **Errors**: when the documented exceptions are raised.
6. For analyses, **Definition**: what is computed, with formulas, so that a
   result can be interpreted.
7. **Please cite**: for a feature built on published work, a box
   (`!!! quote "Please cite"`) at the end of the page, linking the
   papers to cite when the feature is used.

Papers are listed once, on the [References](../reference/references.md)
page, each with an anchor such as `#zhou2004`; pages cite them as links,
for example `[Zhou et al. (2004)](../reference/references.md#zhou2004)`,
and never repeat them. Web pages a feature follows, such as the
specification of a file format or the LAMMPS documentation of a command, are
plain links on the feature's page, in a **See also** section.

Pages do not describe how a feature is implemented, how it is tested or how
fast it is: the tests are the record of correctness, and timings belong on
a separate benchmark page. Sections that do not apply are left out.

## Docstrings

Docstrings use the numpy style and describe inputs and outputs only: meaning,
type, shape and unit of each parameter and return value, and the exceptions
raised. They contain no formulas and no explanation of the method; that
belongs on the feature page. The [Python API](../reference/api/index.md) page is
generated from them.

## Tests

- Correctness comes from sources independent of the code under test:
    - analytic results, such as known values for perfect lattices;
    - physical invariances: rotation, translation, permutation of atoms,
      replication of the cell, choice of equivalent cells;
    - brute-force reference implementations in plain NumPy, used on small
      systems;
    - finite differences for forces and virials;
    - round trips and hand-written examples for file formats.
- Where LAMMPS, OVITO, freud or atomsk implements the same method, the
  tests compare with it, run live at the reference versions, LAMMPS
  stable_30Sep2026, OVITO 3.17.1, freud 3.6.1 and atomsk Beta 0.13.1; freud
  computes in single precision, so its values agree to about 1e-6. They
  are skipped where these are not installed. Results are never compared
  against stored snapshots of other software.
- Parallel code is tested for identical results across thread counts.
- Error paths are tested: invalid input raises the documented exception.
- Comparisons state their tolerance and why, and failure messages show the
  size of the discrepancy.
- Tests run from any working directory and locate their data through
  fixtures.

Coverage thresholds:

- **Total:** at least 90 %, enforced by `pytest --cov`.
- **Per file:** at least 95 %, checked in CI from the JSON coverage report.

```bash
pytest --cov --cov-report=term
```

The environment variable `MDAPY_SOFTWARE_GPU` lets pictures on the GPU use a driver that runs
on the processor, which they never do otherwise. CI has no GPU on Linux and Windows, so it
runs the tests of pictures once more this way there, for the code and the shaders of Vulkan
and Direct3D 12. A real GPU remains the only check of its drivers and ray tracing units.

NumPy and polars are what MDAPY stands on. CI tests them at both ends: one job with
Python 3.11 and the lowest versions `pyproject.toml` accepts, the others with the newest.
Every week the tests also run with the newest releases and with what comes next, the
nightly wheels of NumPy and the pre-releases of polars, so that a change of theirs that
breaks MDAPY, or warns of one, shows before it reaches users. A new lower bound in
`pyproject.toml` is the version the lowest job installs.

### What a complete run needs

`pip install -e ".[dev]"` gives everything a plain `pytest` needs: the test
tools, [matplotlib](https://matplotlib.org), [ASE](https://ase-lib.org) and
[spglib](https://spglib.readthedocs.io). A test that needs more is skipped
where it is missing, so a plain run passes on any machine; `pytest -rs` lists
the skipped tests and why. A complete run needs the following.

```bash
pip install -e ".[dev]"
pip install ovito==3.17.1 freud-analysis==3.6.1 phonopy==4.8.6 seekpath==2.2.2 \
    dscribe==2.1.2 pymatgen-core networkx torch torchnep
```

| Package | Tested with it |
|---|---|
| [OVITO](https://www.ovito.org) | structure analyses, colors and pictures, against OVITO |
| [freud](https://freud.readthedocs.io) | analyses and conversions, against freud |
| [phonopy](https://phonopy.github.io/phonopy) | phonons, against phonopy; its dependency symfc for phonons at finite temperature |
| [seekpath](https://seekpath.readthedocs.io) | band paths from the symmetry |
| [DScribe](https://singroup.github.io/dscribe) | SOAP descriptors, against DScribe |
| [pymatgen](https://pymatgen.org) | conversions to and from it |
| [NetworkX](https://networkx.org) | rings, against a brute-force search |
| [torchnep](https://github.com/mushroomfire/torchnep), with [PyTorch](https://pytorch.org) | the files of training and of predictions |

[LAMMPS](https://www.lammps.org) stable_30Sep2026 is built from source, with
its Python module installed from the wheel that the build leaves:

```bash
git clone --depth 1 --branch stable_30Sep2026 https://github.com/lammps/lammps.git
cmake -S lammps/cmake -B lammps-build -D CMAKE_BUILD_TYPE=Release \
    -D BUILD_SHARED_LIBS=on -D BUILD_MPI=no -D BUILD_OMP=no \
    -D PKG_MANYBODY=on -D PKG_MOLECULE=on -D PKG_EXTRA-COMPUTE=on \
    -D PKG_EXTRA-FIX=on -D PKG_EXTRA-PAIR=on \
    -D PKG_VORONOI=on -D DOWNLOAD_VORO=on -D PKG_PHONON=on \
    -D PKG_KOKKOS=on -D Kokkos_ENABLE_SERIAL=on -D Kokkos_ENABLE_THREADS=on \
    -D Python_EXECUTABLE="$(which python)"
cmake --build lammps-build -j 4
cmake --build lammps-build --target install-python
pip install lammps-build/lammps-*.whl
```

| LAMMPS package | Tested with it |
|---|---|
| `MANYBODY` | the EAM potentials of the tests |
| `MOLECULE` | molecular files |
| `EXTRA-COMPUTE` | bond angles, the entropy fingerprint and Ackland-Jones |
| `EXTRA-FIX`, `EXTRA-PAIR` | free energies of solids and liquids |
| `VORONOI` | the Voronoi analysis |
| `PHONON` | dynamical matrices |
| `KOKKOS` on the CPU | the shutdown of Kokkos at exit; a build of it for GPUs skips this test |

The rest are programs and conditions of the machine:

| Needs | Tested with it |
|---|---|
| [Atomsk](https://atomsk.univ-lille.fr) Beta 0.13.1 on the `PATH` | the structure builder, against Atomsk |
| [FFmpeg](https://ffmpeg.org) on the `PATH` | movies |
| a GPU | pictures on the GPU; ray tracing needs a GPU with ray tracing units |
| `glslc` of the shaderc version named in `render_vulkan.comp` | the compiled shader of Linux |
| `dxc` of the Windows SDK named in `render_d3d12.hlsl` | the compiled shader of Windows |
| the test data of [torchnep](https://github.com/mushroomfire/torchnep), beside this repository or at `MDAPY_TORCHNEP_DATA` | NEP against its reference values |
| Python 3.14 | files compressed with zstd |
| a German or French locale | numbers under a locale with a decimal comma |

## Compiled code

- C++17. All compiled code goes into the single extension module
  `mdapy._core`; each area registers a submodule from `src/cpp/bindings/`.
- Kernels never create threads or use OpenMP. They call
  `mdapy::parallel_for` or `mdapy::parallel_reduce`.
- Kernels that are called from Python run through
  `mdapy::bindings::run_released`, which releases the GIL and supports
  `progress` and `cancel`.
- Code compiles without warnings with `-Wall -Wextra -Wpedantic -Wshadow`; CI
  builds with warnings as errors and also runs the tests under
  AddressSanitizer and UndefinedBehaviorSanitizer.
- `tests/cpp/thread_checks.cpp` checks the thread pool and the parallel
  kernels without Python; CI runs it under ThreadSanitizer. New parallel
  kernels get a check there.

## Style

- `ruff check .` and `ruff format --check .` pass.
- Names private to a class start with two underscores, `self.__table`, so
  that they cannot be reached by accident from outside. Names that other
  modules of MDAPY use, and methods that subclasses override, start with one
  underscore: they are internal to MDAPY, not to the class. Private
  functions and constants of a module start with one underscore.
- Immutable objects (`System`, `Box`, `Topology`) refuse new attributes and
  return themselves from `copy.copy` and `copy.deepcopy`.
- Commit messages are short English sentences describing the change.
