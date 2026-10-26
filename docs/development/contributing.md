# Contributing

mdapy 2.0 grows one feature at a time. This page lists the rules every
change follows.

## When a feature is done

A feature is merged only when it has all of the following:

1. the implementation;
2. tests that establish its correctness (see below), meeting the coverage
   thresholds;
3. its own page, placed by the documentation layout and following the page
   template below;
4. entries in the [Glossary](../reference/glossary.md) for any new names;
5. a line in the [release notes](release-notes.md).

## Documentation layout

The site has one tab per kind of content, each a directory of `docs/`:

| Tab | Directory | Holds |
|---|---|---|
| Getting started | `getting-started/` | installation, the quick tour, the basics every user needs (systems, the box, files, trajectories, neighbor lists, parallelism), and tutorials in `getting-started/tutorials/`: worked examples that combine several features, every one run by the tests (files a page needs sit in a folder next to it, named by a line `<!-- files: folder -->`) |
| Analysis | `analysis/` | analyses of structures and trajectories, in sections: structure identification; local order and distributions; defects and deformation; dynamics |
| Structure builder | `build/` | building structures (crystals, polycrystals, defects, alloys, packed molecules) and transforming them |
| Simulation | `simulation/` | calculators; minimization and dynamics (molecular dynamics through other programs); lattice dynamics (elastic constants, force constants, phonons) |
| MLIP | `mlip/` | training data for machine-learned potentials: descriptors and their principal components, selection of frames, training sets from first-principles calculations |
| Visualization | `visualization/` | rendering images of systems; plot templates for results |
| Interoperability | `interop/` | conversions to and from other Python packages, and how files and calculators connect mdapy with simulation programs |
| Performance | `performance/` | timings of mdapy and of other programs |
| Reference | `reference/` | file formats, units and conventions, elements, the glossary, the references and the Python API |
| Development | `development/` | contributing, the source layout and the release notes |

Each feature has its own page, named after it (`eam.md`, `minimize.md`), in
the section of its tab that fits it; closely tied classes, such as an
analysis and its result, share the page. A feature that fits no section goes
to a section **Misc** of its tab until a section for it exists. A new page is
added to the navigation in `mkdocs.yml` and to the map on the home page.

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
   `mp.read`, `mp.System`; the text names the functions in full, as
   `mdapy.read`. Where a result can be seen, a short example after it draws
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
belongs on the feature page. The [Python API](../reference/api.md) page is
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
  stable_30Sep2026, OVITO 3.17.0, freud 3.6.1 and atomsk Beta 0.13.1; freud
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
  modules of mdapy use, and methods that subclasses override, start with one
  underscore: they are internal to mdapy, not to the class. Private
  functions and constants of a module start with one underscore.
- Immutable objects (`System`, `Box`, `Topology`) refuse new attributes and
  return themselves from `copy.copy` and `copy.deepcopy`.
- Commit messages are short English sentences describing the change.
