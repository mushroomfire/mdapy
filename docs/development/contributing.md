# Contributing

mdapy 2.0 grows one feature at a time. This page lists the rules every
change follows.

## When a feature is done

A feature is merged only when it has all of the following:

1. the implementation;
2. tests that establish its correctness (see below), meeting the coverage
   thresholds;
3. its own page in the user guide, following the page template below;
4. entries in the [Glossary](../reference/glossary.md) for any new names;
5. a line in the [release notes](../release-notes.md).

## Feature pages

Each feature has a hand-written page with these sections, in this order:

1. **Title and one sentence** saying what the feature does.
2. **Minimal example** that runs as shown. `tests/test_docs.py` runs every
   Python block of every page, in order, with shared variables. A block that
   is only an illustration is preceded by the line `<!-- skip-test -->`.
3. **Parameters**, as a table: `| Name | Type | Default | Unit | Meaning |`.
4. **Outputs**, as a table.
5. **Method**: what is computed, with formulas.
6. **Validation**: what the tests guarantee.
7. **Performance** notes, if relevant.
8. **References**: the papers the method follows.

Sections that do not apply are left out.

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
- Results are not compared against snapshots produced by other analysis
  software.
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

## Style

- `ruff check .` and `ruff format --check .` pass.
- Commit messages are short English sentences describing the change.
