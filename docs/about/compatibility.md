# Compatibility

From 2.0.0 on, stability comes first: code written for one MDAPY 2.x release keeps
working, with the same results, in every later 2.x release.

## Versions

Version numbers follow [semantic versioning](https://semver.org), major.minor.patch:

| Release | Holds |
|---|---|
| Patch, such as 2.0.1 | bug fixes and documentation |
| Minor, such as 2.1.0 | new features, which leave what exists as it was |
| Major, such as 3.0.0 | the only releases that may remove or change what exists |

Pre-releases, such as 2.0.0b1 or 2.0.0rc1, make no promise, and `pip` installs them only
with `--pre`.

## What stays

Within 2.x, these stay as they are:

- every name of the [Python API](../reference/api/index.md): functions, classes, their
  methods and the attributes of their results. Names that begin with an underscore are
  not part of it;
- the parameters of each, their names, their order and their defaults;
- the types, shapes and [units](../reference/units.md) of the results;
- the names of the columns that analyses add to a system;
- the files MDAPY writes: every later 2.x release reads them.

A new parameter comes with a default that keeps the earlier behavior, and a new result is
added beside the old ones, never in their place.

## When something has to change

- A name that is replaced keeps working until 3.0. From the release that replaces it on,
  using it warns with a `DeprecationWarning` that names what to use instead.
- A result changes only when it was wrong. The release notes name every result that a
  fix changes.
- Experimental features, marked by a box at the top of their page, such as the
  [dislocation analysis](../user-guide/analysis/dislocations.md), may still change in a
  minor release, and the release notes say how.

## Python, NumPy and polars

Each release supports a version of Python for at least three years after it came out.
The lowest versions of Python, NumPy and polars that MDAPY accepts rise only in a minor
release, and the release notes say so. Every change is tested with the lowest and the
newest versions it accepts, and every week with what NumPy and polars are about to
release (see [Contributing](contributing.md#tests)).

## MDAPY 1.x

MDAPY 1.x is no longer developed. Its code stays on the branch
[`1.x`](https://github.com/mushroomfire/mdapy/tree/1.x) and its releases stay on PyPI:

```bash
pip install "mdapy<2"
```
