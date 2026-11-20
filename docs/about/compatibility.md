# Compatibility

From 2.0.0 on, stability comes first: what code written for one MDAPY 2.x release uses
stays, with the same results, in every later 2.x release. At most the module a name is
imported from changes, and the old import warns two minor releases before it goes.

## Versions

Version numbers follow [semantic versioning](https://semver.org), major.minor.patch:

| Release | Holds |
|---|---|
| Patch, such as 2.0.1 | bug fixes and documentation |
| Minor, such as 2.1.0 | new features, which leave what exists as it was; a name may move to another module |
| Major, such as 3.0.0 | the only releases that may remove or change what exists |

Pre-releases, such as 2.0.0b1 or 2.0.0rc1, make no promise, and `pip` installs them only
with `--pre`.

## What stays

Within 2.x, these stay as they are:

- every name of the [Python API](../reference/api/index.md): functions, classes, their
  methods and the attributes of their results, though a name may move to another
  module (see below). Names that begin with an underscore are not part of it;
- the parameters of each, their names, their order and their defaults;
- the types, shapes and [units](../reference/units.md) of the results;
- the names of the columns that analyses add to a system;
- the files MDAPY writes: every later 2.x release reads them.

A new parameter comes with a default that keeps the earlier behavior and is given by
name, and a new result is added beside the old ones, never in their place. Results, such
as a `RadialDistributionResult`, are made by MDAPY: their attributes are part of the API,
their constructors are not, so that a later release can add to them.

A test holds MDAPY to this: it compares the API with that of each release and fails on
a name that went away, from its module and from every other, on a parameter that
changed, and on an addition or a move the release notes do not name.

## When something has to change

- A name that is replaced keeps working until 3.0. From the release that replaces it on,
  using it warns with a `DeprecationWarning` that names what to use instead.
- A name that moves to another module, so that the modules hold what belongs together,
  keeps its parameters and its results: only its import changes. Its old import keeps
  working in the minor release that moves it and in the next, with a
  `DeprecationWarning` that names the new one, and goes two minor releases after the
  move: the names that moved to `mdapy.properties` in 2.2 are found at their old places
  in 2.2 and 2.3, not in 2.4. The release notes list every move.
- A result changes only when it was wrong. The release notes name every result that a
  fix changes.

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
