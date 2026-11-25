# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The public API keeps what a release had.

The snapshot of the newest release in this directory, ``<version>.txt``, lists
each public name, its kind and its parameters. The API of the code must hold
every name of it, of the same kind, with the same parameters in the same order
and with the same defaults; a parameter added has a default and is given by
name. What the snapshot does not hold must be named in the release notes above
the heading of its version.

A release writes its snapshot:

    python tests/api/test_api.py 2.1.0
"""

import dataclasses
import importlib
import inspect
import pkgutil
import re
import sys
import warnings
from pathlib import Path

import pytest

import mdapy
from mdapy.core import Result

HERE = Path(__file__).resolve().parent
NOTES = HERE.parents[1] / "docs" / "about" / "release-notes.md"
#: Public modules beyond the subpackages of mdapy.
MORE_MODULES = (
    "mdapy.calculators.testing",
    "mdapy.properties.defects",
    "mdapy.properties.elastic",
    "mdapy.properties.phonons",
    "mdapy.properties.thermodynamics",
    "mdapy.properties.transport",
    "mdapy.simulation.controls",
    "mdapy.visualization.plot",
)


def modules() -> list[str]:
    """The names of the public modules."""
    found = [
        "mdapy." + m.name
        for m in pkgutil.iter_modules(mdapy.__path__)
        if not m.name.startswith("_")
    ]
    return ["mdapy", *sorted(found + list(MORE_MODULES))]


def public(module) -> list[str]:
    """The public names of a module: its ``__all__``, else the functions and
    classes it defines; and the names that moved from it, and the modules
    whose names all moved, which it still gives, with a warning, until the
    next major release."""
    from mdapy._moved import MOVED

    names = getattr(module, "__all__", None)
    if names is not None:
        kept = [
            m.rpartition(".")[2]
            for m in MOVED
            if m.rpartition(".")[0] == module.__name__
        ]
        return list(
            dict.fromkeys([*names, *sorted(MOVED.get(module.__name__, {})), *kept])
        )
    return [
        name
        for name, value in vars(module).items()
        if not name.startswith("_")
        and getattr(value, "__module__", None) == module.__name__
        and (inspect.isfunction(value) or inspect.isclass(value))
    ]


def parameters(obj) -> list[str]:
    """The parameters as Python writes them: ``name`` or ``name=default``,
    ``*args`` or a bare ``*`` before those given by name only, ``**kwargs``."""
    out, star = [], False
    for p in inspect.signature(obj).parameters.values():
        if p.name in ("self", "cls"):
            continue
        if p.kind == p.VAR_POSITIONAL:
            out.append(f"*{p.name}")
            star = True
        elif p.kind == p.VAR_KEYWORD:
            out.append(f"**{p.name}")
        else:
            if p.kind == p.KEYWORD_ONLY and not star:
                out.append("*")
                star = True
            default = "" if p.default is p.empty else "=" + repr(p.default)
            out.append(p.name + default)
    return out


def members(cls) -> list[tuple[str, str, list[str]]]:
    """The public methods, properties, attributes and fields that mdapy
    defines for a class."""
    fields = (
        {f.name for f in dataclasses.fields(cls)}
        if dataclasses.is_dataclass(cls)
        else ()
    )
    rows, seen = [], set()
    for klass in cls.__mro__:
        if not (klass.__module__ or "").startswith("mdapy"):
            continue
        for name, value in vars(klass).items():
            if name.startswith("_") or name in seen or name in fields:
                continue
            seen.add(name)
            if isinstance(value, property):
                rows.append((name, "property", []))
            elif isinstance(value, (staticmethod, classmethod)):
                rows.append((name, type(value).__name__, parameters(value.__func__)))
            elif inspect.isfunction(value):
                rows.append((name, "method", parameters(value)))
            else:
                rows.append((name, "attribute", []))
    rows += [(name, "field", []) for name in sorted(fields) if not name.startswith("_")]
    return rows


def api() -> dict[str, tuple[str, list[str]]]:
    """Every public name with its kind and parameters."""
    found = {}
    for module_name in modules():
        module = importlib.import_module(module_name)
        for item in public(module):
            with warnings.catch_warnings():  # a name that moved warns
                warnings.simplefilter("ignore", DeprecationWarning)
                obj = getattr(module, item)
            name = f"{module_name}.{item}"
            if inspect.ismodule(obj):
                found[name] = ("module", [])
            elif inspect.isclass(obj) and issubclass(obj, BaseException):
                bases = [b.__name__ for b in obj.__bases__]
                found[name] = ("exception", bases)
            elif inspect.isclass(obj):
                # results are made by mdapy: their fields are the API, not
                # their constructors
                made = issubclass(obj, Result)
                found[name] = ("result", []) if made else ("class", parameters(obj))
                for member, kind, params in members(obj):
                    found[f"{name}.{member}"] = (kind, params)
            elif callable(obj):
                found[name] = ("function", parameters(obj))
            else:
                found[name] = ("constant", [])
    return found


def write(path: Path) -> None:
    """Writes the API to a snapshot, a name to a line, tab-separated."""
    lines = ["\t".join([name, kind, *params]) for name, (kind, params) in api().items()]
    path.write_text("\n".join(sorted(lines)) + "\n", encoding="utf-8")


def read(path: Path) -> dict[str, tuple[str, list[str]]]:
    found = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        name, kind, *params = line.split("\t")
        found[name] = (kind, params)
    return found


def snapshot() -> Path:
    """The snapshot of the newest release."""

    def version(path: Path) -> tuple:
        return tuple(int(part) for part in re.findall(r"\d+", path.stem))

    return max(HERE.glob("*.txt"), key=version)


def parse(params: list[str]) -> list[tuple[str, str, str | None]]:
    """(name, kind, default) of each parameter: kind positional, keyword
    (given by name only), args or kwargs; default None for none."""
    out, star = [], False
    for param in params:
        if param.startswith("**"):
            out.append((param[2:], "kwargs", None))
        elif param.startswith("*"):
            star = True
            if param != "*":
                out.append((param[1:], "args", None))
        else:
            name, sign, default = param.partition("=")
            out.append(
                (name, "keyword" if star else "positional", default if sign else None)
            )
    return out


def changes(old: list[str], new: list[str]) -> tuple[list[str], list[str]]:
    """What breaks code written for the old parameters, and the names of the
    parameters added."""
    old, new = parse(old), parse(new)
    broken, added = [], []
    old_positional = [p for p in old if p[1] == "positional"]
    new_positional = [p for p in new if p[1] == "positional"]
    for k, (name, _, default) in enumerate(old_positional):
        if k >= len(new_positional) or new_positional[k] != old_positional[k]:
            broken.append(f"positional parameter {name}")
    by_name = {name: (kind, default) for name, kind, default in new}
    for name, kind, default in old:
        if kind == "keyword":
            given = by_name.get(name)
            if given is None or given[0] not in ("keyword", "positional"):
                broken.append(f"parameter {name}")
            elif given[1] != default:
                broken.append(f"default of {name}")
        elif kind in ("args", "kwargs") and kind not in {k for _, k, _ in new}:
            broken.append(f"{'*' if kind == 'args' else '**'}{name}")
    old_names = {name for name, _, _ in old}
    for k, (name, kind, default) in enumerate(new):
        if name in old_names:
            continue
        if kind in ("args", "kwargs"):
            added.append(name)
        elif kind == "positional":
            if k >= len(old_positional):  # among the old ones it broke them
                broken.append(f"new parameter {name} not given by name only")
        elif default is None:
            broken.append(f"new parameter {name} without a default")
        else:
            added.append(name)
    return broken, added


CLASSES = ("class", "result", "exception")


def in_module(api: dict, name: str) -> bool:
    """Whether a name is that of a module or of an object of a module, not a
    member of a class."""
    return api.get(name.rpartition(".")[0], ("module",))[0] not in CLASSES


def moves(old: dict, new: dict) -> tuple[dict[str, str], set[str]]:
    """Where the names that left their modules went, and the modules that
    went because all their names moved. A name moved where the new API has
    the only object of its kind and last part in a module; the members of a
    class go with it."""
    places = {}
    for name, (kind, _) in new.items():
        if in_module(new, name):
            places.setdefault((name.rpartition(".")[2], kind), []).append(name)
    moved = {}
    for name, (kind, _) in old.items():
        if name in new or kind == "module" or not in_module(old, name):
            continue
        there = places.get((name.rpartition(".")[2], kind), [])
        if len(there) == 1:
            moved[name] = there[0]
    for name in old:
        owner, _, member = name.rpartition(".")
        if owner in moved and name not in new:
            moved[name] = f"{moved[owner]}.{member}"
    emptied = set()
    for name, (kind, _) in old.items():
        inside = [n for n in old if n.startswith(name + ".")]
        if kind == "module" and name not in new and inside:
            if all(n in moved for n in inside):
                emptied.add(name)
    return moved, emptied


def compare(old: dict, new: dict) -> tuple[list[str], list[str]]:
    """What breaks code written for the old API, and the names added. A name
    that moved to another module is compared at its new place, and counts as
    added there."""
    broken, added = [], []
    moved, emptied = moves(old, new)
    for name, (kind, params) in old.items():
        if name in emptied:
            continue
        place = name if name in new else moved.get(name)
        if place is None:
            broken.append(f"{name} is gone")
            continue
        new_kind, new_params = new[place]
        if new_kind != kind:
            broken.append(f"{name} was a {kind}, is a {new_kind}")
        elif kind == "exception":
            if new_params != params:
                broken.append(f"{name} derives from {new_params}, not {params}")
        else:
            lost, more = changes(params, new_params)
            broken += [f"{name}: {what}" for what in lost]
            if place == name:  # a new name stands for its parameters
                added += [f"{name}({param})" for param in more]
    # the members of a new class come with it, which its own name stands for
    fresh = [name for name in new if name not in old]
    added += [name for name in fresh if name.rpartition(".")[0] not in fresh]
    return broken, added


def section(notes: str, version: str) -> str:
    """The release notes above the heading of a version, or the first
    section when there is no such heading yet."""
    headings = list(re.finditer(r"^## (\S+)", notes, re.M))
    for k, heading in enumerate(headings):
        if heading.group(1) == version:
            return notes[: heading.start()]
    end = headings[1].start() if len(headings) > 1 else len(notes)
    return notes[:end]


def unnamed(added: list[str], text: str) -> list[str]:
    """The additions whose name the text does not hold, as a word."""
    missing = []
    for addition in added:
        word = re.split(r"[.(]", addition.rstrip(")"))[-1]
        if not re.search(rf"\b{re.escape(word)}\b", text):
            missing.append(addition)
    return missing


def test_the_api_keeps_what_the_last_release_had():
    path = snapshot()
    broken, _ = compare(read(path), api())
    assert not broken, (
        f"the API breaks code written for {path.stem}; such changes wait for the "
        "next major release:\n" + "\n".join(broken)
    )


def test_additions_are_in_the_release_notes():
    path = snapshot()
    _, added = compare(read(path), api())
    missing = unnamed(added, section(NOTES.read_text(encoding="utf-8"), path.stem))
    assert not missing, (
        f"name what {path.stem} did not have in the release notes:\n"
        + "\n".join(missing)
    )


def test_the_snapshot_is_sorted_and_complete():
    """A snapshot as write makes it, of names that each have a kind."""
    text = snapshot().read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines == sorted(lines) and text.endswith("\n")
    kinds = {"module", "exception", "result", "class", "function", "constant",
             "method", "property", "attribute", "field", "staticmethod",
             "classmethod"}  # fmt: skip
    assert {kind for kind, _ in read(snapshot()).values()} <= kinds


@pytest.mark.parametrize(
    "old, new, broken, added",
    [
        (["a", "b=1"], ["a", "b=1", "*", "c=2"], [], ["c"]),
        (["a", "b=1"], ["a", "b=2"], ["positional parameter b"], []),
        (["a", "b=1"], ["a", "*", "b=1"], ["positional parameter b"], []),
        (["a", "*", "b=1"], ["a", "b=1"], [], []),
        (["a", "*", "b=1"], ["a", "*", "b=2"], ["default of b"], []),
        (["a", "*", "b=1"], ["a"], ["parameter b"], []),
        (["a"], ["a", "b=1"], ["new parameter b not given by name only"], []),
        (["a"], ["a", "*", "b"], ["new parameter b without a default"], []),
        (
            ["a", "b"],
            ["b", "a"],
            ["positional parameter a", "positional parameter b"],
            [],
        ),
        (
            ["a", "b"],
            ["c", "a", "b"],
            ["positional parameter a", "positional parameter b"],
            [],
        ),
        (["a", "**kw"], ["a"], ["**kw"], []),
        (["a", "*args"], ["a"], ["*args"], []),
        (["a"], ["a", "*args", "**kw"], [], ["args", "kw"]),
        (["a", "*args", "b=1"], ["a", "*args", "b=1", "c=2"], [], ["c"]),
        (["*", "a=1", "b=2"], ["*", "b=2", "a=1"], [], []),
    ],
)
def test_what_breaks(old, new, broken, added):
    assert changes(old, new) == (broken, added)


def test_compare_and_the_notes():
    old = {
        "m.f": ("function", ["a"]),
        "m.C": ("class", []),
        "m.E": ("exception", ["MdapyError"]),
        "m.R.x": ("field", []),
    }
    new = {
        "m.f": ("function", ["a", "*", "seed=None"]),
        "m.C": ("function", []),
        "m.E": ("exception", ["ValueError"]),
        "m.g": ("function", []),
        "m.N": ("result", []),
        "m.N.columns": ("method", []),
    }
    broken, added = compare(old, new)
    assert broken == [
        "m.C was a class, is a function",
        "m.E derives from ['ValueError'], not ['MdapyError']",
        "m.R.x is gone",
    ]
    assert added == ["m.f(seed)", "m.g", "m.N"]  # the members of N come with it
    notes = "# Notes\n\n## 2.1.0\n\n- `g` and `N` are new.\n\n## 2.0.0\n\n- seed\n"
    assert unnamed(added, section(notes, "2.0.0")) == ["m.f(seed)"]
    assert unnamed(added, section(notes, "2.2.0")) == ["m.f(seed)"]
    assert section("## 2.1.0\n\nnew\n", "2.0.0") == "## 2.1.0\n\nnew\n"


def test_a_name_that_moves_is_not_gone():
    old = {
        "m.f": ("function", ["a"]),
        "m.R": ("result", []),
        "m.R.x": ("field", []),
        "m.old": ("module", []),
        "m.old.g": ("function", ["a"]),
        "m.h": ("function", []),
        "m.k": ("function", []),
        "m.empty": ("module", []),
    }
    new = {
        "n.f": ("function", ["a", "*", "seed=None"]),
        "n.R": ("result", []),
        "n.R.x": ("field", []),
        "p.g": ("function", ["b"]),
        "n.k": ("function", []),
        "p.k": ("function", []),
        "q.h": ("method", []),
    }
    broken, added = compare(old, new)
    assert broken == [
        "m.old.g: positional parameter a",  # compared where it moved
        "m.h is gone",  # no function h in a module
        "m.k is gone",  # two of them: which?
        "m.empty is gone",
    ]
    assert added == ["n.f", "n.R", "p.g", "n.k", "p.k", "q.h"]
    assert unnamed(added, "- `f`, `R` and `g` moved to `n`; `k` is new.") == ["q.h"]


if __name__ == "__main__":
    write(HERE / f"{sys.argv[1]}.txt")
