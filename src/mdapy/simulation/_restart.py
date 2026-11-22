# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Restart files of molecular dynamics: the state of a run, written so that
a crash never leaves half a file, and read back with the calculators given
again. A pickle of MDAPY's own objects, NumPy arrays and polars tables, read
by an unpickler that refuses anything else, and only by the version of MDAPY
that wrote it."""

import io
import os
import pickle
from typing import Any

import mdapy
from mdapy.calculators._base import Calculator
from mdapy.errors import InvalidParameter

_MAGIC = b"MDAPY restart\n"
_FORMAT = 1
# the modules whose classes a restart file may hold; builtins only these
_MODULES = ("mdapy.", "numpy", "polars")
_BUILTINS = {"complex", "frozenset", "set", "slice", "range", "tuple", "list", "dict"}


class _Writer(pickle.Pickler):
    """Writes the calculators as places for those given back on reading."""

    def __init__(self, stream: Any):
        super().__init__(stream, protocol=5)
        self.calculators: list[Calculator] = []

    def persistent_id(self, obj: Any) -> Any:
        if isinstance(obj, Calculator):
            for k, known in enumerate(self.calculators):
                if known is obj:
                    return ("calculator", k)
            self.calculators.append(obj)
            return ("calculator", len(self.calculators) - 1)
        return None


class _Reader(pickle.Unpickler):
    """Reads MDAPY's objects, NumPy and polars only, the calculators given."""

    def __init__(self, stream: Any, calculators: list[Calculator]):
        super().__init__(stream)
        self.calculators = calculators

    def find_class(self, module: str, name: str) -> Any:
        allowed = module.startswith(_MODULES) or (
            module == "builtins" and name in _BUILTINS
        )
        if module in ("copyreg", "collections") or allowed:
            return super().find_class(module, name)
        raise InvalidParameter(f"a restart file holds no {module}.{name}")

    def persistent_load(self, pid: Any) -> Any:
        kind, k = pid
        if kind != "calculator" or k >= len(self.calculators):
            raise InvalidParameter(
                f"this restart file needs {k + 1} calculators: the potential and "
                "the others its runs use, in others"
            )
        return self.calculators[k]


def write(
    path: str | os.PathLike, state: dict[str, Any], *, first: Any = None
) -> list[str]:
    """Writes a state to a file, replaced only once the new one is whole;
    gives the descriptions of the calculators it refers to, in order, first
    the potential of the dynamics."""
    buffer = io.BytesIO()
    writer = _Writer(buffer)
    if isinstance(first, Calculator):
        writer.calculators.append(first)
    writer.dump(state)
    names = [repr(c) for c in writer.calculators]
    head = pickle.dumps(
        {"version": mdapy.__version__, "format": _FORMAT, "calculators": names}, 5
    )
    path = os.fspath(path)
    temporary = path + ".writing"
    try:
        with open(temporary, "wb") as stream:
            stream.write(_MAGIC)
            stream.write(len(head).to_bytes(8, "little"))
            stream.write(head)
            stream.write(buffer.getbuffer())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError as error:
        raise InvalidParameter(f"cannot write the restart file: {error}") from None
    return names


def read(
    path: str | os.PathLike, calculators: list[Calculator]
) -> tuple[dict[str, Any], list[str]]:
    """The state of a restart file, with the calculators given in the order
    it refers to them, and the descriptions of those it was written with."""
    try:
        with open(path, "rb") as stream:
            data = stream.read()
    except OSError as error:
        raise InvalidParameter(f"cannot read the restart file: {error}") from None
    if not data.startswith(_MAGIC):
        raise InvalidParameter(f"{os.fspath(path)} is not a restart file of MDAPY")
    at = len(_MAGIC)
    size = int.from_bytes(data[at : at + 8], "little")
    head = _Reader(io.BytesIO(data[at + 8 : at + 8 + size]), []).load()
    if head.get("version") != mdapy.__version__ or head.get("format") != _FORMAT:
        raise InvalidParameter(
            f"this restart file was written by MDAPY {head.get('version')}; MDAPY "
            f"{mdapy.__version__} reads only its own"
        )
    needed = head["calculators"]
    if len(calculators) < len(needed):
        raise InvalidParameter(
            f"this restart file needs {len(needed)} calculators, {needed}: the "
            "potential and the others its runs use, in others"
        )
    body = io.BytesIO(data[at + 8 + size :])
    return _Reader(body, calculators).load(), needed
