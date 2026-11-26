# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The ground state of a computation of the DFT in a file, NumPy's .npz: what
DFT.save writes and DFT(start=...) reads, with the checks that a computation
is of the structure, the pseudopotentials and the grid of the file."""

import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from mdapy.errors import InvalidParameter

FORMAT = "MDAPY DFT ground state"
VERSION = 1

# the arrays of every file, after format, version and parameters
_ARRAYS = (
    "elements", "cell", "positions", "energy", "fermi", "forces", "density",
    "kpoints", "weights", "eigenvalues", "occupations",
)  # fmt: skip
# those of some: the occupations of the projectors of PAW or DFT+U, and the
# moments of the atoms with spin
_OPTIONAL = ("becsum", "magnetic_moments")

# how far the cell and the atoms may be from those of the file, Å
_TOLERANCE = 1e-6


def write(path: str | os.PathLike, ground: dict[str, Any]) -> None:
    """Writes a ground state, the file whole or not at all."""
    arrays = {
        "format": np.array(FORMAT),
        "version": np.array(VERSION, dtype=np.int64),
        "parameters": np.array(json.dumps(ground["parameters"])),
        "elements": np.array(ground["elements"], dtype=str),
        "energy": np.array(ground["energy"], dtype=float),
        "fermi": np.array(ground["fermi"], dtype=float),
    }
    for name in _ARRAYS + _OPTIONAL:
        if name not in arrays and name in ground:
            arrays[name] = np.asarray(ground[name])
    target = Path(path)
    part = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=f".{target.name}.", delete=False
        ) as part:
            np.savez_compressed(part, **arrays)
        os.replace(part.name, target)
    except OSError as error:
        if part is not None and os.path.exists(part.name):
            os.unlink(part.name)
        raise InvalidParameter(f"cannot write {os.fspath(path)}: {error}") from None


def read(path: str | os.PathLike) -> dict[str, Any]:
    """The ground state of a file that write wrote: its arrays, the
    parameters as a dict and the elements as a list."""
    what = f"{os.fspath(path)} is not a ground state of MDAPY's DFT"
    try:
        loaded = np.load(path, allow_pickle=False)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise InvalidParameter(f"{what}: {error}") from None
    if not isinstance(loaded, np.lib.npyio.NpzFile):
        raise InvalidParameter(f"{what}: one array, not a .npz file")
    with loaded:
        contents = {name: loaded[name] for name in loaded.files}
    if "format" not in contents or str(contents["format"]) != FORMAT:
        raise InvalidParameter(what)
    version = int(contents.get("version", 0))
    if version > VERSION:
        raise InvalidParameter(
            f"{os.fspath(path)} is of version {version} of the file of a ground state, "
            f"from a later MDAPY; this one reads version {VERSION}"
        )
    missing = [n for n in ("parameters",) + _ARRAYS if n not in contents]
    if missing:
        raise InvalidParameter(f"{what}: it lacks {', '.join(missing)}")
    ground = {n: contents[n] for n in _ARRAYS + _OPTIONAL if n in contents}
    ground["parameters"] = json.loads(str(contents["parameters"]))
    ground["elements"] = [str(e) for e in contents["elements"]]
    ground["energy"] = float(ground["energy"])
    ground["fermi"] = float(ground["fermi"])
    return ground


def check(
    ground: dict[str, Any],
    path: Any,
    elements: list[str],
    cell: np.ndarray,
    positions: np.ndarray,
    digests: dict[str, str],
) -> None:
    """Raises unless a computation is of the atoms, the cell and the
    pseudopotentials (by their SHA-256) of a ground state."""
    where = f"the ground state of {os.fspath(path)}"
    if list(elements) != ground["elements"]:
        raise InvalidParameter(f"{where} is of other atoms, or in another order")
    cell = np.asarray(cell, dtype=float)
    if not np.allclose(ground["cell"], cell, rtol=0.0, atol=_TOLERANCE):
        raise InvalidParameter(f"{where} is of another cell")
    shift = (np.asarray(positions, dtype=float) - ground["positions"]) @ np.linalg.inv(
        cell
    )
    shift -= np.round(shift)
    if np.linalg.norm(shift @ cell, axis=1).max(initial=0.0) > _TOLERANCE:
        raise InvalidParameter(f"{where} has its atoms elsewhere")
    taken = ground["parameters"].get("pseudopotentials", {})
    for element, digest in digests.items():
        if taken.get(element, {}).get("sha256") != digest:
            raise InvalidParameter(
                f"{where} was computed with another pseudopotential of {element}"
            )


def check_grid(ground: dict[str, Any], path: Any, spins: int, grid) -> None:
    """Raises unless the density of a ground state has the spins and the grid
    of a computation."""
    expected = (int(spins), *map(int, grid))
    shape = tuple(ground["density"].shape)
    if shape != expected:
        raise InvalidParameter(
            f"the ground state of {os.fspath(path)} has {shape[0]} spin channels on "
            f"the grid {' '.join(map(str, shape[1:]))}, this computation {spins} on "
            f"{' '.join(map(str, grid))}: the same cutoffs, symmetry and spin give "
            "the same grid"
        )
