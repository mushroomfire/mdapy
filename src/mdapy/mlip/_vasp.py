# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Frames of VASP calculations: energies, forces and virials from OUTCAR."""

import re
from collections.abc import Sequence
from os import PathLike
from pathlib import Path

import numpy as np

from mdapy.core.box import Box
from mdapy.core.system import System
from mdapy.errors import FileFormatError, InvalidParameter

_FLOAT = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def _floats(line: str) -> list[float]:
    """The numbers of a line, also where VASP prints them without spaces."""
    return [float(x) for x in _FLOAT.findall(line)]


def read_outcar(path: str | PathLike, *, unconverged: bool = False) -> list[System]:
    """The ionic steps of a VASP OUTCAR: a single point, a relaxation or
    molecular dynamics.

    Parameters
    ----------
    path : str or path-like
        The OUTCAR file.
    unconverged : bool, optional
        Keep the steps whose electronic loop did not converge. Default False.

    Returns
    -------
    list of System
        One system per ionic step, with the ``forces`` column in eV/Å and in
        ``info`` the free energy ``energy`` (TOTEN, eV), the ``virial``
        ((3, 3) eV, when VASP wrote the stress) and whether the electronic
        loop ``converged``.
    """
    path = Path(path)
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    symbols: list[str] = []
    counts: list[int] | None = None
    atoms = None
    lattice: np.ndarray | None = None
    virial: np.ndarray | None = None
    converged = False
    frame: tuple[np.ndarray, np.ndarray] | None = None
    frames: list[System] = []
    k = 0
    while k < len(lines):
        line = lines[k]
        if "POTCAR:" in line:
            symbol = line.split()[2].split("_")[0].split(".")[0]
            if symbol not in symbols:
                symbols.append(symbol)
        elif "ions per type" in line:
            counts = [int(x) for x in line.split("=")[1].split()]
        elif "number of ions" in line and "NIONS" in line:
            atoms = int(line.split()[-1])
        elif "direct lattice vectors" in line:
            rows = [_floats(lines[k + 1 + i])[:3] for i in range(3)]
            lattice = np.array(rows, dtype=float)
            k += 3
        elif "aborting loop because EDIFF is reached" in line:
            converged = True
        elif "FORCE on cell =-STRESS" in line:
            j = k + 1
            while j < len(lines) and not lines[j].strip().startswith("Total"):
                j += 1
            if j < len(lines):
                xx, yy, zz, xy, yz, zx = _floats(lines[j])[:6]
                virial = np.array([[xx, xy, zx], [xy, yy, yz], [zx, yz, zz]])
                k = j
        elif "TOTAL-FORCE (eV/Angst)" in line:
            if atoms is None or lattice is None:
                raise FileFormatError(
                    "forces before the number of ions or the lattice", path
                )
            block = [_floats(lines[k + 2 + i]) for i in range(atoms)]
            if any(len(row) < 6 for row in block):
                raise FileFormatError("a line of positions and forces is cut", path)
            values = np.array([row[:6] for row in block])
            frame = (values[:, :3], values[:, 3:])
            k += 1 + atoms
        elif "free  energy   TOTEN" in line and frame is not None:
            energy = _floats(line.split("=")[1])[0]
            if converged or unconverged:
                if counts is None or len(symbols) < len(counts):
                    raise FileFormatError("no elements for the ions per type", path)
                elements = [s for s, c in zip(symbols, counts) for _ in range(c)]
                if len(elements) != len(frame[0]):
                    raise FileFormatError(
                        "the ions per type do not add up to the ions", path
                    )
                info = {"energy": energy, "converged": converged}
                if virial is not None:
                    info["virial"] = virial
                system = System(frame[0], Box(lattice), elements=elements, info=info)
                frames.append(system.with_columns(forces=frame[1]))
            frame, virial, converged = None, None, False
        k += 1
    return frames


def outcar_to_xyz(
    outcars: str | PathLike | Sequence[str | PathLike],
    path: str | PathLike,
    *,
    unconverged: bool = False,
    max_force: float | None = None,
) -> int:
    """Writes the ionic steps of OUTCAR files to an extended XYZ file for
    training potentials.

    Parameters
    ----------
    outcars : str, path-like or sequence of them
        The OUTCAR files.
    path : str or path-like
        The extended XYZ file, with energy, virial and forces.
    unconverged : bool, optional
        Keep steps whose electronic loop did not converge. Default False.
    max_force : float, optional
        Leave out the frames with a force component larger than this, in
        eV/Å.

    Returns
    -------
    int
        Number of frames written.
    """
    from mdapy.io import write

    if isinstance(outcars, str | PathLike):
        outcars = [outcars]
    if max_force is not None and not max_force > 0:
        raise InvalidParameter(f"max_force must be positive, not {max_force!r}")
    frames: list[System] = []
    for outcar in outcars:
        for frame in read_outcar(outcar, unconverged=unconverged):
            forces = frame.table["forces"].to_numpy()
            if max_force is None or np.abs(forces).max(initial=0.0) <= max_force:
                frames.append(frame.with_info(converged=None))
    write(path, frames)
    return len(frames)
