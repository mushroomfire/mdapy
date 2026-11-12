# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Force constants of a crystal in a supercell, and the FORCE_CONSTANTS files
of phonopy."""

import os
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from mdapy._core import neighbor as _search
from mdapy.core.system import System
from mdapy.errors import FileFormatError, InvalidParameter
from mdapy.lattice_dynamics._phonons import (
    Phonons,
    _kinds,
    _lattice_points,
    _masses,
    _sheet_supercell,
    _supercell,
    _supercell_matrix,
)
from mdapy.lattice_dynamics._sheets import _is_sheet

# how far in Å an atom may be from the image of another under a translation
# of the crystal
_TOLERANCE = 1e-3


@dataclass(frozen=True, eq=False, repr=False)
class ForceConstants:
    """Second order force constants of a crystal in a supercell.

    Parameters
    ----------
    cell : System
        The cell of the crystal, periodic in every direction, or of a sheet,
        periodic along two vectors in the x-y plane and not along a third
        along z.
    supercell : int32 or array_like
        Repeats of the cell along its vectors, one or three, or a (3, 3)
        integer matrix whose rows times the vectors of the cell are the
        vectors of the supercell; a sheet repeats along its plane only.
    second : array_like of float64
        (n, n S, 3, 3) second derivatives of the energy in eV/Å² with respect
        to the positions of each atom of the cell and of each atom of the
        supercell, which holds S cells, each with the atoms of the cell in
        their order, the first cell that of the cell itself, as the
        ``force_constants`` of ``Phonons``.

    Attributes
    ----------
    cell : System
        The cell of the crystal.
    supercell : numpy.ndarray
        (3, 3) int64 matrix of the supercell; read-only.
    second : numpy.ndarray
        (n, n S, 3, 3) float64 second order force constants in eV/Å²;
        read-only.

    Raises
    ------
    InvalidParameter
        If the box is neither periodic nor that of a sheet, there are no
        atoms, the supercell is not positive integers or a matrix of them
        with a positive determinant, or second is not of the shape of the
        cell and the supercell or not finite.
    """

    cell: System
    supercell: np.ndarray
    second: np.ndarray

    def __post_init__(self):
        matrix = _matrix(self.cell, self.supercell)
        n, size = len(self.cell), round(np.linalg.det(matrix))
        second = np.array(self.second, dtype=float)
        if second.shape != (n, n * size, 3, 3):
            raise InvalidParameter(
                f"second must be of shape {(n, n * size, 3, 3)} for {n} atoms in "
                f"a supercell of {size} cells, not {second.shape}"
            )
        if not np.isfinite(second).all():
            raise InvalidParameter("second must be finite")
        matrix.flags.writeable = False
        second.flags.writeable = False
        object.__setattr__(self, "supercell", matrix)
        object.__setattr__(self, "second", second)

    def __repr__(self) -> str:
        cells = round(np.linalg.det(self.supercell))
        atoms = f"{len(self.cell)} atom" + ("s" if len(self.cell) > 1 else "")
        return f"ForceConstants({atoms}, supercell of {cells} cells)"

    def phonons(self) -> Phonons:
        """The harmonic phonons of these force constants.

        Returns
        -------
        Phonons
            Frequencies, polarization vectors, band structures, densities of
            states and thermal properties, as ``phonons`` gives them.

        Raises
        ------
        InvalidParameter
            If an atom of the cell has neither element nor mass.
        """
        return Phonons(self.cell, self.supercell, self.second, _masses(self.cell))

    def write(self, path: str | os.PathLike, *, full: bool = False) -> None:
        """Writes the force constants in the format of phonopy's
        FORCE_CONSTANTS, in its order of the atoms of the supercell.

        Parameters
        ----------
        path : str or path-like
            The file.
        full : bool, optional
            Write the force constants of every atom of the supercell, which
            phonopy reads whatever its primitive cell; otherwise those of the
            atoms of the cell, which phonopy reads when its primitive cell is
            the cell. Default False.
        """
        n, matrix = len(self.cell), self.supercell
        size = round(np.linalg.det(matrix))
        ours = _phonopy_atoms(matrix, n)  # mdapy's atom of each of phonopy's
        if full:
            repeats = _lattice_points(matrix)
            # the cell of each atom of the supercell, seen from each cell
            apart = _cell_of(repeats[None, :, :] - repeats[:, None, :], matrix)
            columns = (apart[:, :, None] * n + np.arange(n)).reshape(size, -1)
            rows = ours // n, ours % n  # the cell and the atom of each row
            constants = self.second[rows[1][:, None], columns[rows[0]][:, ours]]
            labels = np.arange(n * size)
        else:
            constants = self.second[:, ours]
            labels = np.arange(n) * size
        lines = [f"{len(labels):4d} {n * size:4d}"]
        for label, block in zip(labels, constants):
            for j, tensor in enumerate(block):
                lines.append(f"{label + 1} {j + 1}")
                lines.extend(
                    f"{x:22.15f}{y:22.15f}{z:22.15f}" for x, y, z in tensor.tolist()
                )
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")


def _matrix(cell: System, supercell: int | ArrayLike) -> np.ndarray:
    """The (3, 3) int64 matrix of a supercell of the cell."""
    if len(cell) == 0:
        raise InvalidParameter(
            "force constants need a box periodic in every direction with atoms"
        )
    if _is_sheet(cell.box, "force constants"):
        return _sheet_supercell(supercell).astype(np.int64)
    return _supercell_matrix(supercell).astype(np.int64)


def _cell_of(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """The index in ``_lattice_points(matrix)`` of the cell that each lattice
    point, in units of the cell, is a copy of in the supercell."""
    repeats = _lattice_points(matrix)
    fractional = points @ np.linalg.inv(matrix)
    inside = np.round((fractional - np.floor(fractional + 1e-9)) @ matrix).astype(int)
    span = np.ptp(repeats, axis=0) + 1
    low = repeats.min(axis=0)

    def key(p: np.ndarray) -> np.ndarray:
        return ((p[..., 0] - low[0]) * span[1] + p[..., 1] - low[1]) * span[2] + (
            p[..., 2] - low[2]
        )

    table = np.full(int(np.prod(span)), -1, dtype=np.int64)
    table[key(repeats)] = np.arange(len(repeats))
    return table[key(inside)]


def _phonopy_atoms(matrix: np.ndarray, n: int) -> np.ndarray:
    """(n S,) the atom of mdapy's supercell that is each atom of phonopy's:
    phonopy orders the atoms by the atom of the cell, and the cells by the
    lattice points of the smallest box of whole cells around the supercell,
    the first vector fastest, each copy of a cell in the supercell where it
    first comes."""
    corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    frame = np.ptp(corners @ matrix, axis=0)
    c, b, a = np.meshgrid(*(np.arange(f) for f in frame[::-1]), indexing="ij")
    cells = _cell_of(np.c_[a.ravel(), b.ravel(), c.ravel()], matrix)
    _, first = np.unique(cells, return_index=True)
    order = cells[np.sort(first)]
    return (order[None, :] * n + np.arange(n)[:, None]).reshape(-1)


def _translated(big: System, shift: np.ndarray, kinds: np.ndarray) -> np.ndarray | None:
    """(N,) the atom of the supercell at each atom moved by shift, or None
    if the shift does not map the crystal onto itself."""
    box = big.box
    positions = np.ascontiguousarray(box.wrap(big["positions"])[0])
    tree = _search.KdTree(
        positions,
        np.ascontiguousarray(box.matrix),
        np.ascontiguousarray(box.origin),
        np.ascontiguousarray(box.pbc),
    )
    image, _, distance = tree.nearest(np.ascontiguousarray(positions + shift), 1)
    image = image[:, 0].astype(np.int64)
    if distance.max() > _TOLERANCE or (kinds[image] != kinds).any():
        return None
    return image


def read_force_constants(
    path: str | os.PathLike, cell: System, supercell: int | ArrayLike
) -> ForceConstants:
    """Force constants from a file in the format of phonopy's
    FORCE_CONSTANTS.

    Parameters
    ----------
    path : str or path-like
        The file, in eV/Å², such as one phonopy wrote from a calculation with
        VASP.
    cell : System
        The unit cell of phonopy's calculation, its atoms in its order.
    supercell : int32 or array_like
        Repeats of the cell along its vectors, one or three, or a (3, 3)
        integer matrix whose rows times the vectors of the cell are the
        vectors of the supercell: the transpose of phonopy's supercell
        matrix, the same for repeats.

    Returns
    -------
    ForceConstants
        The force constants of the atoms of the cell, those of atoms that the
        file leaves out found from atoms that a translation of the crystal
        takes to them.

    Raises
    ------
    FileFormatError
        If the file is not in that format.
    InvalidParameter
        If the file holds the force constants of another number of atoms
        than the supercell has, or neither those of an atom of the cell nor
        those of an atom that a translation of the crystal takes to it.
    """
    with open(path) as f:
        lines = [line for line in f.read().splitlines() if line.strip()]
    try:
        shape = [int(x) for x in lines[0].split()]
        if len(shape) == 1:
            shape *= 2
        rows, columns = shape
        if rows < 1 or columns < 1 or len(lines) < 1 + 4 * rows * columns:
            raise ValueError
        body = lines[1 : 1 + 4 * rows * columns]
        # the first number of the first line of each block is the atom of the
        # row, as phonopy reads it; the rest of that line is left alone
        labels = np.array(
            [line.split()[0] for line in body[::4]], dtype=np.int64
        ).reshape(rows, columns)
        values = np.array(
            " ".join(line for k, line in enumerate(body) if k % 4).split(), dtype=float
        ).reshape(rows, columns, 3, 3)
    except ValueError:
        raise FileFormatError("not a FORCE_CONSTANTS of phonopy", str(path)) from None
    if (labels != labels[:, :1]).any():
        raise FileFormatError("not a FORCE_CONSTANTS of phonopy", str(path))
    matrix = _matrix(cell, supercell)
    n, size = len(cell), round(np.linalg.det(matrix))
    if columns != n * size:
        raise InvalidParameter(
            f"the file holds force constants of {columns} atoms; the supercell "
            f"of {size} cells of {n} atoms has {n * size}"
        )
    ours = _phonopy_atoms(matrix, n)
    given = {}  # mdapy's atom of each row, and its force constants
    for label, block in zip(labels[:, 0], values):
        if not 1 <= label <= columns:
            raise FileFormatError(
                f"atom {label} of the file is not in the supercell", str(path)
            )
        row = np.empty_like(block)
        row[ours] = block
        given[int(ours[label - 1])] = row
    second = np.empty((n, n * size, 3, 3))
    big, kinds = None, None
    for a in range(n):
        if a in given:
            second[a] = given[a]
            continue
        if big is None:
            big = _supercell(cell, matrix)
            kinds = np.tile(_kinds(cell), size)
        for p, row in given.items():
            if kinds[p] != kinds[a]:
                continue
            image = _translated(big, big["positions"][a] - big["positions"][p], kinds)
            if image is not None:
                second[a, image] = row
                break
        else:
            raise InvalidParameter(
                f"the file holds no force constants of atom {a} of the cell, nor "
                f"of an atom that a translation of the crystal takes to it"
            )
    return ForceConstants(cell, matrix, second)
