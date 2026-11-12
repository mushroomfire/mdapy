# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Bond stiffness against bond length: springs along and across the bonds of
crystals, whose force constants give phonons of any arrangement of atoms."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from mdapy.core.system import System
from mdapy.errors import InvalidParameter
from mdapy.lattice_dynamics._force_constants import (
    _SHELL,
    ForceConstants,
    _cell_of,
    _design,
    _matched,
    _matrix,
    _pair_table,
    _supercell_of,
)


def _elements(cell: System, what: str) -> list[str]:
    if "element" not in cell.table.columns:
        raise InvalidParameter(f"{what} needs an element column")
    return cell.table["element"].cast(str).to_list()


def _bonds(
    cell: System, matrix: np.ndarray, shells: np.ndarray, what: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[str, str, int]]]:
    """The bonds of the cell no longer than the last shell, the first atom in
    the cell: their row in the layout of ``_basis``, their vectors in Å, and
    the two elements of each, in alphabetical order, with its shell."""
    elements = _elements(cell, "bond stiffness")
    n = len(cell)
    count = n * round(np.linalg.det(matrix))
    a, b, lattice = _pair_table(cell, matrix, float(shells[-1]), what)
    vectors = lattice @ cell.box.matrix + cell["positions"][b] - cell["positions"][a]
    lengths = np.linalg.norm(vectors, axis=1)
    bond = lengths > 1e-9  # not an atom with itself
    a, b, lattice, vectors = a[bond], b[bond], lattice[bond], vectors[bond]
    shell = np.searchsorted(shells + _SHELL, lengths[bond])
    rows = a * count + _cell_of(lattice, matrix) * n + b
    names = [
        (*sorted((elements[i], elements[j])), int(s)) for i, j, s in zip(a, b, shell)
    ]
    return rows, a, vectors, names


def _springs(
    rows: np.ndarray,
    first: np.ndarray,
    vectors: np.ndarray,
    longitudinal: np.ndarray,
    transverse: np.ndarray,
    n: int,
    count: int,
) -> np.ndarray:
    """(n N, 9, K) the force constants of springs along and across bonds,
    the stiffnesses of each bond (B, K), with those of each atom with itself
    that make the sum over its bonds zero."""
    lengths = np.linalg.norm(vectors, axis=1)
    unit = vectors / lengths[:, None]
    along = np.einsum("ba,bc->bac", unit, unit).reshape(-1, 9)
    across = np.eye(3).reshape(1, 9) - along
    blocks = -(
        along[:, :, None] * longitudinal[:, None, :]
        + across[:, :, None] * transverse[:, None, :]
    )
    out = np.zeros((n * count, 9, blocks.shape[2]))
    out[rows] = blocks
    selves = np.arange(n) * count + np.arange(n)  # each atom of the cell
    np.subtract.at(out, selves[first], blocks)
    return out


def _shells(cutoff: float | ArrayLike) -> np.ndarray:
    """(M,) the outer radii of the shells, increasing."""
    shells = np.atleast_1d(np.asarray(cutoff, dtype=float))
    if (
        shells.ndim != 1
        or len(shells) == 0
        or not np.isfinite(shells).all()
        or not shells[0] > 0
        or (np.diff(shells) <= 0).any()
    ):
        raise InvalidParameter(
            "cutoff must be a positive length or increasing positive lengths, "
            f"not {cutoff!r}"
        )
    return shells


@dataclass(frozen=True, eq=False, repr=False)
class BondStiffness:
    """Springs along and across the bonds of crystals, whose stiffnesses are
    polynomials of the length of the bond, one for each pair of elements in
    each shell of neighbors.

    Made by ``bond_stiffness``.

    Attributes
    ----------
    pairs : tuple of (str, str, int64)
        The two elements, in alphabetical order, and the shell from 0, of
        each set of bonds.
    shells : numpy.ndarray
        (M,) float64 the outer radius of each shell in Å, increasing: a bond
        of length r is of the first shell whose radius is r or more; the
        last is the longest bond. Read-only.
    degree : int64
        The degree of the polynomials.
    longitudinal : numpy.ndarray
        (P, degree + 1) float64 coefficients of the stiffness along the bonds
        of each pair in eV/Å², of the powers of their length in Å from the
        lowest; read-only.
    transverse : numpy.ndarray
        (P, degree + 1) float64 coefficients of the stiffness across them,
        the same way; read-only.
    lengths : numpy.ndarray
        (P, 2) float64 the shortest and the longest bond of each pair in the
        fit, in Å; read-only.
    error : float64
        The relative error of the forces: the norm of the forces less those
        of the springs over the norm of the forces.
    """

    pairs: tuple[tuple[str, str, int], ...]
    shells: np.ndarray
    degree: int
    longitudinal: np.ndarray
    transverse: np.ndarray
    lengths: np.ndarray
    error: float

    def __post_init__(self):
        for array in (self.shells, self.longitudinal, self.transverse, self.lengths):
            array.flags.writeable = False

    def __repr__(self) -> str:
        names = ", ".join(sorted({f"{a}-{b}" for a, b, _ in self.pairs}))
        count = len(self.shells)
        shells = f"{count} shell{'s' if count > 1 else ''}"
        return (
            f"BondStiffness({names}; {shells} up to {self.shells[-1]:g} Å, degree "
            f"{self.degree})"
        )

    def stiffness(
        self, first: str, second: str, length: ArrayLike
    ) -> tuple[np.ndarray, np.ndarray]:
        """The stiffness along and across bonds of two elements.

        Parameters
        ----------
        first, second : str
            The elements, in any order.
        length : array_like of float64
            The lengths of the bonds in Å, each in a shell.

        Returns
        -------
        longitudinal : numpy.ndarray
            float64 stiffness along each bond in eV/Å², of the shape of
            length.
        transverse : numpy.ndarray
            float64 stiffness across each bond in eV/Å².

        Raises
        ------
        InvalidParameter
            If a length is not in a shell, or the elements have no bonds in
            its shell.
        """
        pair = tuple(sorted((str(first), str(second))))
        length = np.asarray(length, dtype=float)
        shell = np.searchsorted(self.shells + _SHELL, length)
        if (shell >= len(self.shells)).any() or not (length > 0).all():
            raise InvalidParameter(
                f"bonds are no longer than {self.shells[-1]:g} Å and not of no length"
            )
        rows = np.empty(length.shape, dtype=np.int64)
        for s in np.unique(shell):
            if (*pair, int(s)) not in self.pairs:
                raise InvalidParameter(
                    f"no stiffness of {pair[0]}-{pair[1]} bonds in shell {s}"
                )
            rows[shell == s] = self.pairs.index((*pair, int(s)))
        powers = length[..., None] ** np.arange(self.degree + 1)
        return (
            (powers * self.longitudinal[rows]).sum(axis=-1),
            (powers * self.transverse[rows]).sum(axis=-1),
        )

    def force_constants(
        self, cell: System, supercell: int | ArrayLike = 1
    ) -> ForceConstants:
        """The force constants of the springs of the bonds of a crystal.

        Parameters
        ----------
        cell : System
            The cell of the crystal, periodic in every direction, or of a
            sheet, with an element column, its atoms where they rest, such as
            a special quasirandom structure relaxed.
        supercell : int32 or array_like, optional
            Repeats of the cell along its vectors, one or three, or a (3, 3)
            integer matrix whose rows times the vectors of the cell are the
            vectors of the supercell, in which half the shortest vector is
            longer than the last shell. Default 1, the cell itself.

        Returns
        -------
        ForceConstants
            The force constants, zero beyond the last shell, which is their
            cutoff.

        Raises
        ------
        InvalidParameter
            If the cell has no element column, a bond is of a pair of
            elements and a shell that the springs do not hold, or the
            supercell is not valid or too small for the shells.
        """
        matrix = _matrix(cell, supercell)
        rows, first, vectors, names = _bonds(
            cell, matrix, np.asarray(self.shells), "the last shell"
        )
        lengths = np.linalg.norm(vectors, axis=1)
        longitudinal = np.empty(len(rows))
        transverse = np.empty(len(rows))
        for name in sorted(set(names)):
            mine = np.array([other == name for other in names])
            longitudinal[mine], transverse[mine] = self.stiffness(
                name[0], name[1], lengths[mine]
            )
        n, count = len(cell), len(cell) * round(np.linalg.det(matrix))
        second = _springs(
            rows, first, vectors, longitudinal[:, None], transverse[:, None], n, count
        )
        result = ForceConstants(cell, matrix, second.reshape(n, count, 3, 3))
        object.__setattr__(result, "cutoff", float(self.shells[-1]))
        return result


def bond_stiffness(
    data: Sequence[tuple[System, Sequence[System]]],
    *,
    cutoff: float | ArrayLike,
    degree: int = 1,
) -> BondStiffness:
    """Springs along and across the bonds of crystals, whose stiffnesses are
    polynomials of the length of the bond for each pair of elements in each
    shell of neighbors, fitted to the forces on atoms displaced in
    supercells.

    Parameters
    ----------
    data : sequence of (System, sequence of System)
        For each crystal, its cell, periodic in every direction or a sheet,
        with an element column and its atoms where they rest, and supercells
        of it with atoms displaced, such as those of ``displaced_supercells``,
        each with a ``forces`` column in eV/Å, its atoms in any order. Cells
        of other sizes, such as strained ones, give bonds of other lengths.
    cutoff : float64 or array_like of float64
        The longest bond in Å, or the outer radii of shells of neighbors,
        increasing, each shell with springs of its own; the last at most half
        the shortest vector of each supercell less 1e-3 Å.
    degree : int32, optional
        The degree of the polynomials of the length. Default 1.

    Returns
    -------
    BondStiffness
        The stiffnesses, with the range of the lengths of the bonds and the
        relative error of the forces.

    Raises
    ------
    InvalidParameter
        If there is no crystal, a crystal has no frames, a cell has no
        element column or no atoms, a frame is not a supercell of its cell
        with forces, its atoms one at each atom of the supercell, the cutoff
        is not positive lengths, increasing, or too long for a supercell,
        there are no bonds, the degree is not an integer from 0, or the bonds
        of a pair have fewer lengths than the degree needs.
    """
    data = [(cell, list(frames)) for cell, frames in data]
    if not data:
        raise InvalidParameter("bond stiffness needs crystals with frames")
    if (
        isinstance(degree, bool)
        or not isinstance(degree, (int, np.integer))
        or degree < 0
    ):
        raise InvalidParameter(f"degree must be an integer from 0, not {degree!r}")
    degree, shells = int(degree), _shells(cutoff)
    found = []
    for cell, frames in data:
        if len(cell) == 0:
            raise InvalidParameter("bond stiffness needs cells with atoms")
        if not frames:
            raise InvalidParameter("bond stiffness needs frames of each crystal")
        matrix = _supercell_of(cell, frames)
        found.append((matrix, *_bonds(cell, matrix, shells, "the last shell")))
    pairs = tuple(sorted({name for *_, names in found for name in names}))
    if not pairs:
        raise InvalidParameter(f"no bonds of {shells[-1]:g} Å or shorter")
    powers = degree + 1
    parameters = 2 * len(pairs) * powers
    lengths = np.full((len(pairs), 2), [np.inf, -np.inf])
    seen = [set() for _ in pairs]
    designs, forces = [], []
    for (cell, frames), (matrix, rows, first, vectors, names) in zip(data, found):
        n, count = len(cell), len(cell) * round(np.linalg.det(matrix))
        r = np.linalg.norm(vectors, axis=1)
        index = np.array([pairs.index(name) for name in names])
        # the columns: for each pair, the powers along, then across
        bonds = np.arange(len(r))
        longitudinal = np.zeros((len(r), parameters))
        transverse = np.zeros((len(r), parameters))
        for q in range(powers):
            longitudinal[bonds, index * 2 * powers + q] = r**q
            transverse[bonds, index * 2 * powers + powers + q] = r**q
        basis = _springs(rows, first, vectors, longitudinal, transverse, n, count)
        design, values = _design(
            cell, matrix, basis.reshape(-1, parameters), _matched(cell, matrix, frames)
        )
        designs.append(design)
        forces.append(values)
        for k in np.unique(index):
            mine = r[index == k]
            lengths[k] = min(lengths[k, 0], mine.min()), max(lengths[k, 1], mine.max())
            seen[k].update(np.round(mine, 6).tolist())
    for k, (a, b, shell) in enumerate(pairs):
        if len(seen[k]) < powers:
            raise InvalidParameter(
                f"the {a}-{b} bonds of shell {shell} have {len(seen[k])} length"
                f"{'s' if len(seen[k]) != 1 else ''}, a degree of {degree} needs "
                f"{powers}: add cells of other sizes, or lower the degree"
            )
    design, forces = np.concatenate(designs), np.concatenate(forces)
    # columns of powers of lengths of a few Å differ in size: solve scaled
    scale = np.linalg.norm(design, axis=0)
    scale[scale == 0] = 1.0
    solution, _, rank, _ = np.linalg.lstsq(design / scale, forces)
    if rank < parameters:
        raise InvalidParameter(
            f"the frames determine {rank} of the {parameters} coefficients: "
            "displace more atoms, along more directions"
        )
    solution /= scale
    norm = np.linalg.norm(forces)
    error = np.linalg.norm(forces - design @ solution) / norm if norm > 0 else 0.0
    coefficients = solution.reshape(len(pairs), 2, powers)
    return BondStiffness(
        pairs=pairs,
        shells=shells,
        degree=degree,
        longitudinal=coefficients[:, 0].copy(),
        transverse=coefficients[:, 1].copy(),
        lengths=lengths,
        error=float(error),
    )
