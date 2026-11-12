# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Force constants of a crystal in a supercell, and the FORCE_CONSTANTS files
of phonopy."""

import os
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

from mdapy._core import neighbor as _search
from mdapy.analysis._symmetry import _symmetry
from mdapy.core.system import System
from mdapy.errors import FileFormatError, InvalidParameter
from mdapy.lattice_dynamics._phonons import (
    Phonons,
    _kept,
    _kinds,
    _lattice_points,
    _masses,
    _moves,
    _reduced,
    _sheet_supercell,
    _supercell,
    _supercell_matrix,
    _widths,
)
from mdapy.lattice_dynamics._sheets import _in_periodic_box, _is_sheet

# how far in Å an atom may be from the image of another under a translation
# of the crystal
_TOLERANCE = 1e-3
# the most memory in bytes that the basis of a fit may take
_MEMORY = 2e9
# pairs this much farther apart in Å than the cutoff are within it, so that
# a cutoff typed at the distance of a shell holds the whole shell
_SHELL = 1e-4
# the cutoff stays this much in Å below half the shortest vector of the
# supercell, so that each pair has one shortest image
_MARGIN = 1e-3


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
    cutoff : float64 or None
        The cutoff in Å of a fit by ``force_constants`` beyond which the
        force constants are zero; None for every pair of the supercell, or
        not a fit.
    parameters : int64 or None
        The number of free force constants that symmetry and the sum rule
        leave, of a fit by ``force_constants``; None otherwise.
    error : float64 or None
        The relative error of the forces of a fit by ``force_constants``:
        the norm of the forces less those of the force constants over the
        norm of the forces; None otherwise.

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
    cutoff: float | None = field(default=None, init=False)
    parameters: int | None = field(default=None, init=False)
    error: float | None = field(default=None, init=False)

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


def displaced_supercells(
    cell: System,
    *,
    supercell: int | ArrayLike,
    method: str = "finite",
    displacement: float = 0.01,
    count: int = 1,
    seed: int | None = None,
    symmetry: bool | None = None,
) -> list[System]:
    """Supercells of a crystal with atoms displaced, whose forces, from any
    calculator, give its force constants by ``force_constants``.

    Parameters
    ----------
    cell : System
        The cell of the crystal, periodic in every direction, or of a sheet,
        periodic along two vectors in the x-y plane and not along a third
        along z, with atoms at rest.
    supercell : int32 or array_like
        Repeats of the cell along its vectors, one or three, or a (3, 3)
        integer matrix whose rows times the vectors of the cell are the
        vectors of the supercell; a sheet repeats along its plane only.
    method : str, optional
        "finite": one atom displaced in each supercell, the moves of
        ``phonons``; "random": every atom displaced along a random direction,
        each pattern by plus and minus. Default "finite".
    displacement : float64, optional
        Displacement of the atoms in Å. Default 0.01.
    count : int32, optional
        The number of random patterns, each giving two supercells. Default 1.
    seed : int64, optional
        Seed of the random directions. Default None, a new one each time.
    symmetry : bool, optional
        For "finite", move only the atoms and along the directions that the
        symmetry of the cell, found by spglib, does not relate to others:
        True, False, or by default whenever spglib is installed.

    Returns
    -------
    list of System
        The supercells, with the columns of the cell, the atoms of each cell
        of the supercell in the order of the cell.

    Raises
    ------
    InvalidParameter
        If the box is neither periodic nor that of a sheet, there are no
        atoms, the supercell is not positive integers or a matrix of them
        with a positive determinant, the method is unknown, the displacement
        is not positive, count is not a positive integer, or symmetry is
        True without spglib.
    """
    matrix = _matrix(cell, supercell)
    if not displacement > 0:
        raise InvalidParameter(f"displacement must be positive, not {displacement!r}")
    big = _supercell(cell, matrix)
    positions = big["positions"]

    def moved(shift: np.ndarray) -> System:
        return big.with_columns(positions=positions + shift)

    if method == "finite":
        sheet = _is_sheet(cell.box, "force constants")
        operations = _symmetry(_in_periodic_box(cell) if sheet else cell, symmetry)
        _, cartesian, atoms, _ = _kept(operations, matrix, len(cell))
        out = []
        for a, _, axes, flips in _moves(cartesian, atoms):
            for alpha, flip in zip(axes, flips):
                for sign in (1.0,) if flip >= 0 else (1.0, -1.0):
                    shift = np.zeros_like(positions)
                    shift[a, alpha] = sign * displacement
                    out.append(moved(shift))
        return out
    if method == "random":
        if (
            isinstance(count, bool)
            or not isinstance(count, (int, np.integer))
            or count < 1
        ):
            raise InvalidParameter(f"count must be a positive integer, not {count!r}")
        generator = np.random.default_rng(seed)
        out = []
        for _ in range(int(count)):
            directions = generator.normal(size=positions.shape)
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)
            out += [moved(displacement * directions), moved(-displacement * directions)]
        return out
    raise InvalidParameter(f'method must be "finite" or "random", not {method!r}')


def _orthogonal(matrices: np.ndarray) -> np.ndarray:
    """The nearest orthogonal matrices, which the Cartesian rotations of
    spglib's operations are to its tolerance."""
    u, _, vt = np.linalg.svd(matrices)
    return u @ vt


def _operations(
    cell: System, symmetry: bool | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """The symmetry operations of the cell, found by spglib when wanted: a
    sheet's in a box periodic along z too; None without symmetry."""
    sheet = _is_sheet(cell.box, "force constants")
    return _symmetry(_in_periodic_box(cell) if sheet else cell, symmetry)


def _torus_pairs(
    cell: System, matrix: np.ndarray, symmetry: bool | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Every pair of an atom of the cell and an atom of the supercell, with
    its periodic boundaries: the first atom and the atom of the supercell of
    each pair, the Cartesian rotation of each operation that maps the
    supercell onto itself, and the pair it takes each pair to."""
    n = len(cell)
    repeats = _lattice_points(matrix)
    size = len(repeats)
    count = n * size
    rotations, cartesian, atoms, shifts = _kept(_operations(cell, symmetry), matrix, n)
    # atom a of the cell moved back into the first cell, as _displaced_forces
    # does
    images = np.empty((len(rotations), n * count), dtype=np.int64)
    for k in range(len(rotations)):
        for a in range(n):
            lattice = (repeats @ rotations[k].T)[:, None, :] + shifts[k] - shifts[k, a]
            j = _cell_of(lattice.reshape(-1, 3), matrix) * n + np.tile(atoms[k], size)
            images[k, a * count : (a + 1) * count] = atoms[k, a] * count + j
    first = np.repeat(np.arange(n), count)
    second = np.tile(np.arange(count), n)
    # the pair of b in cell L and a in cell 0 is that of a in cell -L and b
    opposite = _cell_of(-repeats, matrix)
    exchanged = second % n * count + opposite[second // n] * n + first
    return first, second, cartesian, np.concatenate([images, exchanged[images]])


def _shortest(vectors: np.ndarray) -> float:
    """The length of the shortest vector of the lattice of these vectors."""
    reduced = _reduced(vectors)
    combinations = np.stack(
        np.meshgrid(*[np.arange(-2, 3)] * 3, indexing="ij"), -1
    ).reshape(-1, 3)
    lengths = np.linalg.norm(combinations @ reduced, axis=1)
    return float(lengths[lengths > 1e-9].min())


def _pair_table(
    cell: System, matrix: np.ndarray, cutoff: float, what: str = "cutoff"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The pairs of atoms of the crystal no farther apart than the cutoff,
    the first in the cell: the first atom, the second, and the cell of the
    second, in units of the cell. The cutoff is at most half the shortest
    vector of the supercell less a margin, so that each pair is its own
    shortest image in the supercell."""
    n = len(cell)
    box, positions = cell.box.matrix, cell["positions"]
    sheet = _is_sheet(cell.box, "force constants")
    big = matrix @ box
    if sheet:
        # a sheet has images in its plane only
        limit = 0.5 * _shortest(np.vstack([big[:2], [0.0, 0.0, 1e6]]))
    else:
        limit = 0.5 * _shortest(big)
    if not 0 < cutoff <= limit - _MARGIN:
        raise InvalidParameter(
            f"{what} must be positive and at most {limit - _MARGIN:.4g} Å, half "
            f"the shortest vector of the supercell less {_MARGIN} Å, not {cutoff!r}"
        )
    reach = np.ceil((cutoff + _SHELL) / _widths(box)).astype(int) + 1
    if sheet:
        reach[2] = 0
    grid = np.stack(
        np.meshgrid(*(np.arange(-r, r + 1) for r in reach), indexing="ij"), -1
    ).reshape(-1, 3)
    first, second, cells = [], [], []
    for a in range(n):
        vectors = (grid @ box)[:, None, :] + positions[None] - positions[a]
        g, b = np.nonzero(np.linalg.norm(vectors, axis=2) <= cutoff + _SHELL)
        first.append(np.full(len(g), a))
        second.append(b)
        cells.append(grid[g])
    return tuple(np.concatenate(x) for x in (first, second, cells))


def _cutoff_pairs(
    cell: System, matrix: np.ndarray, cutoff: float, symmetry: bool | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The pairs of atoms of the crystal no farther apart than the cutoff,
    the first in the cell, as ``_torus_pairs`` gives them: with every
    operation of the crystal, and each pair at its atom of the supercell."""
    n = len(cell)
    a, b, lattice = _pair_table(cell, matrix, cutoff)
    # a table of the pairs by their atoms and cell
    low = lattice.min(axis=0)
    span = np.ptp(lattice, axis=0) + 1

    def key(a: np.ndarray, b: np.ndarray, cell: np.ndarray) -> np.ndarray:
        r = cell - low
        inside = ((r >= 0) & (r < span)).all(axis=-1)
        k = (((a * n + b) * span[0] + r[..., 0]) * span[1] + r[..., 1]) * span[2] + r[
            ..., 2
        ]
        return np.where(inside, k, -1)

    table = np.full(n * n * int(np.prod(span)), -1, dtype=np.int64)
    table[key(a, b, lattice)] = np.arange(len(a))

    def find(a: np.ndarray, b: np.ndarray, cell: np.ndarray) -> np.ndarray:
        k = key(a, b, cell)
        return np.where(k >= 0, table[np.maximum(k, 0)], -1)

    operations = _operations(cell, symmetry)
    rotations, cartesian, atoms, shifts = _kept(operations, np.eye(3), n)
    images = np.stack(
        [
            find(
                atoms[k, a],
                atoms[k, b],
                lattice @ rotations[k].T + shifts[k, b] - shifts[k, a],
            )
            for k in range(len(rotations))
        ]
    )
    maps = np.concatenate([images, find(b, a, -lattice)[images]])
    # distances are kept, so every image is a pair of the table
    assert (maps >= 0).all()
    return a, _cell_of(lattice, matrix) * n + b, cartesian, maps


def _basis(
    cell: System,
    matrix: np.ndarray,
    cutoff: float | None,
    symmetry: bool | None,
) -> tuple[np.ndarray, int]:
    """(n N 9, K) the force constants of the pairs of an atom of the cell and
    an atom of the supercell, components by rows, of each of the K free
    parameters that the symmetry of the crystal, the exchange of the two
    atoms and the sum rule leave, zero for pairs beyond the cutoff; and K."""
    n, count = len(cell), len(cell) * round(np.linalg.det(matrix))
    if cutoff is None:
        first, second, cartesian, maps = _torus_pairs(cell, matrix, symmetry)
    else:
        first, second, cartesian, maps = _cutoff_pairs(cell, matrix, cutoff, symmetry)
    pairs = len(first)
    cartesian = _orthogonal(cartesian)
    turns = np.einsum("kab,kcd->kacbd", cartesian, cartesian).reshape(-1, 9, 9)
    transpose = np.eye(9)[np.arange(9).reshape(3, 3).T.reshape(-1)]
    matrices = np.concatenate([turns, transpose @ turns])
    blocks, done = [], np.zeros(pairs, dtype=bool)
    for p in range(pairs):
        if done[p]:
            continue
        reached = maps[:, p]
        # the operations that keep the pair make a group; the mean of its
        # matrices projects onto the components they keep
        projector = matrices[reached == p].mean(axis=0)
        u, s, _ = np.linalg.svd(projector)
        free = u[:, s > 0.5]
        orbit, by = np.unique(reached, return_index=True)
        done[orbit] = True
        if free.shape[1]:
            blocks.append((orbit, matrices[by] @ free))
    total = sum(b.shape[2] for _, b in blocks)
    size = max(pairs, n * count) * 9 * total * 8
    if size > _MEMORY:
        raise InvalidParameter(
            f"a fit of {total} free force constants of {pairs} pairs of atoms "
            f"needs {size / 1e9:.0f} GB: too many for a crystal of so little "
            "symmetry in so large a supercell; phonons finds its force constants "
            "by finite differences, phonons_from_frames from frames of any "
            "dynamics"
        )
    basis = np.zeros((pairs, 9, total))
    column = 0
    for orbit, block in blocks:
        basis[orbit, :, column : column + block.shape[2]] = block
        column += block.shape[2]
    # the sum rule: the force constants of each atom of the cell sum to zero
    sums = np.zeros((n, 9, total))
    np.add.at(sums, first, basis)
    _, s, vt = np.linalg.svd(sums.reshape(9 * n, total))
    rank = int((s > 1e-9 * max(1.0, s[0] if len(s) else 0.0)).sum())
    null = vt[rank:].T
    dense = np.zeros((n * count, 9, null.shape[1]))
    dense[first * count + second] = basis @ null
    return dense.reshape(n * count * 9, -1), null.shape[1]


def _displacements(
    big: System, frame: System, k: int, tolerance: float
) -> tuple[np.ndarray, np.ndarray]:
    """(N, 3) the displacement and the force of the atom of the frame at
    each atom of the supercell, the atoms matched by position."""
    if len(frame) != len(big) or "forces" not in frame.table.columns:
        raise InvalidParameter(
            f"frame {k} must hold the {len(big)} atoms of the supercell, with forces"
        )
    box = big.box
    if not np.allclose(frame.box.matrix, box.matrix, rtol=0, atol=tolerance):
        raise InvalidParameter(f"the box of frame {k} is not that of the supercell")
    sites = np.ascontiguousarray(box.wrap(big["positions"])[0])
    tree = _search.KdTree(
        sites,
        np.ascontiguousarray(box.matrix),
        np.ascontiguousarray(box.origin),
        np.ascontiguousarray(box.pbc),
    )
    # positions from the origin of the box, which a file such as an OUTCAR
    # may put elsewhere
    positions = frame["positions"] - frame.box.origin + box.origin
    found, _, _ = tree.nearest(np.ascontiguousarray(positions), 1)
    site = found[:, 0].astype(np.int64)
    if (np.bincount(site, minlength=len(big)) != 1).any():
        raise InvalidParameter(
            f"the atoms of frame {k} are not one at each atom of the supercell"
        )
    for column in ("element", "type"):
        if column in frame.table.columns and column in big.table.columns:
            if (
                frame.table[column].to_numpy() != big.table[column].to_numpy()[site]
            ).any():
                raise InvalidParameter(
                    f"an atom of frame {k} is not of the {column} of its atom of the "
                    "supercell"
                )
    displacements = np.empty((len(big), 3))
    forces = np.empty((len(big), 3))
    displacements[site] = box.minimum_image(positions - sites[site])
    forces[site] = frame["forces"]
    return displacements, forces


def _design(
    cell: System, matrix: np.ndarray, basis: np.ndarray, frames: Sequence[System]
) -> tuple[np.ndarray, np.ndarray]:
    """The forces of each of the K parameters of a basis, (n N 9, K) by the
    layout of ``_basis``, on the atoms of the frames, and the forces of the
    frames, both flat over the frames, atoms and axes."""
    n = len(cell)
    big = _supercell(cell, matrix)
    points = _lattice_points(matrix)
    size, count = len(points), len(big)
    parameters = basis.shape[1]
    # the atom of the supercell that each atom is, moved by each cell
    moved = _cell_of(points[:, None, :] + points[None, :, :], matrix)
    atom = np.arange(count)
    translated = moved[:, atom // n] * n + atom % n  # (S, N)
    # the forces of each parameter: F of atom b of cell c is minus the sum
    # over j of the force constants of b and j times the displacement of j
    # moved by c
    weights = (
        basis.reshape(n, count, 3, 3, parameters)
        .transpose(1, 3, 0, 2, 4)
        .reshape(count * 3, n * 3 * parameters)
    )
    rows, values = [], []
    for k, frame in enumerate(frames):
        u, f = _displacements(big, frame, k, 1e-4)
        rows.append(
            -(u[translated].reshape(size, count * 3) @ weights).reshape(-1, parameters)
        )
        values.append(f.reshape(-1))
    return np.concatenate(rows), np.concatenate(values)


def _supercell_of(cell: System, frames: Sequence[System]) -> np.ndarray:
    """The matrix of the supercell of the cell that the box of the first
    frame is."""
    repeats = np.asarray(frames[0].box.matrix) @ np.linalg.inv(cell.box.matrix)
    if np.abs(repeats - np.round(repeats)).max() > 1e-6:
        raise InvalidParameter("the box of frame 0 is not a supercell of the cell")
    return _matrix(cell, np.round(repeats).astype(np.int64))


def force_constants(
    cell: System,
    frames: Sequence[System],
    *,
    cutoff: float | None = None,
    symmetry: bool | None = None,
) -> ForceConstants:
    """Second order force constants of a crystal fitted to the forces on
    atoms displaced in a supercell.

    Parameters
    ----------
    cell : System
        The cell of the crystal, periodic in every direction, or of a sheet,
        periodic along two vectors in the x-y plane and not along a third
        along z, with atoms at rest (relax it first).
    frames : sequence of System
        Supercells of the cell with atoms displaced, such as those of
        ``displaced_supercells``, each with a ``forces`` column in eV/Å, its
        atoms in any order.
    cutoff : float64, optional
        Fit only the force constants of pairs no farther apart than this, in
        Å, at most half the shortest vector of the supercell less 1e-3 Å,
        with every symmetry operation of the crystal; by default those of
        every pair of the supercell, with its periodic boundaries and the
        operations that map it onto itself.
    symmetry : bool, optional
        Use the symmetry of the cell, found by spglib, in the fit: True,
        False, or by default whenever spglib is installed.

    Returns
    -------
    ForceConstants
        The force constants that give the forces best, symmetric under the
        operations and in the two atoms, and summing to zero over the atoms
        of the supercell, with the cutoff, the number of free parameters and
        the relative error of the forces.

    Raises
    ------
    InvalidParameter
        If the box is neither periodic nor that of a sheet, there are no
        atoms or no frames, a frame is not a supercell of the cell with
        forces, its atoms one at each atom of the supercell, the cutoff is
        not positive or too long for the supercell, the frames do not
        determine every free parameter, the fit is too large for memory, or
        symmetry is True without spglib.
    """
    frames = list(frames)
    if len(cell) == 0:
        raise InvalidParameter(
            "force constants need a box periodic in every direction with atoms"
        )
    if not frames:
        raise InvalidParameter("force constants need frames with forces")
    matrix = _supercell_of(cell, frames)
    n, count = len(cell), len(cell) * round(np.linalg.det(matrix))
    if cutoff is not None:
        cutoff = float(cutoff)
    basis, parameters = _basis(cell, matrix, cutoff, symmetry)
    design, forces = _design(cell, matrix, basis, frames)
    solution, _, rank, _ = np.linalg.lstsq(design, forces)
    if rank < parameters:
        raise InvalidParameter(
            f"the frames determine {rank} of the {parameters} free force constants: "
            "displace more atoms, along more directions"
        )
    result = ForceConstants(cell, matrix, (basis @ solution).reshape(n, count, 3, 3))
    norm = np.linalg.norm(forces)
    error = np.linalg.norm(forces - design @ solution) / norm if norm > 0 else 0.0
    object.__setattr__(result, "cutoff", cutoff)
    object.__setattr__(result, "parameters", int(parameters))
    object.__setattr__(result, "error", float(error))
    return result
