# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The space group of a crystal, found by spglib."""

from dataclasses import dataclass

import numpy as np
import polars as pl

from mdapy.core.result import Result
from mdapy.core.system import System
from mdapy.errors import InvalidParameter


def _spglib():
    try:
        import spglib
    except ImportError:
        raise InvalidParameter(
            "the symmetry of crystals needs spglib: pip install spglib"
        ) from None
    return spglib


def _kinds(system: System) -> np.ndarray:
    """(N,) an integer for each kind of atom: by element, else by type."""
    table = system.table
    for name in ("element", "type"):
        if name in table.columns:
            column = table[name].cast(pl.String)
            return column.rank("dense").to_numpy().astype(np.int64)
    return np.zeros(len(system), dtype=np.int64)


@dataclass(frozen=True)
class _Operations:
    """The symmetry operations of a crystal in the units of its cell: atom
    i goes to atom ``atoms[k, i]`` of the cell moved by ``shifts[k, i]``."""

    rotations: np.ndarray  # (K, 3, 3) integer, on fractional columns
    translations: np.ndarray  # (K, 3) fractional
    cartesian: np.ndarray  # (K, 3, 3) on Cartesian columns
    atoms: np.ndarray  # (K, N)
    shifts: np.ndarray  # (K, N, 3) integer


def _dataset(system: System, tolerance: float):
    """The symmetry dataset of spglib for a periodic crystal."""
    box = system.box
    cell = (box.matrix, box.to_fractional(system["positions"]), _kinds(system))
    return _spglib().get_symmetry_dataset(cell, symprec=tolerance)


def _operations(system: System, tolerance: float) -> _Operations:
    """The symmetry operations of a periodic crystal, with the atom each
    takes each atom to; for cells of a few hundred atoms."""
    dataset = _dataset(system, tolerance)
    rotations = np.asarray(dataset.rotations, dtype=np.int64)
    translations = np.asarray(dataset.translations, dtype=float)
    lattice = system.box.matrix.T  # vectors as columns
    cartesian = lattice @ rotations @ np.linalg.inv(lattice)
    fractional = system.box.to_fractional(system["positions"])
    n = len(system)
    atoms = np.zeros((len(rotations), n), dtype=np.int64)
    shifts = np.zeros((len(rotations), n, 3), dtype=np.int64)
    for k, (rotation, translation) in enumerate(zip(rotations, translations)):
        images = fractional @ rotation.T + translation
        apart = images[:, None, :] - fractional[None, :, :]  # (N, N, 3)
        off = np.linalg.norm((apart - np.round(apart)) @ lattice.T, axis=2)
        atoms[k] = np.argmin(off, axis=1)
        shifts[k] = np.round(apart[np.arange(n), atoms[k]]).astype(np.int64)
    return _Operations(rotations, translations, cartesian, atoms, shifts)


# atoms closer than this in Å are images of each other
_TOLERANCE = 1e-5


def _symmetry(system: System, symmetry: bool | None) -> _Operations | None:
    """The symmetry operations of a cell when asked for (True), by default
    (None) when spglib is installed, or None."""
    if symmetry is False:
        return None
    if symmetry is None:
        try:
            import spglib  # noqa: F401
        except ImportError:
            return None
    return _operations(system, _TOLERANCE)


def _keeping(rotations: np.ndarray, *matrices: np.ndarray) -> np.ndarray:
    """(K,) whether each rotation, in the units of the cell, maps onto
    itself the lattice of each integer matrix, whose rows times the vectors
    of the cell are its vectors, such as a supercell."""
    keep = np.ones(len(rotations), dtype=bool)
    for matrix in matrices:
        vectors = np.asarray(matrix, dtype=float).T
        mapped = np.linalg.inv(vectors) @ rotations @ vectors
        keep &= (np.abs(mapped - np.round(mapped)) < 1e-8).all(axis=(1, 2))
    return keep


def _rotations(
    system: System, symmetry: bool | None, *matrices: np.ndarray
) -> np.ndarray:
    """(K, 3, 3) the distinct rotations on Cartesian columns of the symmetry
    operations of a cell, as :func:`_symmetry` finds them, that keep the
    lattices of the matrices; the identity alone without symmetry."""
    operations = _symmetry(system, symmetry)
    if operations is None:
        return np.eye(3)[None]
    keep = np.flatnonzero(_keeping(operations.rotations, *matrices))
    _, first = np.unique(operations.rotations[keep], axis=0, return_index=True)
    return operations.cartesian[keep[np.sort(first)]]


@dataclass(frozen=True)
class SpaceGroupResult(Result):
    """The space group of a crystal.

    Attributes
    ----------
    number : int
        The number of the space group in the International Tables, 1 to 230.
    symbol : str
        Its Hermann–Mauguin symbol, such as "Fm-3m".
    rotations : numpy.ndarray
        (K, 3, 3) the rotations of the K symmetry operations, acting on
        Cartesian column vectors.
    translations : numpy.ndarray
        (K, 3) their translations in Å: operation k moves the point r to
        ``rotations[k] @ r + translations[k]``.
    equivalent : numpy.ndarray
        (N,) int64 the first atom that symmetry makes each atom equivalent
        to.
    """

    number: int
    symbol: str
    rotations: np.ndarray
    translations: np.ndarray
    equivalent: np.ndarray


class SpaceGroup:
    """The space group of a crystal, its symmetry operations and the atoms
    they make equivalent, found by spglib.

    Parameters
    ----------
    tolerance : float, optional
        Distance in Å within which atoms count as images of each other.
        Default 1e-5.
    """

    def __init__(self, *, tolerance: float = 1e-5):
        if not tolerance > 0:
            raise InvalidParameter(f"tolerance must be positive, not {tolerance!r}")
        self.tolerance = float(tolerance)

    def compute(self, system: System) -> SpaceGroupResult:
        """The space group of a crystal.

        Parameters
        ----------
        system : System
            The cell of the crystal, periodic in every direction; atoms of
            different elements, or else of different types, are different.

        Returns
        -------
        SpaceGroupResult
            The space group, its operations and the equivalent atoms.

        Raises
        ------
        InvalidParameter
            If the box is not periodic, there are no atoms, or spglib is
            not installed.
        """
        if not system.box.pbc.all() or len(system) == 0:
            raise InvalidParameter(
                "the space group needs a box periodic in every direction with atoms"
            )
        dataset = _dataset(system, self.tolerance)
        lattice = system.box.matrix.T
        origin = system.box.origin
        rotations = lattice @ np.asarray(dataset.rotations) @ np.linalg.inv(lattice)
        translations = (
            np.asarray(dataset.translations) @ lattice.T
            + origin
            - np.einsum("kab,b->ka", rotations, origin)
        )
        return SpaceGroupResult(
            number=int(dataset.number),
            symbol=str(dataset.international),
            rotations=rotations,
            translations=translations,
            equivalent=np.asarray(dataset.equivalent_atoms, dtype=np.int64),
        )

    def __repr__(self) -> str:
        return f"SpaceGroup(tolerance={self.tolerance!r})"
