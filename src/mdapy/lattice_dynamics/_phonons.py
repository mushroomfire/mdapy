# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Harmonic phonons from the forces on displaced atoms."""

import os
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import ArrayLike

from mdapy import units
from mdapy._core import neighbor as _search
from mdapy.analysis._symmetry import _keeping, _Operations, _symmetry
from mdapy.calculators._base import Calculator
from mdapy.core.box import Box
from mdapy.core.system import System
from mdapy.elements import atomic_masses
from mdapy.errors import FileFormatError, InvalidParameter

# the frequency in THz of an eigenvalue of 1 eV/(Å² amu) of a dynamical matrix
_THZ = 1 / (2 * np.pi * np.sqrt(units.AMU_A2_PER_PS2_TO_EV))
# modes below this frequency in THz, the translations, carry no heat
_LOWEST = 1e-3
# frequencies below this in THz are a crystal that is not stable
_IMAGINARY = -1e-2
# frequencies closer than this in THz are taken as equal
_DEGENERATE = 1e-3


def _read_only(*arrays: np.ndarray | None) -> None:
    for array in arrays:
        if array is not None:
            array.flags.writeable = False


@dataclass(frozen=True)
class BandStructure:
    """Phonon frequencies along a path through the Brillouin zone.

    Attributes
    ----------
    q : numpy.ndarray
        (P, 3) wave vectors in units of the reciprocal lattice vectors of
        the cell.
    distances : numpy.ndarray
        (P,) distance of each wave vector along the path, in 1/Å.
    frequencies : numpy.ndarray
        (P, 3n) frequencies in THz, ascending at each wave vector, or each
        column a branch when connected; imaginary ones as negative.
    labels : tuple of (float, str)
        The distance and the label of each special point; two points that
        meet at a break of the path share a label such as "U|K".
    """

    q: np.ndarray
    distances: np.ndarray
    frequencies: np.ndarray
    labels: tuple[tuple[float, str], ...]

    def __post_init__(self):
        _read_only(self.q, self.distances, self.frequencies)

    def write(self, path: str | os.PathLike) -> None:
        """Writes the band structure in the format of phonopy's band.dat:
        the distances of the ends of the lines in 1/Å without the factor 2π,
        then each branch, line by line.

        Parameters
        ----------
        path : str or path-like
            The file.
        """
        starts = np.flatnonzero(np.diff(self.distances) == 0) + 1
        lines = np.split(np.arange(len(self.distances)), starts)
        x = self.distances / (2 * np.pi)
        ends = [d / (2 * np.pi) for d, _ in self.labels]
        if len(ends) != len(lines) + 1:
            ends = [x[lines[0][0]]] + [x[line[-1]] for line in lines]
        with open(path, "w") as f:
            f.write("# End points of segments: \n")
            f.write("#  " + "".join(f" {e:.8f}" for e in ends) + " \n")
            for branch in self.frequencies.T:
                for line in lines:
                    for k in line:
                        f.write(f"{x[k]:f} {branch[k]:f}\n")
                    f.write("\n")
                f.write("\n")


@dataclass(frozen=True)
class PhononDOS:
    """The phonon density of states per atom.

    Attributes
    ----------
    frequencies : numpy.ndarray
        (F,) frequencies in THz.
    dos : numpy.ndarray
        (F,) states per THz per atom; it integrates to 3.
    projected : numpy.ndarray or None
        (F, n) the density of states of each atom of the cell, each
        integrating to 3, whose mean is ``dos``; when asked for.
    """

    frequencies: np.ndarray
    dos: np.ndarray
    projected: np.ndarray | None = None

    def __post_init__(self):
        _read_only(self.frequencies, self.dos, self.projected)


@dataclass(frozen=True)
class ThermalProperties:
    """Harmonic thermodynamic properties of the vibrations, per atom.

    Attributes
    ----------
    temperatures : numpy.ndarray
        (T,) temperatures in K.
    free_energy : numpy.ndarray
        (T,) Helmholtz free energy of the vibrations in eV, with the zero
        point energy.
    energy : numpy.ndarray
        (T,) energy of the vibrations in eV, with the zero point energy.
    entropy : numpy.ndarray
        (T,) entropy in eV/K.
    heat_capacity : numpy.ndarray
        (T,) heat capacity at constant volume in eV/K.
    """

    temperatures: np.ndarray
    free_energy: np.ndarray
    energy: np.ndarray
    entropy: np.ndarray
    heat_capacity: np.ndarray

    def __post_init__(self):
        _read_only(
            self.temperatures,
            self.free_energy,
            self.energy,
            self.entropy,
            self.heat_capacity,
        )


# Special points of the standard paths in Cartesian coordinates, in units of
# 2π/a for the cubic lattices with a along x, y and z, and of 2π/a in the
# plane and 2π/c along z for the hexagonal lattice with a along x
# (Setyawan and Curtarolo 2010).
_S3 = np.sqrt(3)
_PATHS = {
    "sc": (
        (("Γ", "X", "M", "Γ", "R", "X"), ("M", "R")),
        {"Γ": (0, 0, 0), "X": (0, 0.5, 0), "M": (0.5, 0.5, 0), "R": (0.5, 0.5, 0.5)},
    ),
    "fcc": (
        (("Γ", "X", "U"), ("K", "Γ", "L", "W", "X")),
        {
            "Γ": (0, 0, 0),
            "X": (0, 1, 0),
            "U": (0.25, 1, 0.25),
            "K": (0.75, 0.75, 0),
            "L": (0.5, 0.5, 0.5),
            "W": (0.5, 1, 0),
        },
    ),
    "bcc": (
        (("Γ", "H", "N", "Γ", "P", "H"), ("P", "N")),
        {"Γ": (0, 0, 0), "H": (0, 1, 0), "N": (0.5, 0.5, 0), "P": (0.5, 0.5, 0.5)},
    ),
    "hexagonal": (
        (("Γ", "M", "K", "Γ", "A", "L", "H", "A"), ("L", "M"), ("K", "H")),
        {
            "Γ": (0, 0, 0),
            "M": (0.5, _S3 / 6, 0),
            "K": (1 / 3, _S3 / 3, 0),
            "A": (0, 0, 0.5),
            "L": (0.5, _S3 / 6, 0.5),
            "H": (1 / 3, _S3 / 3, 0.5),
        },
    ),
}


def read_band_structure(path: str | os.PathLike) -> BandStructure:
    """A band structure from a file in the format of phonopy's band.dat.

    Parameters
    ----------
    path : str or path-like
        The file, such as one phonopy wrote from a calculation with VASP.

    Returns
    -------
    BandStructure
        The distances, in 1/Å with the factor 2π, and the frequencies; the
        wave vectors are not in the file and are NaN, and the special points
        at the ends of the lines have no names.

    Raises
    ------
    FileFormatError
        If the file is not in that format.
    """
    with open(path) as f:
        lines = f.read().splitlines()
    try:
        ends = np.array(lines[1].lstrip("#").split(), dtype=float)
        # the branches, each ended by two blank lines
        branches, rows, blank = [], [], 0
        for line in lines[2:]:
            if line.strip():
                rows.append(line.split())
                blank = 0
                continue
            blank += 1
            if blank == 2 and rows:
                branches.append(np.array(rows, dtype=float))
                rows = []
        if rows:
            branches.append(np.array(rows, dtype=float))
        data = np.stack([b[:, 1] for b in branches], axis=1)
        distances = branches[0][:, 0]
    except (IndexError, ValueError):
        raise FileFormatError("not a band.dat of phonopy", str(path)) from None
    scale = 2 * np.pi
    return BandStructure(
        q=np.full((len(distances), 3), np.nan),
        distances=distances * scale,
        frequencies=data,
        labels=tuple((float(e) * scale, "") for e in ends),
    )


def _masses(system: System) -> np.ndarray:
    if "mass" in system.table.columns:
        return system["mass"].astype(float)
    if "element" in system.table.columns:
        return atomic_masses(system.table["element"].cast(pl.String).to_list())
    raise InvalidParameter("phonons need the mass of every atom: an element or a mass")


def _kinds(system: System) -> np.ndarray:
    """The kind of each atom: atoms of one kind have the same element, type
    and mass, where the system has them."""
    columns = [c for c in ("element", "type", "mass") if c in system.table.columns]
    if not columns:
        return np.zeros(len(system), dtype=np.int64)
    values = np.column_stack(
        [system.table[c].to_physical().to_numpy().astype(float) for c in columns]
    )
    return np.unique(values, axis=0, return_inverse=True)[1].reshape(-1)


def _lattice_basis(rows: np.ndarray) -> np.ndarray:
    """A basis of the lattice that the integer rows generate: their Hermite
    normal form, by integer row reduction."""
    rows = [[int(x) for x in row] for row in rows]
    basis = []
    for column in range(3):
        while True:
            live = [r for r in rows if r[column] != 0]
            if len(live) <= 1:
                break
            pivot = min(live, key=lambda r: abs(r[column]))
            for r in live:
                if r is not pivot:
                    factor = r[column] // pivot[column]
                    for c in range(3):
                        r[c] -= factor * pivot[c]
        pivot = next(r for r in rows if r[column] != 0)
        rows = [r for r in rows if r is not pivot]
        basis.append(pivot)
    return np.array(basis, dtype=float)


def _reduced(vectors: np.ndarray) -> np.ndarray:
    """Short, nearly orthogonal vectors of the same lattice (LLL), right-handed."""
    b = np.array(vectors, dtype=float)

    def orthogonalized():
        star, mu = b.copy(), np.zeros((3, 3))
        for i in range(3):
            for j in range(i):
                mu[i, j] = b[i] @ star[j] / (star[j] @ star[j])
                star[i] -= mu[i, j] * star[j]
        return star, mu

    k = 1
    while k < 3:
        for j in range(k - 1, -1, -1):
            _, mu = orthogonalized()
            b[k] -= np.round(mu[k, j]) * b[j]
        star, mu = orthogonalized()
        if star[k] @ star[k] >= (0.75 - mu[k, k - 1] ** 2) * (
            star[k - 1] @ star[k - 1]
        ):
            k += 1
        else:
            b[[k - 1, k]] = b[[k, k - 1]]
            k = max(k - 1, 1)
    if np.linalg.det(b) < 0:
        b[2] = -b[2]
    return b


def _translations(system: System, tolerance: float) -> tuple[np.ndarray, list]:
    """The translations that map the crystal onto itself, the cell's own
    lattice vectors aside: (m, 3) vectors in Å, the first zero, and for each
    the atom that each atom goes to."""
    box = system.box
    positions, _ = box.wrap(system["positions"])
    positions = np.ascontiguousarray(positions)
    kinds = _kinds(system)
    tree = _search.KdTree(
        positions,
        np.ascontiguousarray(box.matrix),
        np.ascontiguousarray(box.origin),
        np.ascontiguousarray(box.pbc),
    )
    found, maps = [np.zeros(3)], [np.arange(len(system))]
    for j in np.flatnonzero(kinds == kinds[0])[1:]:
        shift = positions[j] - positions[0]
        image, _, distance = tree.nearest(np.ascontiguousarray(positions + shift), 1)
        image = image[:, 0].astype(np.int64)
        if distance.max() < tolerance and (kinds[image] == kinds).all():
            found.append(shift)
            maps.append(image)
    return np.array(found), maps


def primitive_cell(system: System, *, tolerance: float = 1e-4) -> System:
    """The smallest periodic cell of a crystal.

    Parameters
    ----------
    system : System
        A crystal in a box periodic in every direction. Atoms are alike when
        they have the same element, type and mass, where the system has
        these columns.
    tolerance : float, optional
        How far, in Å, an atom may be from the image of another under a
        translation of the crystal. Default 1e-4.

    Returns
    -------
    System
        The atoms of one primitive cell, with all the columns of system, in
        a box of short lattice vectors of the crystal with the same origin;
        system itself when its box is already primitive.

    Raises
    ------
    InvalidParameter
        If the box is not periodic in every direction or has no atoms.
    """
    if not system.box.pbc.all() or len(system) == 0:
        raise InvalidParameter("a primitive cell needs a periodic box with atoms")
    found, maps = _translations(system, tolerance)
    count = len(found)
    if count == 1:
        return system
    box = system.box
    # count times each translation is a vector of the lattice of the box
    fractional = found @ np.linalg.inv(box.matrix)
    integers = np.round(fractional * count)
    rows = np.vstack([count * np.eye(3), integers])
    vectors = _reduced(_lattice_basis(rows) / count @ box.matrix)
    # one atom of each set that the translations map onto each other
    first = np.min(np.stack(maps), axis=0)
    keep = np.flatnonzero(first == np.arange(len(system)))
    cell = Box(vectors, origin=box.origin)
    wrapped, _ = cell.wrap(system["positions"][keep])
    return system.gather(keep).with_box(cell).with_columns(positions=wrapped)


def _repeat(value: int | ArrayLike, name: str) -> np.ndarray:
    counts = np.array(value)
    if counts.ndim == 0:
        counts = np.repeat(counts, 3)
    if (
        counts.shape != (3,)
        or not np.issubdtype(counts.dtype, np.integer)
        or (counts < 1).any()
    ):
        raise InvalidParameter(
            f"{name} must be one or three positive integers, not {value!r}"
        )
    return counts.astype(int)


def _supercell_matrix(value: int | ArrayLike) -> np.ndarray:
    """(3, 3) integer matrix of a supercell: its vectors are its rows times
    the vectors of the cell."""
    array = np.array(value)
    if array.shape == (3, 3):
        if not np.issubdtype(array.dtype, np.integer) or not np.linalg.det(array) > 0.5:
            raise InvalidParameter(
                "a supercell matrix must hold integers, with a positive determinant"
            )
        return array.astype(int)
    return np.diag(_repeat(value, "supercell"))


def _lattice_points(matrix: np.ndarray) -> np.ndarray:
    """(S, 3) the lattice vectors of the cell, in its units, that reach the S
    cells of a supercell: zero first, then ordered by the first, the second
    and the third; for a diagonal matrix, the third varies fastest."""
    corners = (
        np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)]) @ matrix
    )
    axes = [
        np.arange(lo, hi + 1)
        for lo, hi in zip(corners.min(axis=0), corners.max(axis=0))
    ]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    fractional = grid @ np.linalg.inv(matrix)
    inside = ((fractional > -1e-9) & (fractional < 1 - 1e-9)).all(axis=1)
    points = grid[inside]
    # the cell itself first
    origin = np.flatnonzero((points == 0).all(axis=1))
    return np.concatenate([points[origin], np.delete(points, origin, axis=0)])


def _widths(matrix: np.ndarray) -> np.ndarray:
    """The distances between opposite faces of a box of these vectors."""
    areas = np.linalg.norm(np.cross(matrix[[1, 2, 0]], matrix[[2, 0, 1]]), axis=1)
    return abs(np.linalg.det(matrix)) / areas


def _default_supercell(cell: np.ndarray, reach: float) -> np.ndarray:
    """The supercell nearest a cube that is at least reach thick in every
    direction: of the supercells round(edge inv(cell)) for edges from reach
    to twice reach, the fewest cells among those as near a cube, by the
    thinnest width over the edge of a cube of the same volume, as 0.95 of
    the nearest."""
    if (_widths(cell) >= reach - 1e-9).all():
        return np.eye(3, dtype=int)
    inverse = np.linalg.inv(cell)
    found = {}
    edge = reach
    while edge < 2 * reach + np.linalg.norm(cell, axis=1).max():
        matrix = np.round(edge * inverse).astype(int)
        size = round(np.linalg.det(matrix))
        if size > 0 and (_widths(matrix @ cell) >= reach - 1e-9).all():
            box = matrix @ cell
            cubic = _widths(box).min() / abs(np.linalg.det(box)) ** (1 / 3)
            found[matrix.tobytes()] = (size, cubic, matrix)
        edge *= 1.01
    candidates = list(found.values())
    best = max(c[1] for c in candidates)
    return min((c for c in candidates if c[1] >= 0.95 * best), key=lambda c: c[0])[2]


def _mesh(mesh: int | ArrayLike) -> np.ndarray:
    """(K, 3) wave vectors of a mesh through Γ, in units of the reciprocal
    lattice vectors, in the order of the indices of the third vector first."""
    counts = _repeat(mesh, "mesh")
    axes = [np.arange(n) / n for n in counts]
    return np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)


def _tetrahedra(counts: np.ndarray, reciprocal: np.ndarray) -> np.ndarray:
    """(6K, 4) mesh points at the corners of the tetrahedra that fill the
    mesh: six in each of its cells, around its shortest main diagonal."""
    n1, n2, n3 = counts
    step = reciprocal / counts[:, None]
    starts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    signs = 1 - 2 * starts
    lengths = [np.linalg.norm(s @ step) for s in signs]
    start, sign = starts[np.argmin(lengths)], signs[np.argmin(lengths)]
    corners = []
    for a, b in ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)):
        first = start.copy()
        second = first.copy()
        second[a] += sign[a]
        third = second.copy()
        third[b] += sign[b]
        corners.append([first, second, third, start + sign])
    corners = np.array(corners)  # (6, 4, 3)
    cells = np.stack(
        np.meshgrid(np.arange(n1), np.arange(n2), np.arange(n3), indexing="ij"), -1
    ).reshape(-1, 1, 1, 3)
    points = (cells + corners) % counts
    flat = (points[..., 0] * n2 + points[..., 1]) * n3 + points[..., 2]
    return flat.reshape(-1, 4)


def _tetrahedron_dos(corners: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """(T, F) density of states of each tetrahedron of a band, with the
    (T, 4) values at its corners, each integrating to 1."""
    e = np.sort(corners, axis=1)
    e1, e2, e3, e4 = (e[:, k : k + 1] for k in range(4))
    x = grid[None, :]

    def safe(value):
        return np.where(value == 0, 1.0, value)

    rising = 3 * (x - e1) ** 2 / safe((e2 - e1) * (e3 - e1) * (e4 - e1))
    middle = (
        3 * (e2 - e1)
        + 6 * (x - e2)
        - 3 * (e3 - e1 + e4 - e2) * (x - e2) ** 2 / safe((e3 - e2) * (e4 - e2))
    ) / safe((e3 - e1) * (e4 - e1))
    falling = 3 * (e4 - x) ** 2 / safe((e4 - e1) * (e4 - e2) * (e4 - e3))
    return np.select(
        [(e1 < x) & (x < e2), (e2 <= x) & (x < e3), (e3 <= x) & (x < e4)],
        [rising, middle, falling],
        0.0,
    )


def _groups(frequencies: np.ndarray, tolerance: float) -> list[np.ndarray]:
    """The sets of two or more ascending frequencies within tolerance of
    the next."""
    apart = np.flatnonzero(np.diff(frequencies) > tolerance) + 1
    return [g for g in np.split(np.arange(len(frequencies)), apart) if len(g) > 1]


def _assigned(overlap: np.ndarray) -> np.ndarray:
    """The mode of each branch, the largest overlaps first, each mode once."""
    count = len(overlap)
    mode = np.full(count, -1)
    taken = np.zeros(count, dtype=bool)
    for flat in np.argsort(-overlap, axis=None, kind="stable"):
        branch, m = divmod(int(flat), count)
        if mode[branch] < 0 and not taken[m]:
            mode[branch], taken[m] = m, True
    return mode


def _connected(
    frequencies: np.ndarray, vectors: np.ndarray, starts: list[int], tolerance: float
) -> np.ndarray:
    """Frequencies along a path, each column one branch followed from point
    to point by the overlap of its polarization vectors; within a set of
    equal frequencies the vectors mix, so a branch keeps the vector it had
    before them, and branches that come out of such a set at the start of a
    section are matched as one."""
    count, branches = frequencies.shape
    flat = vectors.reshape(count, branches, -1)
    out = frequencies.copy()
    ends = starts[1:] + [count]
    for start, end in zip(starts, ends):
        reference = flat[start].copy()
        loose = _groups(frequencies[start], tolerance)  # branches matched as one
        for k in range(start + 1, end):
            overlap = np.abs(reference.conj() @ flat[k].T) ** 2
            for group in loose:
                overlap[group] = overlap[group].sum(axis=0)
            mode = _assigned(overlap)
            for group in loose:
                mode[group] = sorted(mode[group], key=lambda m: frequencies[k, m])
            out[k] = frequencies[k, mode]
            equal = _groups(frequencies[k], tolerance)
            mixed = np.zeros(branches, dtype=bool)
            for group in equal:
                mixed[np.isin(mode, group)] = True
            was_loose = np.zeros(branches, dtype=bool)
            for group in loose:
                was_loose[group] = True
            renew = ~mixed | was_loose
            reference[renew] = flat[k, mode[renew]]
            loose = [
                np.flatnonzero(np.isin(mode, group) & was_loose) for group in equal
            ]
            loose = [g for g in loose if len(g) > 1]
    return out


class Phonons:
    """Harmonic phonons of a crystal from its force constants.

    Made by :func:`phonons`. Wave vectors are in units of the reciprocal
    lattice vectors of the cell: q = (h, k, l) is h b1 + k b2 + l b3 with
    a_i · b_j = 2π δ_ij.

    Attributes
    ----------
    cell : System
        The crystal whose vibrations these are.
    supercell : numpy.ndarray
        (3, 3) integer matrix of the supercell in which the force constants
        were found: its vectors are the rows of the matrix times the vectors
        of the cell.
    force_constants : numpy.ndarray
        (n, n S, 3, 3) second derivatives of the energy in eV/Å² with
        respect to the positions of each atom of the cell and of each atom
        of the supercell, which holds S cells, each with the atoms of the
        cell in their order, the first cell that of the cell itself.
    """

    def __init__(
        self,
        cell: System,
        supercell: np.ndarray,
        force_constants: np.ndarray,
        masses: np.ndarray,
        reference: System | None = None,
    ):
        # reference: the same atoms in the same order, whose geometry picks
        # the periodic images that the force constants of each pair go to;
        # strained copies of a crystal take the unstrained one, so that
        # their frequencies change smoothly with the strain
        self.cell = cell
        self.supercell = np.array(supercell, dtype=int)
        self.supercell.flags.writeable = False
        self.force_constants = force_constants
        self.force_constants.flags.writeable = False
        n = len(cell)
        matrix = cell.box.matrix
        repeats = _lattice_points(self.supercell)
        grid = np.stack(np.meshgrid(*[np.arange(-2, 3)] * 3, indexing="ij"), -1)

        def images(geometry: System, a: int) -> np.ndarray:
            """(M, 125, 3) the vectors from atom a to the images of each
            atom of the supercell."""
            box, positions = geometry.box.matrix, geometry["positions"]
            atoms = ((repeats @ box)[:, None, :] + positions[None]).reshape(-1, 3)
            shifts = grid.reshape(-1, 3) @ (self.supercell @ box)
            return (atoms - positions[a])[:, None, :] + shifts[None]

        # each pair at its shortest distances in the supercell, shared
        # equally among the images that are as short
        first, second, vectors, weights = [], [], [], []
        for a in range(n):
            here = images(cell, a)
            chosen = here if reference is None else images(reference, a)
            lengths = np.linalg.norm(chosen, axis=2)
            shortest = lengths.min(axis=1, keepdims=True)
            near = lengths < shortest + 1e-5 * (1.0 + shortest)
            j, s = np.nonzero(near)
            first.append(np.full(len(j), a))
            second.append(j)
            vectors.append(here[j, s])
            weights.append(1.0 / near.sum(axis=1)[j])
        self.__a = np.concatenate(first)
        self.__j = np.concatenate(second)
        self.__vectors = np.concatenate(vectors)
        self.__weights = np.concatenate(weights)
        self.__constants = force_constants[self.__a, self.__j].reshape(-1, 9)
        b = self.__j % n
        self.__blocks = [
            (a, c, np.flatnonzero((self.__a == a) & (b == c)))
            for a in range(n)
            for c in range(n)
        ]
        self.__masses = masses
        self.__reciprocal = 2 * np.pi * np.linalg.inv(matrix).T

    def __repr__(self) -> str:
        cells = round(np.linalg.det(self.supercell))
        return f"Phonons({len(self.cell)} atoms, supercell of {cells} cells)"

    def __wave_vectors(self, q: ArrayLike) -> tuple[np.ndarray, bool]:
        q = np.asarray(q, dtype=float)
        single = q.ndim == 1
        q = np.atleast_2d(q)
        if q.ndim != 2 or q.shape[1] != 3 or not np.isfinite(q).all():
            raise InvalidParameter("q must be a wave vector or (K, 3) wave vectors")
        return q, single

    def __dynamical(self, q: np.ndarray) -> np.ndarray:
        """(K, 3n, 3n) dynamical matrices in eV/(Å² amu)."""
        n = len(self.cell)
        phase = np.exp(1j * (q @ self.__reciprocal) @ self.__vectors.T) * self.__weights
        blocks = np.zeros((len(q), n, n, 9), dtype=complex)
        for a, b, pairs in self.__blocks:
            blocks[:, a, b] = phase[:, pairs] @ self.__constants[pairs]
        masses = np.sqrt(np.outer(self.__masses, self.__masses))
        blocks /= masses[None, :, :, None]
        matrices = (
            blocks.reshape(len(q), n, n, 3, 3)
            .transpose(0, 1, 3, 2, 4)
            .reshape(len(q), 3 * n, 3 * n)
        )
        return 0.5 * (matrices + matrices.conj().transpose(0, 2, 1))

    def __solve(self, q: np.ndarray, vectors: bool, chunk: int = 2048):
        frequencies, modes = [], []
        for start in range(0, len(q), chunk):
            matrices = self.__dynamical(q[start : start + chunk])
            if vectors:
                values, eigen = np.linalg.eigh(matrices)
                modes.append(eigen.transpose(0, 2, 1))
            else:
                values = np.linalg.eigvalsh(matrices)
            frequencies.append(np.sign(values) * np.sqrt(np.abs(values)) * _THZ)
        frequencies = np.concatenate(frequencies)
        if not vectors:
            return frequencies, None
        n = len(self.cell)
        return frequencies, np.concatenate(modes).reshape(len(q), 3 * n, n, 3)

    def frequencies(self, q: ArrayLike) -> np.ndarray:
        """Phonon frequencies at wave vectors.

        Parameters
        ----------
        q : array_like
            A wave vector or (K, 3) wave vectors, in units of the reciprocal
            lattice vectors of the cell.

        Returns
        -------
        numpy.ndarray
            (3n,) or (K, 3n) frequencies in THz, ascending; imaginary ones
            as negative.
        """
        q, single = self.__wave_vectors(q)
        frequencies, _ = self.__solve(q, vectors=False)
        return frequencies[0] if single else frequencies

    def modes(self, q: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Phonon frequencies and polarization vectors at wave vectors.

        Parameters
        ----------
        q : array_like
            A wave vector or (K, 3) wave vectors, in units of the reciprocal
            lattice vectors of the cell.

        Returns
        -------
        frequencies : numpy.ndarray
            (3n,) or (K, 3n) frequencies in THz, ascending.
        vectors : numpy.ndarray
            (3n, n, 3) or (K, 3n, n, 3) complex polarization vector of each
            mode, normalized over the atoms of the cell; atom b of the cell
            in the cell at R moves along its vector divided by the square
            root of its mass, times exp(i q · (R + r_b)).
        """
        q, single = self.__wave_vectors(q)
        frequencies, vectors = self.__solve(q, vectors=True)
        if single:
            return frequencies[0], vectors[0]
        return frequencies, vectors

    def __standard_path(self, lattice: str) -> list[list[tuple[str, np.ndarray]]]:
        if lattice not in _PATHS:
            raise InvalidParameter(
                f"path must be one of {tuple(_PATHS)} or a list of segments, "
                f"not {lattice!r}"
            )
        sections, points = _PATHS[lattice]
        # the lattice constants, from the translations of the crystal
        matrix = primitive_cell(self.cell).box.matrix
        combinations = np.stack(
            np.meshgrid(*[np.arange(-3, 4)] * 3, indexing="ij"), -1
        ).reshape(-1, 3)
        translations = combinations @ matrix
        lengths = np.linalg.norm(translations, axis=1)
        nonzero = lengths > 1e-9
        shortest = lengths[nonzero].min()
        if lattice == "hexagonal":
            flat = nonzero & (np.abs(translations[:, 2]) < 1e-6 * shortest)
            upright = nonzero & (
                np.linalg.norm(translations[:, :2], axis=1) < 1e-6 * shortest
            )
            if not flat.any() or not upright.any():
                raise InvalidParameter(
                    "the cell is not a hexagonal lattice with c along z"
                )
            a, c = lengths[flat].min(), lengths[upright].min()
            expected = np.array([[a, 0, 0], [-a / 2, a * _S3 / 2, 0], [0, 0, c]])
            scale = 2 * np.pi * np.array([1 / a, 1 / a, 1 / c])
        else:
            a = shortest * {"sc": 1.0, "fcc": np.sqrt(2), "bcc": 2 / _S3}[lattice]
            expected = {
                "sc": a * np.eye(3),
                "fcc": a / 2 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]),
                "bcc": a / 2 * np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]]),
            }[lattice]
            scale = np.full(3, 2 * np.pi / a)
        integers = expected @ np.linalg.inv(matrix)
        if np.abs(integers - np.round(integers)).max() > 1e-6 or not np.isclose(
            abs(np.linalg.det(np.round(integers))), 1
        ):
            raise InvalidParameter(
                f"the cell is not a {lattice} lattice with its axes along x, y and z"
            )
        to_cell = self.cell.box.matrix.T / (2 * np.pi)
        return [
            [(label, (np.array(points[label]) * scale) @ to_cell) for label in section]
            for section in sections
        ]

    def __auto_path(self) -> list[list[tuple[str, np.ndarray]]]:
        try:
            import seekpath
        except ImportError:
            raise ImportError(
                'path="auto" needs seekpath: pip install seekpath'
            ) from None
        cell = self.cell
        found = seekpath.get_path(
            (
                cell.box.matrix,
                cell.box.to_fractional(cell["positions"]),
                (_kinds(cell) + 1).tolist(),
            )
        )
        # its points in its own standard cell, turned back into this one
        reciprocal = np.array(found["reciprocal_primitive_lattice"])
        turn = np.array(found["rotation_matrix"])
        to_cell = cell.box.matrix.T / (2 * np.pi)
        points = {
            label: (np.array(q) @ reciprocal) @ turn @ to_cell
            for label, q in found["point_coords"].items()
        }
        sections: list[list[str]] = []
        for start, end in found["path"]:
            if sections and sections[-1][-1] == start:
                sections[-1].append(end)
            else:
                sections.append([start, end])
        name = {"GAMMA": "Γ"}
        return [
            [(name.get(label, label), points[label]) for label in section]
            for section in sections
        ]

    def band_structure(
        self,
        path: str | Sequence[Sequence[tuple[str, ArrayLike]]],
        *,
        points: int = 51,
        connected: bool = False,
    ) -> BandStructure:
        """Phonon frequencies along a path through the Brillouin zone.

        Parameters
        ----------
        path : str or sequence of sequences of (str, array_like)
            "auto" for the standard path of the crystal by its symmetry,
            from seekpath; "sc", "fcc", "bcc" or "hexagonal" for the standard
            path of that lattice, the cell having the axes of its
            conventional cell along x, y and z, or a along x and c along z;
            or the sections of a path, each a sequence of labeled wave
            vectors in units of the reciprocal lattice vectors of the cell,
            joined by straight lines.
        points : int, optional
            Number of wave vectors on each line, its ends included.
            Default 51.
        connected : bool, optional
            Follow each branch along the path by its polarization vector, so
            that branches that cross stay apart; otherwise the frequencies
            are ascending at each wave vector. Default False.

        Returns
        -------
        BandStructure
            The wave vectors, their distances along the path, the
            frequencies and the labels of the special points.

        Raises
        ------
        ImportError
            For path "auto" without seekpath.
        """
        if path == "auto":
            sections = self.__auto_path()
        elif isinstance(path, str):
            sections = self.__standard_path(path)
        else:
            try:
                sections = [
                    [(str(label), np.asarray(q, dtype=float)) for label, q in section]
                    for section in path
                ]
            except (TypeError, ValueError):
                raise InvalidParameter(
                    "path must be sections of (label, wave vector) points"
                ) from None
            if any(len(s) < 2 or any(q.shape != (3,) for _, q in s) for s in sections):
                raise InvalidParameter(
                    "each section of a path needs two or more wave vectors of 3 values"
                )
        if int(points) < 2:
            raise InvalidParameter(f"points must be at least 2, not {points!r}")
        t = np.linspace(0, 1, int(points))[:, None]
        wave_vectors, distances, labels, starts = [], [], [], []
        reached = 0.0
        for section in sections:
            starts.append(sum(len(w) for w in wave_vectors))
            if labels:
                labels[-1] = (reached, f"{labels[-1][1]}|{section[0][0]}")
            else:
                labels.append((reached, section[0][0]))
            for (_, start), (label, end) in zip(section, section[1:]):
                line = start + t * (end - start)
                length = np.linalg.norm((end - start) @ self.__reciprocal)
                wave_vectors.append(line)
                distances.append(reached + t[:, 0] * length)
                reached += length
                labels.append((reached, label))
        q = np.concatenate(wave_vectors)
        if connected:
            frequencies, vectors = self.__solve(q, vectors=True)
            frequencies = _connected(frequencies, vectors, starts, _DEGENERATE)
        else:
            frequencies = self.frequencies(q)
        return BandStructure(
            q=q,
            distances=np.concatenate(distances),
            frequencies=frequencies,
            labels=tuple((float(d), label) for d, label in labels),
        )

    def __on_mesh(self, mesh, vectors: bool, stable: bool):
        q = _mesh(mesh)
        frequencies, modes = self.__solve(q, vectors=vectors)
        lowest = frequencies.min()
        if stable and lowest < _IMAGINARY:
            raise InvalidParameter(
                f"imaginary frequencies down to {lowest:.3g} THz: the crystal is "
                f"not stable, or the supercell is too small"
            )
        return q, frequencies, modes

    def dos(
        self,
        mesh: int | ArrayLike = 20,
        *,
        frequencies: ArrayLike | None = None,
        sigma: float | None = None,
        projected: bool = False,
    ) -> PhononDOS:
        """The phonon density of states, by the tetrahedron method or with
        Gaussians.

        Parameters
        ----------
        mesh : int or array_like, optional
            Number of wave vectors along each reciprocal lattice vector, of
            a mesh through Γ. Default 20.
        frequencies : array_like, optional
            The frequencies in THz at which to give the density. Default 501
            from 0, or a little below the lowest imaginary frequency, to a
            little above the highest.
        sigma : float, optional
            Width in THz of a Gaussian for each mode; by default the
            tetrahedron method.
        projected : bool, optional
            Also give the part of each atom of the cell. Default False.

        Returns
        -------
        PhononDOS
            The frequencies and the density of states per atom.

        Raises
        ------
        InvalidParameter
            For a mesh or a sigma that is not positive.
        """
        if sigma is not None and not sigma > 0:
            raise InvalidParameter(f"sigma must be positive, not {sigma!r}")
        counts = _repeat(mesh, "mesh")
        _, values, modes = self.__on_mesh(counts, vectors=projected, stable=False)
        if frequencies is None:
            low = min(0.0, 1.05 * values.min() if values.min() < _IMAGINARY else 0.0)
            grid = np.linspace(low, 1.05 * values.max(), 501)
        else:
            grid = np.asarray(frequencies, dtype=float).reshape(-1)
        n = len(self.cell)
        weights = None
        if projected:
            weights = (np.abs(modes) ** 2).sum(axis=3)  # (K, 3n, n)
        dos = np.zeros(len(grid))
        parts = np.zeros((len(grid), n)) if projected else None
        if sigma is not None:
            for k in range(len(values)):
                g = np.exp(-0.5 * ((grid[:, None] - values[k][None]) / sigma) ** 2)
                g /= sigma * np.sqrt(2 * np.pi)
                dos += g.sum(axis=1)
                if projected:
                    parts += g @ weights[k]
            count = len(values)
        else:
            tetrahedra = _tetrahedra(counts, self.__reciprocal)
            for band in range(3 * n):
                for start in range(0, len(tetrahedra), 4096):
                    corners = tetrahedra[start : start + 4096]
                    g = _tetrahedron_dos(values[corners, band], grid)
                    dos += g.sum(axis=0)
                    if projected:
                        parts += g.T @ weights[corners, band].mean(axis=1)
            count = len(tetrahedra)
        return PhononDOS(
            frequencies=grid,
            dos=dos / (count * n),
            projected=None if parts is None else parts / count,
        )

    def thermal(
        self, temperatures: ArrayLike, mesh: int | ArrayLike = 20
    ) -> ThermalProperties:
        """Harmonic thermodynamic properties of the vibrations, by sums over
        the modes on a mesh.

        Parameters
        ----------
        temperatures : array_like
            Temperatures in K, not negative.
        mesh : int or array_like, optional
            Number of wave vectors along each reciprocal lattice vector, of
            a mesh through Γ. Default 20.

        Returns
        -------
        ThermalProperties
            Free energy, energy, entropy and heat capacity per atom; modes
            below 1e-3 THz, the translations of the crystal, are left out.

        Raises
        ------
        InvalidParameter
            For a negative temperature, a mesh that is not positive, or
            imaginary frequencies on the mesh.
        """
        temperatures = np.atleast_1d(np.asarray(temperatures, dtype=float))
        if temperatures.ndim != 1 or not (temperatures >= 0).all():
            raise InvalidParameter("temperatures must not be negative")
        _, values, _ = self.__on_mesh(mesh, vectors=False, stable=True)
        quanta = units.PLANCK * values[values > _LOWEST]  # eV
        count = values.size / (3 * len(self.cell))  # wave vectors per atom
        kt = units.BOLTZMANN * temperatures
        free, energy, entropy, capacity = [], [], [], []
        for t in kt:
            if t == 0:
                free.append(0.5 * quanta.sum())
                energy.append(0.5 * quanta.sum())
                entropy.append(0.0)
                capacity.append(0.0)
                continue
            x = quanta / t
            occupation = 1.0 / np.expm1(x)
            free.append((0.5 * quanta + t * np.log(-np.expm1(-x))).sum())
            energy.append((quanta * (0.5 + occupation)).sum())
            entropy.append(
                units.BOLTZMANN * (x * occupation - np.log(-np.expm1(-x))).sum()
            )
            capacity.append(
                units.BOLTZMANN * (x**2 * occupation * (1 + occupation)).sum()
            )
        scale = 1.0 / (count * len(self.cell))
        return ThermalProperties(
            temperatures=temperatures,
            free_energy=np.array(free) * scale,
            energy=np.array(energy) * scale,
            entropy=np.array(entropy) * scale,
            heat_capacity=np.array(capacity) * scale,
        )


def _supercell(cell: System, matrix: np.ndarray) -> System:
    repeats = _lattice_points(matrix)
    positions = (repeats @ cell.box.matrix)[:, None, :] + cell["positions"][None]
    table = pl.concat([cell.table] * len(repeats)).with_columns(
        pl.Series("positions", positions.reshape(-1, 3), dtype=pl.Array(pl.Float64, 3))
    )
    box = Box(matrix @ cell.box.matrix, origin=cell.box.origin)
    return System(table=table, box=box)


def _displaced_forces(
    cell: System,
    calculator: Calculator,
    matrix: np.ndarray,
    displacement: float,
    operations: _Operations | None = None,
) -> np.ndarray:
    """(n, n S, 3, 3) force constants by central differences of the forces
    on the atoms of the supercell. Without symmetry operations each atom of
    the first cell moves along x, y and z by plus and minus displacement.
    With them, one atom of each set of equivalent atoms moves, along the
    fewest axes whose images under the operations that keep it in place
    span space, and only by plus displacement where such an operation turns
    the axis around; the forces of the other moves are images of these."""
    big = _supercell(cell, matrix)
    positions = big["positions"]
    n, size = len(cell), len(big)
    if operations is None:
        rotations = np.eye(3, dtype=np.int64)[None]
        cartesian = np.eye(3)[None]
        atoms = np.arange(n)[None]
        shifts = np.zeros((1, n, 3), dtype=np.int64)
    else:
        # only the operations that map the supercell onto itself relate its
        # force constants
        keep = _keeping(operations.rotations, matrix)
        rotations, cartesian = operations.rotations[keep], operations.cartesian[keep]
        atoms, shifts = operations.atoms[keep], operations.shifts[keep]
    repeats = _lattice_points(matrix)
    found = {tuple(r): k for k, r in enumerate(repeats.tolist())}
    inverse = np.linalg.inv(matrix)

    def image(k: int, a: int) -> np.ndarray:
        """(n S,) the atom of the supercell that operation k takes each atom
        to, the image of atom a moved back into the first cell."""
        lattice = (repeats @ rotations[k].T)[:, None, :] + shifts[k] - shifts[k, a]
        fractional = lattice.reshape(-1, 3) @ inverse
        inside = np.round((fractional - np.floor(fractional + 1e-9)) @ matrix)
        cells = np.array([found[tuple(r)] for r in inside.astype(int).tolist()])
        return cells * n + np.tile(atoms[k], len(repeats))

    def forces(a: int, alpha: int, sign: float) -> np.ndarray:
        moved = positions.copy()
        moved[a, alpha] += sign * displacement
        return calculator.compute(big.with_columns(positions=moved), ("forces",)).forces

    def rank(vectors: list) -> int:
        return int(np.linalg.matrix_rank(np.array(vectors), tol=1e-6)) if vectors else 0

    constants = np.zeros((n, size, 3, 3))
    done = np.zeros(n, dtype=bool)
    for a in range(n):
        if done[a]:
            continue
        site = np.flatnonzero(atoms[:, a] == a)
        # the axes to move along: each adds directions to the span of the others
        axes, span = [], []
        for alpha in range(3):
            images = list(cartesian[site, :, alpha])
            if rank(span + images) > rank(span):
                axes.append(alpha)
                span += images
        directions, responses = [], []
        for alpha in axes:
            plus = forces(a, alpha, 1.0)
            flip = [k for k in site if cartesian[k, alpha, alpha] < -1 + 1e-6]
            if flip:
                minus = np.empty_like(plus)
                minus[image(flip[0], a)] = plus @ cartesian[flip[0]].T
            else:
                minus = forces(a, alpha, -1.0)
            response = -(plus - minus) / (2 * displacement)  # row alpha of each 3 x 3
            for k in site:
                directions.append(cartesian[k, :, alpha])
                moved = np.empty_like(response)
                moved[image(k, a)] = response @ cartesian[k].T
                responses.append(moved)
        constants[a] = np.einsum(
            "am,mjb->jab", np.linalg.pinv(np.array(directions)), np.array(responses)
        )
        # the atoms equivalent to a: images of its force constants
        for b in np.unique(atoms[:, a]):
            if b != a:
                k = np.flatnonzero(atoms[:, a] == b)[0]
                rotation = cartesian[k]
                constants[b, image(k, a)] = rotation @ constants[a] @ rotation.T
                done[b] = True
    return constants


def _symmetrized(constants: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Force constants made symmetric in the two atoms and summing to zero
    over the atoms of the supercell, so that a translation costs nothing."""
    n = constants.shape[0]
    repeats = _lattice_points(matrix)
    # the pair (b in cell L, a in cell 0) is the pair (a in cell -L, b in cell 0)
    fractional = -repeats @ np.linalg.inv(matrix)
    opposite = np.round((fractional - np.floor(fractional + 1e-9)) @ matrix).astype(int)
    found = {tuple(r): k for k, r in enumerate(repeats.tolist())}
    index = np.array([found[tuple(r)] for r in opposite.tolist()])
    j = np.arange(constants.shape[1])
    other = index[j // n][None, :] * n + np.arange(n)[:, None]  # (n, M)
    partner = j % n
    constants = constants.copy()
    for _ in range(10):
        transposed = constants[partner[None, :], other].transpose(0, 1, 3, 2)
        constants = 0.5 * (constants + transposed)
        constants[np.arange(n), np.arange(n)] -= constants.sum(axis=1)
    return constants


def phonons(
    system: System,
    calculator: Calculator,
    *,
    supercell: int | ArrayLike | None = None,
    displacement: float = 0.01,
    symmetry: bool | None = None,
) -> Phonons:
    """Harmonic phonons of a crystal, from the forces on atoms displaced in
    a supercell.

    Parameters
    ----------
    system : System
        The cell of the crystal, periodic in every direction, with atoms at
        rest (relax it first); best its primitive cell. Each atom needs an
        element or a mass.
    calculator : Calculator
        Computes the forces.
    supercell : int or array_like, optional
        Repeats of the cell along its vectors, one or three, or a (3, 3)
        integer matrix whose rows times the vectors of the cell are the
        vectors of the supercell. Default the supercell nearest a cube that
        is at least twice the cutoff of the calculator thick in every
        direction.
    displacement : float, optional
        Displacement of the atoms in Å. Default 0.01.
    symmetry : bool, optional
        Move only the atoms and along the directions that the symmetry of
        the cell, found by spglib, does not relate to others: True, False,
        or by default whenever spglib is installed.

    Returns
    -------
    Phonons
        The force constants, which give frequencies, polarization vectors,
        band structures, densities of states and thermal properties.

    Raises
    ------
    InvalidParameter
        If the box is not periodic, there are no atoms, an atom has no
        mass, the calculator does not compute forces, the supercell is not
        positive integers or a matrix of them with a positive determinant,
        the displacement is not positive, or symmetry is True without
        spglib.
    """
    return _phonons(system, calculator, supercell, displacement, symmetry)


def _phonons(
    system: System,
    calculator: Calculator,
    supercell: int | ArrayLike | None,
    displacement: float,
    symmetry: bool | None,
    reference: System | None = None,
) -> Phonons:
    """The phonons of :func:`phonons`, the periodic images of the force
    constants picked by the geometry of reference."""
    if not system.box.pbc.all() or len(system) == 0:
        raise InvalidParameter(
            "phonons need a box periodic in every direction with atoms"
        )
    if not displacement > 0:
        raise InvalidParameter(f"displacement must be positive, not {displacement!r}")
    if "forces" not in calculator.implemented:
        raise InvalidParameter(f"{type(calculator).__name__} does not compute forces")
    masses = _masses(system)
    if supercell is None:
        matrix = _default_supercell(system.box.matrix, 2 * calculator.cutoff)
    else:
        matrix = _supercell_matrix(supercell)
    operations = _symmetry(system, symmetry)
    constants = _displaced_forces(system, calculator, matrix, displacement, operations)
    return Phonons(
        system, matrix, _symmetrized(constants, matrix), masses, reference=reference
    )
