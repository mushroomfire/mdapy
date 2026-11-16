# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Training structures that cover descriptor space: small periodic cells
whose atoms are moved to raise the entropy of the set."""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from mdapy._core import analysis as _analysis
from mdapy._core import neighbor as _neighbor
from mdapy.core.box import Box
from mdapy.core.system import System
from mdapy.core.threads import _native_token
from mdapy.elements import COVALENT_RADII, SYMBOLS
from mdapy.errors import Cancelled, InvalidParameter, MissingProperty
from mdapy.neighbor import NeighborList

# candidates in a row that add nothing before the generation stops
_PATIENCE = 100
# the window of candidates over which the fraction kept is measured
_WINDOW = 50
# random cells that set the scale of each value of the descriptor
_PILOT = 20
# random cells tried in a row before the volume is found too small
_ATTEMPTS = 200
# the fractions of the moves that strain the box and, with atoms of several
# elements, that swap two of them; the others displace an atom
_STRAIN = 0.2
_SWAP = 0.1
# the basis: radial functions, those of the angular sums, the highest degree,
# the radial functions of each element, and the radial and angular cutoffs
# of the distance over the sum of the covalent radii of the pair
_BASIS = (8, 4, 4, 2, 2.4, 1.7)
_PERIODIC = np.ones(3, dtype=bool)
_ORIGIN = np.zeros(3)


class _Cell:
    """A periodic cell of the generation: its atoms, box and elements, as
    the indices of the elements, and its pairs within the reach."""

    __slots__ = ("positions", "matrix", "kinds", "pairs")

    def __init__(self, positions, matrix, kinds, reach):
        self.positions = np.ascontiguousarray(positions, dtype=np.float64)
        self.matrix = np.ascontiguousarray(matrix, dtype=np.float64)
        self.kinds = np.ascontiguousarray(kinds, dtype=np.int32)
        self.pairs = _neighbor.build(
            self.positions, self.matrix, _ORIGIN, _PERIODIC, reach, None, None
        )


class _Generator:
    """Random cells of the elements and moves of them, no two atoms closer
    than closest times the sum of their covalent radii, and the descriptor
    of their atoms."""

    def __init__(self, elements, atoms, volume, closest, rng):
        self.elements = elements
        self.radii = np.array([COVALENT_RADII[SYMBOLS.index(e)] for e in elements])
        self.atoms, self.volume, self.closest, self.rng = atoms, volume, closest, rng
        self.reach = float(2.0 * self.radii.max() * _BASIS[4])

    def descriptors(self, cell: _Cell) -> np.ndarray:
        offsets, j, shifts, _ = cell.pairs
        return _analysis.scaled_basis(
            cell.kinds, len(self.elements), self.radii, cell.positions, cell.matrix,
            offsets, j, shifts, *_BASIS, None, None,
        )  # fmt: skip

    def frame_descriptors(self, system: System) -> np.ndarray:
        """The descriptor of the atoms of any system of the elements."""
        index = {name: k for k, name in enumerate(self.elements)}
        kinds = np.array([index[e] for e in system["element"]], dtype=np.int32)
        neighbors = NeighborList.build(system, self.reach)
        with neighbors._reading():
            return _analysis.scaled_basis(
                kinds, len(self.elements), self.radii,
                np.ascontiguousarray(system["positions"]),
                np.ascontiguousarray(system.box.matrix), neighbors.offsets,
                np.ascontiguousarray(neighbors.j),
                np.ascontiguousarray(neighbors.shifts), *_BASIS, None, None,
            )  # fmt: skip

    def apart(self, cell: _Cell) -> bool:
        offsets, j, _, distances = cell.pairs
        r = self.radii[cell.kinds]
        i = np.repeat(cell.kinds, np.diff(offsets))
        return bool((distances >= self.closest * (self.radii[i] + r[j])).all())

    def shape_ok(self, matrix: np.ndarray, count: int | None = None) -> bool:
        """No face of the box much narrower than its longest edge, and with
        count, a volume per atom in the range, to its rounding."""
        if count is not None:
            volume = abs(float(np.linalg.det(matrix))) / count
            slack = 1e-12 * self.volume[1]
            if not self.volume[0] - slack <= volume <= self.volume[1] + slack:
                return False
        lengths = np.linalg.norm(matrix, axis=1)
        return bool(lengths.max() <= 4.0 * Box(matrix).thickness.min())

    def random(self) -> _Cell:
        rng = self.rng
        for _ in range(_ATTEMPTS):
            n = int(rng.integers(self.atoms[0], self.atoms[1] + 1))
            if len(self.elements) > 1:
                fractions = rng.dirichlet(np.ones(len(self.elements)))
                kinds = rng.choice(len(self.elements), size=n, p=fractions)
            else:
                kinds = np.zeros(n, dtype=np.int32)
            low, high = np.log(self.volume)
            volume = math.exp(rng.uniform(low, high))
            matrix = np.diag(np.exp(rng.uniform(-0.35, 0.35, 3)))
            matrix[1, 0], matrix[2, 0], matrix[2, 1] = rng.uniform(-0.3, 0.3, 3)
            matrix *= (n * volume / np.linalg.det(matrix)) ** (1.0 / 3.0)
            if not self.shape_ok(matrix):
                continue
            positions = self.__place(matrix, kinds)
            if positions is not None:
                cell = _Cell(positions, matrix, kinds, self.reach)
                if self.apart(cell):  # also from the images of each atom
                    return cell
        raise InvalidParameter(
            f"no random cell of {self.atoms[0]} to {self.atoms[1]} atoms fits in "
            f"{_ATTEMPTS} attempts: the volume is too small for closest"
        )

    def __place(self, matrix: np.ndarray, kinds: np.ndarray) -> np.ndarray | None:
        inverse = np.linalg.inv(matrix)
        placed = np.zeros((0, 3))
        for a, kind in enumerate(kinds):
            for _ in range(200):
                p = self.rng.random(3) @ matrix
                f = (p - placed) @ inverse
                f -= np.round(f)
                d = np.linalg.norm(f @ matrix, axis=1)
                if (
                    d >= self.closest * (self.radii[kind] + self.radii[kinds[:a]])
                ).all():
                    placed = np.vstack([placed, p])
                    break
            else:
                return None
        return placed

    def move(self, cell: _Cell) -> _Cell | None:
        """The cell with two atoms of other elements swapped, its box
        strained, or an atom displaced; None for a box out of range."""
        rng = self.rng
        n = len(cell.kinds)
        choice = rng.random()
        if _STRAIN <= choice < _STRAIN + _SWAP and len(np.unique(cell.kinds)) > 1:
            a = int(rng.integers(n))
            b = int(rng.choice(np.flatnonzero(cell.kinds != cell.kinds[a])))
            kinds = cell.kinds.copy()
            kinds[a], kinds[b] = cell.kinds[b], cell.kinds[a]
            return _Cell(cell.positions, cell.matrix, kinds, self.reach)
        if choice < _STRAIN:
            e = rng.normal(0.0, 0.03, (3, 3))
            deformation = np.eye(3) + 0.5 * (e + e.T)
            matrix = cell.matrix @ deformation
            if not self.shape_ok(matrix, n):
                return None
            positions = cell.positions @ deformation
            return _Cell(positions, matrix, cell.kinds, self.reach)
        a = int(rng.integers(n))
        positions = cell.positions.copy()
        positions[a] += rng.normal(0.0, 0.3 * self.radii.mean(), 3)
        return _Cell(positions, cell.matrix, cell.kinds, self.reach)

    def system(self, cell: _Cell) -> System:
        """The cell as a system, its atoms wrapped into the box."""
        fractional = cell.positions @ np.linalg.inv(cell.matrix)
        fractional -= np.floor(fractional)
        return System(
            fractional @ cell.matrix,
            Box(cell.matrix),
            elements=[self.elements[k] for k in cell.kinds],
        )


class _Entropy:
    """Half the log determinant of the covariance of a set of points,
    regularized, as the entropy of a Gaussian of them; a candidate is kept
    when it raises it."""

    def __init__(self, dimension: int):
        self.first = np.zeros(dimension)
        self.second = np.zeros((dimension, dimension))
        self.count = 0

    @staticmethod
    def __value(first: np.ndarray, second: np.ndarray, count: int) -> float:
        mean = first / count
        covariance = second / count - np.outer(mean, mean)
        covariance += 1e-3 * np.eye(len(mean))
        return 0.5 * float(np.linalg.slogdet(covariance)[1])

    def with_points(self, z: np.ndarray) -> float:
        return self.__value(
            self.first + z.sum(axis=0), self.second + z.T @ z, self.count + len(z)
        )

    def raised_by(self, z: np.ndarray) -> bool:
        if self.count == 0:
            return True
        return self.with_points(z) > self.__value(self.first, self.second, self.count)

    def add(self, z: np.ndarray) -> None:
        self.first += z.sum(axis=0)
        self.second += z.T @ z
        self.count += len(z)


def _element_list(elements: str | Sequence[str]) -> tuple[str, ...]:
    names = (elements,) if isinstance(elements, str) else tuple(elements)
    if not names:
        raise InvalidParameter("give at least one element")
    for name in names:
        if name not in SYMBOLS[1:] or not np.isfinite(
            COVALENT_RADII[SYMBOLS.index(name)]
        ):
            raise InvalidParameter(f"{name!r} is not an element with a covalent radius")
    if len(set(names)) != len(names):
        raise InvalidParameter("an element is given twice")
    return names


def _range(value: Any, name: str, integer: bool) -> tuple:
    try:
        low, high = value
    except (TypeError, ValueError):
        raise InvalidParameter(f"{name} must be a pair (smallest, largest)") from None
    kind = int | np.integer if integer else int | float | np.number
    for v in (low, high):
        if isinstance(v, bool) or not isinstance(v, kind):
            raise InvalidParameter(f"{name} must hold two numbers, not {value!r}")
    if not 0 < low <= high < math.inf:
        raise InvalidParameter(f"{name} must be 0 < smallest <= largest, not {value!r}")
    return (int(low), int(high)) if integer else (float(low), float(high))


def generate_structures(
    elements: str | Sequence[str],
    count: int,
    *,
    volume: tuple[float, float],
    atoms: tuple[int, int] = (2, 24),
    closest: float = 0.75,
    per: str = "atom",
    steps: int = 200,
    reference: Sequence[System] | None = None,
    calculator: Any = None,
    max_energy: float | None = None,
    tolerance: float = 0.0,
    seed: int | np.random.Generator | None = None,
    progress: Any = None,
    cancel: Any = None,
) -> list[System]:
    """Small periodic cells whose atoms cover descriptor space, for a
    training set: each a random cell whose atoms, box and elements are moved
    while they raise the entropy of the set, kept if it rises.

    Parameters
    ----------
    elements : str or sequence of str
        The elements; with several, each cell has a random composition.
    count : int32
        The most structures to generate.
    volume : tuple of float64
        The smallest and largest volume per atom in Å³.
    atoms : tuple of int32, optional
        The fewest and most atoms of a cell. Default (2, 24).
    closest : float64, optional
        The shortest distance of two atoms as a fraction of the sum of their
        covalent radii. Default 0.75.
    per : {"atom", "structure"}, optional
        The points of the set: the descriptor of each atom, or their mean
        in each structure. Default atom.
    steps : int32, optional
        The moves tried on each structure. Default 200.
    reference : sequence of System, optional
        A training set of the elements for the structures to complement.
    calculator : Calculator, optional
        Computes the energy of each structure, with ``max_energy``.
    max_energy : float64, optional
        The highest energy per atom in eV a structure may have.
    tolerance : float64, optional
        Stop when fewer than this fraction of the last 50 candidates was
        kept. Default 0: only 100 candidates in a row not kept stop it.
    seed : int64 or numpy.random.Generator, optional
        Seed of the random numbers.

    Returns
    -------
    list of System
        The structures in the order they were kept, so that the first n are
        a set of their own; fewer than count when the generation stops.
    """
    names = _element_list(elements)
    if isinstance(count, bool) or not isinstance(count, int | np.integer) or count < 0:
        raise InvalidParameter(f"count must be a non-negative integer, not {count!r}")
    volume = _range(volume, "volume", integer=False)
    atoms = _range(atoms, "atoms", integer=True)
    if (
        isinstance(closest, bool)
        or not isinstance(closest, int | float | np.number)
        or not 0 < closest < math.inf
    ):
        raise InvalidParameter(f"closest must be above zero, not {closest!r}")
    if per not in ("atom", "structure"):
        raise InvalidParameter(f"per must be atom or structure, not {per!r}")
    if isinstance(steps, bool) or not isinstance(steps, int | np.integer) or steps < 0:
        raise InvalidParameter(f"steps must be a non-negative integer, not {steps!r}")
    if (calculator is None) != (max_energy is None):
        raise InvalidParameter("give calculator and max_energy together")
    if not (isinstance(tolerance, int | float | np.number) and 0 <= tolerance < 1):
        raise InvalidParameter(
            f"tolerance must be from 0 to below 1, not {tolerance!r}"
        )
    _native_token(cancel)  # checks its type
    frames = list(reference or ())
    for frame in frames:
        if not isinstance(frame, System):
            raise InvalidParameter("reference must be a sequence of System")
        if "element" not in frame.table.columns:
            raise MissingProperty("each frame of reference needs an element column")
        missing = sorted(set(frame["element"]) - set(names))
        if missing:
            raise InvalidParameter(f"reference has atoms of {missing}, not given")
    generator = _Generator(
        names, atoms, volume, float(closest), np.random.default_rng(seed)
    )

    def low_enough(cell: _Cell) -> bool:
        if calculator is None:
            return True
        found = calculator.compute(generator.system(cell), ("energy",))
        return found.energy / len(cell.kinds) <= max_energy

    def per_structure(x: np.ndarray) -> np.ndarray:
        return x.mean(axis=0, keepdims=True) if per == "structure" else x

    # the scale of each value, from random cells
    pilot = np.concatenate(
        [
            per_structure(generator.descriptors(generator.random()))
            for _ in range(_PILOT)
        ]
    )
    scale = pilot.std(axis=0)
    scale[scale == 0] = 1.0
    entropy = _Entropy(len(scale))
    for frame in frames:
        entropy.add(per_structure(generator.frame_descriptors(frame)) / scale)

    kept: list[System] = []
    history: list[bool] = []
    while len(kept) < count:
        if cancel is not None and cancel.cancelled:
            raise Cancelled("the generation of structures was cancelled")
        cell = generator.random()
        if not low_enough(cell):
            continue
        z = per_structure(generator.descriptors(cell)) / scale
        if entropy.count:
            best = entropy.with_points(z)
            for _ in range(int(steps)):
                trial = generator.move(cell)
                if trial is None or not generator.apart(trial):
                    continue
                zt = per_structure(generator.descriptors(trial)) / scale
                value = entropy.with_points(zt)
                if value > best and low_enough(trial):
                    cell, z, best = trial, zt, value
        keep = entropy.raised_by(z)
        history.append(keep)
        if keep:
            entropy.add(z)
            kept.append(generator.system(cell))
            if progress is not None:
                progress(len(kept), count)
        recent = history[-_WINDOW:]
        if len(recent) == _WINDOW and sum(recent) < tolerance * _WINDOW:
            break
        if len(history) >= _PATIENCE and not any(history[-_PATIENCE:]):
            break
    return kept
