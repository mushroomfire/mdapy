# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Training structures that cover descriptor space: small periodic cells,
some of them slabs or clusters in vacuum, whose atoms are moved to raise the
entropy of the set."""

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
# the least vacuum in Å between the atoms of a slab or cluster and their
# images; never less than the reach of the descriptor
_VACUUM = 10.0
_PERIODIC = np.ones(3, dtype=bool)
_ORIGIN = np.zeros(3)
# the axes along which a cell has vacuum: none, the third (a slab), all
_BULK = np.zeros(3, dtype=bool)
_SLAB = np.array([False, False, True])
_CLUSTER = np.ones(3, dtype=bool)


class _Cell:
    """A periodic cell of the generation: its atoms, the box of the atoms
    (core) and the axes along which vacuum of gap Å is added to it, the
    elements, as their indices, and the pairs within the reach."""

    __slots__ = ("positions", "core", "vacuum", "matrix", "kinds", "pairs")

    def __init__(self, positions, core, kinds, reach, vacuum=_BULK, gap=0.0):
        self.positions = np.ascontiguousarray(positions, dtype=np.float64)
        self.core = np.ascontiguousarray(core, dtype=np.float64)
        self.kinds = np.ascontiguousarray(kinds, dtype=np.int32)
        self.vacuum = vacuum
        matrix = self.core.copy()
        if vacuum.any():
            # each edge with vacuum lengthened so that the planes across it
            # move apart by the gap
            thickness = Box(self.core).thickness
            matrix[vacuum] *= (1.0 + gap / thickness[vacuum])[:, None]
        self.matrix = matrix
        self.pairs = _neighbor.build(
            self.positions, self.matrix, _ORIGIN, _PERIODIC, reach, None, None
        )


class _Generator:
    """Random cells of the elements, the given fraction of them slabs or
    clusters, and moves of them, no two atoms closer than closest times the
    sum of their covalent radii, and the descriptor of their atoms."""

    def __init__(self, elements, atoms, volume, closest, rng, vacuum=0.0):
        self.elements = elements
        self.radii = np.array([COVALENT_RADII[SYMBOLS.index(e)] for e in elements])
        self.atoms, self.volume, self.closest, self.rng = atoms, volume, closest, rng
        self.vacuum = vacuum
        self.reach = float(2.0 * self.radii.max() * _BASIS[4])
        self.gap = max(_VACUUM, self.reach)

    def __cell(self, positions, core, kinds, vacuum=_BULK) -> _Cell:
        return _Cell(positions, core, kinds, self.reach, vacuum, self.gap)

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

    def shape_ok(
        self, matrix: np.ndarray, count: int | None = None, slab: bool = False
    ) -> bool:
        """No face of the box much narrower than its longest edge; with
        count, a volume per atom in the range, to its rounding; for a slab,
        at least the reach of the descriptor across its third edge."""
        if count is not None:
            volume = abs(float(np.linalg.det(matrix))) / count
            slack = 1e-12 * self.volume[1]
            if not self.volume[0] - slack <= volume <= self.volume[1] + slack:
                return False
        lengths = np.linalg.norm(matrix, axis=1)
        thickness = Box(matrix).thickness
        if slab and thickness[2] < self.reach * (1.0 - 1e-12):
            return False
        return bool(lengths.max() <= 4.0 * thickness.min())

    def random(self) -> _Cell:
        rng = self.rng
        kind = _BULK
        if self.vacuum and rng.random() < self.vacuum:
            kind = _SLAB if rng.random() < 0.5 else _CLUSTER
        slab = kind is _SLAB
        for _ in range(_ATTEMPTS):
            # as many cells of each doubling of the number of atoms
            low, high = math.log(self.atoms[0]), math.log(self.atoms[1] + 1)
            n = min(int(math.exp(rng.uniform(low, high))), self.atoms[1])
            if len(self.elements) > 1:
                fractions = rng.dirichlet(np.ones(len(self.elements)))
                kinds = rng.choice(len(self.elements), size=n, p=fractions)
            else:
                kinds = np.zeros(n, dtype=np.int32)
            volume = math.exp(rng.uniform(*np.log(self.volume)))
            matrix = np.diag(np.exp(rng.uniform(-0.35, 0.35, 3)))
            matrix[1, 0], matrix[2, 0], matrix[2, 1] = rng.uniform(-0.3, 0.3, 3)
            matrix *= (n * volume / np.linalg.det(matrix)) ** (1.0 / 3.0)
            if slab:
                # at least the reach thick, so that the atoms of a surface
                # have others beneath them: longer along the third edge,
                # narrower along the others, the volume kept
                stretch = self.reach / Box(matrix).thickness[2]
                if stretch > 1.0:
                    matrix[2] *= stretch
                    matrix[:2] /= math.sqrt(stretch)
            if not self.shape_ok(matrix, slab=slab):
                continue
            positions = self.__place(matrix, kinds)
            if positions is None:
                positions = self.__grid(matrix, kinds)
            if positions is not None:
                # placed apart in the periodic cell, also from the images of
                # each atom; vacuum around a slab or cluster only adds room
                if self.apart(self.__cell(positions, matrix, kinds)):
                    return self.__cell(positions, matrix, kinds, kind)
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

    def __grid(self, matrix: np.ndarray, kinds: np.ndarray) -> np.ndarray | None:
        """The atoms on random points of a grid of the cell with at least
        as many points, each moved at random by up to a fifth of a step, or
        less when they come too close; None when even the grid does."""
        n = len(kinds)
        thickness = Box(matrix).thickness
        counts = np.maximum(1, np.round(thickness * (n / thickness.prod()) ** (1 / 3)))
        while counts.prod() < n:
            counts[np.argmin(thickness / counts)] += 1
        points = np.stack(
            np.meshgrid(*[np.arange(c) for c in counts], indexing="ij"), axis=-1
        ).reshape(-1, 3)
        chosen = points[self.rng.choice(len(points), n, replace=False)]
        jitter = self.rng.uniform(-1.0, 1.0, (n, 3))
        for size in (0.2, 0.1, 0.05, 0.0):
            positions = ((chosen + 0.5 + size * jitter) / counts) @ matrix
            if self.apart(self.__cell(positions, matrix, kinds)):
                return positions
        return None

    def move(self, cell: _Cell) -> _Cell | None:
        """The cell with two atoms of other elements swapped, its box of the
        atoms strained, or an atom displaced; None for a box out of range or
        an atom of a slab or cluster leaving its box into the vacuum."""
        rng = self.rng
        n = len(cell.kinds)
        choice = rng.random()
        if _STRAIN <= choice < _STRAIN + _SWAP and len(np.unique(cell.kinds)) > 1:
            a = int(rng.integers(n))
            b = int(rng.choice(np.flatnonzero(cell.kinds != cell.kinds[a])))
            kinds = cell.kinds.copy()
            kinds[a], kinds[b] = cell.kinds[b], cell.kinds[a]
            return self.__cell(cell.positions, cell.core, kinds, cell.vacuum)
        if choice < _STRAIN:
            e = rng.normal(0.0, 0.03, (3, 3))
            deformation = np.eye(3) + 0.5 * (e + e.T)
            core = cell.core @ deformation
            slab = bool(cell.vacuum[2] and not cell.vacuum[:2].any())
            if not self.shape_ok(core, n, slab=slab):
                return None
            positions = cell.positions @ deformation
            return self.__cell(positions, core, cell.kinds, cell.vacuum)
        a = int(rng.integers(n))
        positions = cell.positions.copy()
        positions[a] += rng.normal(0.0, 0.3 * self.radii.mean(), 3)
        if cell.vacuum.any():
            f = positions[a] @ np.linalg.inv(cell.core)
            if ((f < 0.0) | (f >= 1.0))[cell.vacuum].any():
                return None
        return self.__cell(positions, cell.core, cell.kinds, cell.vacuum)

    def system(self, cell: _Cell) -> System:
        """The cell as a system, its atoms wrapped into the box, and its
        kind, bulk, slab or cluster, as the config_type of its info."""
        fractional = cell.positions @ np.linalg.inv(cell.matrix)
        fractional -= np.floor(fractional)
        kind = ("bulk", "slab", "slab", "cluster")[int(cell.vacuum.sum())]
        return System(
            fractional @ cell.matrix,
            Box(cell.matrix),
            elements=[self.elements[k] for k in cell.kinds],
            info={"config_type": kind},
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
    atoms: tuple[int, int] = (2, 32),
    vacuum: float = 0.2,
    closest: float = 0.75,
    per: str = "atom",
    steps: int = 10,
    reference: Sequence[System] | None = None,
    calculator: Any = None,
    max_energy: float | None = None,
    tolerance: float = 0.0,
    seed: int | np.random.Generator | None = None,
    progress: Any = None,
    cancel: Any = None,
) -> list[System]:
    """Small periodic cells, some of them slabs or clusters in vacuum, whose
    atoms cover descriptor space, for a training set: each a random cell
    whose atoms, box and elements are moved while they raise the entropy of
    the set, kept if it rises.

    Parameters
    ----------
    elements : str or sequence of str
        The elements; with several, each cell has a random composition.
    count : int32
        The most structures to generate.
    volume : tuple of float64
        The smallest and largest volume per atom in Å³, of the atoms' box
        without vacuum.
    atoms : tuple of int32, optional
        The fewest and most atoms of a cell, with as many cells of each
        doubling of the number. Default (2, 32).
    vacuum : float64, optional
        The fraction of the cells that are slabs, with vacuum along the
        third axis, or clusters, with vacuum along all three, half of each.
        Default 0.2.
    closest : float64, optional
        The shortest distance of two atoms as a fraction of the sum of their
        covalent radii. Default 0.75.
    per : {"atom", "structure"}, optional
        The points of the set: the descriptor of each atom, or their mean
        in each structure. Default atom.
    steps : int32, optional
        The moves tried on each structure per atom. Default 10.
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
        isinstance(vacuum, bool)
        or not isinstance(vacuum, int | float | np.number)
        or not 0 <= vacuum <= 1
    ):
        raise InvalidParameter(f"vacuum must be from 0 to 1, not {vacuum!r}")
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
        names, atoms, volume, float(closest), np.random.default_rng(seed),
        float(vacuum),
    )  # fmt: skip

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
            for _ in range(int(steps) * len(cell.kinds)):
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
