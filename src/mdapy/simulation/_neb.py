# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The minimum energy path between two states by the nudged elastic band,
with the improved tangent of Henkelman and Jónsson (2000) and the climbing
image of Henkelman, Uberuaga and Jónsson (2000), as LAMMPS's neb."""

import math
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

from mdapy.calculators._base import Calculator
from mdapy.core.result import Result
from mdapy.core.system import System
from mdapy.errors import CalculationFailed, InvalidParameter
from mdapy.neighbor import NeighborList
from mdapy.simulation._log import _Log
from mdapy.simulation._minimize import _FTM2V, _Fire, _masses, _Neighbors

#: How the images between the two states start.
INTERPOLATIONS = ("linear", "idpp")

# the columns of the log: name, format, width
_LOG_COLUMNS = (
    ("step", "{:d}", 10),
    ("max_force", "{:.6e}", 14),
    ("barrier", "{:.8f}", 16),
    ("highest", "{:d}", 8),
)

# The image dependent pair potential of Smidstrup et al. (2014): the pairs
# within this distance, in Å, in either state; its band relaxes until no
# force exceeds this, in 1/Å³, or for this many steps.
_IDPP_CUTOFF = 6.0
_IDPP_MAX_FORCE = 0.01
_IDPP_MAX_STEPS = 500


@dataclass(frozen=True)
class NEBResult(Result):
    """The outcome of ``neb``.

    Attributes
    ----------
    images : tuple of System
        The band, the two states first and last, each with a ``forces``
        column and ``energy`` in ``info``: the forces of the calculator, not
        those of the band.
    energies : numpy.ndarray
        (M,) float64 energy of each image in eV.
    distances : numpy.ndarray
        (M,) float64 distance of each image from the first along the band, in
        Å: the sums of the lengths of the steps between images, over all
        atoms.
    barrier : float64
        Highest energy of the band less that of the first state, in eV.
    highest : int32
        Index of the image of highest energy; with ``climb``, it is on the
        saddle point.
    converged : bool
        Whether no force of the band on an atom of the images between the
        states exceeds ``max_force``.
    steps : int32
        Number of steps taken.
    max_force : float64
        Largest force of the band on an atom of the images between the
        states, in eV/Å.
    """

    images: tuple[System, ...]
    energies: np.ndarray
    distances: np.ndarray
    barrier: float
    highest: int
    converged: bool
    steps: int
    max_force: float

    def profile(self, points: int = 200) -> tuple[np.ndarray, np.ndarray]:
        """The energy along the band between the images: a cubic between each
        two, through their energies, with the slopes the forces give along
        the band.

        Parameters
        ----------
        points : int32, optional
            Number of points, from the first image to the last. Default 200.

        Returns
        -------
        distances : numpy.ndarray
            (points,) float64 distance along the band in Å.
        energies : numpy.ndarray
            (points,) float64 energy in eV.

        Raises
        ------
        InvalidParameter
            If ``points`` is less than 2.
        """
        if isinstance(points, bool) or not isinstance(points, int) or points < 2:
            raise InvalidParameter(
                f"points must be an integer of 2 or more, not {points!r}"
            )
        band = np.stack([image["positions"] for image in self.images])
        forces = np.stack([image["forces"] for image in self.images])
        # the direction of the band at each image: to its neighbors, or to
        # its one neighbor at the ends
        ahead = np.concatenate([band[1:2], band[2:], band[-1:]])
        behind = np.concatenate([band[:1], band[:-2], band[-2:-1]])
        directions = ahead - behind
        lengths = np.sqrt((directions * directions).sum(axis=(1, 2)))
        slopes = -(forces * directions).sum(axis=(1, 2)) / np.where(
            lengths > 0, lengths, 1.0
        )
        s, e = self.distances, self.energies
        grid = np.linspace(s[0], s[-1], points)
        k = np.clip(np.searchsorted(s, grid, side="right") - 1, 0, len(s) - 2)
        h = s[k + 1] - s[k]
        u = np.where(h > 0, (grid - s[k]) / np.where(h > 0, h, 1.0), 0.0)
        # cubic Hermite polynomials
        h00 = 2 * u**3 - 3 * u**2 + 1
        h10 = u**3 - 2 * u**2 + u
        h01 = -2 * u**3 + 3 * u**2
        h11 = u**3 - u**2
        values = (
            h00 * e[k] + h10 * h * slopes[k] + h01 * e[k + 1] + h11 * h * slopes[k + 1]
        )
        return grid, values


def _tangent(
    plus: np.ndarray, minus: np.ndarray, before: float, here: float, after: float
) -> np.ndarray:
    """The unit tangent of an image, from the steps to the next image and
    from the previous, and the energies of the three."""
    if after > here > before:
        tangent = plus
    elif after < here < before:
        tangent = minus
    else:
        # at an extremum, mixed by the energies so that the tangent turns
        # smoothly between the two
        big = max(abs(after - here), abs(before - here))
        small = min(abs(after - here), abs(before - here))
        if after > before:
            tangent = big * plus + small * minus
        elif after < before:
            tangent = small * plus + big * minus
        else:
            tangent = plus + minus
    length = math.sqrt(float((tangent * tangent).sum()))
    return tangent / length if length > 0.0 else tangent


def _band_forces(
    band: np.ndarray,
    energies: np.ndarray,
    forces: np.ndarray,
    spring: float,
    climber: int | None,
) -> np.ndarray:
    """The forces of the band on the images between the states: the forces
    across the band and the springs along it, or for the climbing image the
    forces with their part along the band reversed."""
    out = np.empty_like(forces)
    for k in range(1, len(band) - 1):
        plus, minus = band[k + 1] - band[k], band[k] - band[k - 1]
        tangent = _tangent(plus, minus, energies[k - 1], energies[k], energies[k + 1])
        force = forces[k - 1]
        along = float((force * tangent).sum())
        if k == climber:
            out[k - 1] = force - 2.0 * along * tangent
        else:
            stretch = math.sqrt(float((plus * plus).sum())) - math.sqrt(
                float((minus * minus).sum())
            )
            out[k - 1] = force - along * tangent + spring * stretch * tangent
    return out


def _largest(forces: np.ndarray) -> float:
    """The largest force on an atom of rows of (N, 3) forces."""
    return float(np.sqrt((forces * forces).sum(axis=-1)).max(initial=0.0))


def _idpp(
    band: np.ndarray,
    initial: System,
    spring: float,
    movable: np.ndarray | None,
) -> np.ndarray:
    """The band relaxed on the image dependent pair potential: the distances
    of each pair of atoms of each image near those interpolated between the
    states, closer pairs weighing more."""
    first, last = band[0], band[-1]
    pairs: dict[tuple[int, int, int, int, int], None] = {}
    for positions in (first, last):
        state = initial.with_columns(positions=positions)
        neighbors = NeighborList.build(state, _IDPP_CUTOFF)
        for i, j, shift in zip(neighbors.i, neighbors.j, neighbors.shifts):
            if i < j:
                pairs[(int(i), int(j), *map(int, shift))] = None
    if not pairs:
        return band
    keys = np.array(list(pairs), dtype=np.int64)
    i, j = keys[:, 0], keys[:, 1]
    offsets = keys[:, 2:].astype(float) @ initial.box.matrix

    def pair_vectors(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vectors = positions[j] - positions[i] + offsets
        return vectors, np.sqrt((vectors * vectors).sum(axis=1))

    start, end = pair_vectors(first)[1], pair_vectors(last)[1]
    count = len(band) - 1

    def evaluate(moving: np.ndarray):
        # the energies of the band and the forces on its moving images
        images = moving.reshape(count - 1, *first.shape)
        energies = np.empty(count + 1)
        forces = np.empty_like(images)
        for k, positions in enumerate(images, start=1):
            vectors, d = pair_vectors(positions)
            target = start + k / count * (end - start)
            weight = d**-4
            energies[k] = float((weight * (target - d) ** 2).sum())
            # the derivative of each term by its distance, along its vector
            slope = -4.0 * weight / d * (target - d) ** 2 - 2.0 * weight * (target - d)
            pull = (slope / d)[:, None] * vectors
            f = np.zeros_like(positions)
            np.add.at(f, i, pull)
            np.add.at(f, j, -pull)
            forces[k - 1] = f if movable is None else f * movable
        energies[0] = energies[count] = 0.0
        full = np.concatenate([first[None], images, last[None]])
        return _band_forces(full, energies, forces, spring, None).reshape(-1, 3)

    moving = band[1:-1].reshape(-1, 3).copy()
    fire = _Fire(moving, evaluate(moving), np.ones((len(moving), 1)), 0.1, 0.1)
    while _largest(fire.forces) > _IDPP_MAX_FORCE and fire.step < _IDPP_MAX_STEPS:
        fire.advance(evaluate)
    return np.concatenate(
        [first[None], fire.variables.reshape(count - 1, *first.shape), last[None]]
    )


def _check_states(initial: System, final: System) -> None:
    if len(initial) != len(final):
        raise InvalidParameter(
            f"the states have {len(initial)} and {len(final)} atoms; the band needs "
            "the same atoms in the same order"
        )
    if not (
        np.allclose(initial.box.matrix, final.box.matrix, rtol=0, atol=1e-8)
        and np.array_equal(initial.box.pbc, final.box.pbc)
    ):
        raise InvalidParameter("the states need the same box")
    for column in ("element", "type"):
        if column in initial.table.columns and column in final.table.columns:
            if initial.table[column].to_list() != final.table[column].to_list():
                raise InvalidParameter(
                    f"the {column} of the atoms differ between the states: the band "
                    "needs the same atoms in the same order"
                )


def neb(
    initial: System,
    final: System,
    calculator: Calculator | Sequence[Calculator],
    *,
    images: int | Sequence[System] = 5,
    climb: bool = True,
    spring: float = 1.0,
    interpolation: str = "linear",
    max_force: float = 0.01,
    max_steps: int = 1000,
    max_step: float = 0.1,
    timestep: float = 0.001,
    callback: Any = None,
    log: Any = None,
    log_every: int = 10,
) -> NEBResult:
    """Finds the minimum energy path between two states with the nudged
    elastic band, its highest image climbing to the saddle point.

    Parameters
    ----------
    initial, final : System
        The two states, each a minimum of the energy (relaxed with
        ``minimize``), with the same atoms in the same order and the same
        box; a ``mass`` or an ``element`` column; a ``movable`` column of the
        initial state, (N, 3) bool, keeps each atom still along the
        directions where it is False.
    calculator : Calculator or sequence of Calculator
        Computes energies and forces; or one for each image between the
        states, which then compute at the same time, each in a thread of its
        own: for calculators that each run on their own, such as VASP in
        directories of their own.
    images : int32 or sequence of System, optional
        Number of images between the states, interpolated by
        ``interpolation``, or the images themselves to start from. Default 5.
    climb : bool, optional
        Whether the image of highest energy climbs to the saddle point.
        Default True.
    spring : float64, optional
        Constant of the springs between the images along the band, in
        eV/Å². Default 1.0.
    interpolation : {"linear", "idpp"}, optional
        How the images start: on the straight line between the states, each
        atom by its shortest periodic displacement, or relaxed from it on the
        image dependent pair potential, which keeps the distances between
        atoms near those of the states. Default "linear".
    max_force : float64, optional
        Convergence: the largest force of the band on an atom of an image, in
        eV/Å. Default 0.01.
    max_steps : int32, optional
        Largest number of steps. Default 1000.
    max_step : float64, optional
        Largest displacement of an atom along x, y or z in one step, in Å.
        Default 0.1.
    timestep : float64, optional
        First time step of the FIRE 2.0 dynamics of the band, in ps. Default
        0.001.
    callback : callable, optional
        Called after every step as ``callback(step, images, energies,
        max_force)``, with the band as a tuple of System; returning True
        stops it.
    log : str, os.PathLike or stream, optional
        Writes the band's relaxation to this file, which it overwrites, or to
        this stream: a heading, the largest force, the barrier and the image
        of highest energy every ``log_every`` steps and at the last, and how
        it ended. Lines that are not rows of numbers start with #. Default
        None, writing none.
    log_every : int32, optional
        Writes a row to the log every this many steps. Default 10.

    Returns
    -------
    NEBResult
        The band, its energies and the barrier.

    Raises
    ------
    InvalidParameter
        For invalid options, states that differ in their atoms or box,
        images that do not fit them, or calculators that do not compute
        energies and forces.
    MissingProperty
        If the system has neither masses nor elements.
    CalculationFailed
        If an energy or a force is not finite.
    """
    _check_states(initial, final)
    if isinstance(images, int) and not isinstance(images, bool):
        if images < 1:
            raise InvalidParameter(f"images must be at least 1, not {images}")
        count, given = images, None
    elif isinstance(images, Sequence) and len(images) > 0:
        given = list(images)
        for image in given:
            if not isinstance(image, System):
                raise InvalidParameter(
                    "images must be an integer or a sequence of System"
                )
            _check_states(initial, image)
        count = len(given)
    else:
        raise InvalidParameter("images must be an integer or a sequence of System")
    if interpolation not in INTERPOLATIONS:
        raise InvalidParameter(
            f"interpolation must be one of {INTERPOLATIONS}, not {interpolation!r}"
        )
    if not spring > 0 or not max_force > 0:
        raise InvalidParameter("spring and max_force must be positive")
    if max_steps < 0 or not max_step > 0 or not timestep > 0:
        raise InvalidParameter(
            "max_steps must be at least 0, and max_step and timestep positive"
        )
    if isinstance(log_every, bool) or not isinstance(log_every, int) or log_every < 1:
        raise InvalidParameter(
            f"log_every must be a positive integer, not {log_every!r}"
        )
    if isinstance(calculator, Calculator):
        calculators = [calculator] * count
    else:
        calculators = list(calculator)
        if len(calculators) != count or not all(
            isinstance(c, Calculator) for c in calculators
        ):
            raise InvalidParameter(
                f"give one calculator, or one for each of the {count} images"
            )
    for c in calculators:
        if not {"energy", "forces"} <= c.implemented:
            raise InvalidParameter(
                f"{type(c).__name__} does not compute energies and forces"
            )

    masses = _masses(initial)
    movable = (
        initial.table["movable"].to_numpy().astype(bool)
        if "movable" in initial.table.columns
        else None
    )
    # the band, each atom of the final state where its shortest periodic
    # displacement from the initial state takes it, so that the images
    # change continuously along it
    box = initial.box
    start = initial["positions"]
    inverse = np.linalg.inv(box.matrix)

    def nearest(positions: np.ndarray) -> np.ndarray:
        # each atom moved by whole box vectors to within half a box of its
        # place in the initial state
        whole = np.where(box.pbc, np.round((positions - start) @ inverse), 0.0)
        return positions - whole @ box.matrix

    end = nearest(final["positions"])
    if given is None:
        fractions = np.arange(1, count + 1)[:, None, None] / (count + 1)
        middle = start[None] + fractions * (end - start)[None]
    else:
        middle = np.stack([nearest(image["positions"]) for image in given])
    band = np.concatenate([start[None], middle, end[None]])
    if interpolation == "idpp":
        band = _idpp(band, initial, spring, movable)

    lists = [_Neighbors(c.cutoff) for c in calculators]
    ends = [_Neighbors(calculators[0].cutoff), _Neighbors(calculators[-1].cutoff)]
    # the images that share a calculator compute one after another, the
    # groups at the same time
    groups: dict[int, list[int]] = {}
    for k, c in enumerate(calculators):
        groups.setdefault(id(c), []).append(k)
    pool = ThreadPoolExecutor(len(groups)) if len(groups) > 1 else None

    def computed(c: Calculator, system: System, neighbors: _Neighbors):
        result = c.compute(system, ("energy", "forces"), neighbors=neighbors.of(system))
        if not (np.isfinite(result.energy) and np.isfinite(result.forces).all()):
            raise CalculationFailed(
                "the energy or the forces are not finite: atoms may overlap"
            )
        return result

    first = computed(calculators[0], initial.with_columns(positions=start), ends[0])
    last = computed(calculators[-1], initial.with_columns(positions=end), ends[1])
    systems: list[System] = [initial] * count
    results: list[Any] = [None] * count

    def compute(moving: np.ndarray) -> None:
        for k in range(count):
            systems[k] = initial.with_columns(positions=moving[k])

        def run(indices: list[int]) -> None:
            for k in indices:
                results[k] = computed(calculators[k], systems[k], lists[k])

        if pool is None:
            run(list(range(count)))
        else:
            for done in [pool.submit(run, indices) for indices in groups.values()]:
                done.result()

    energies = np.empty(count + 2)
    energies[0], energies[-1] = first.energy, last.energy
    climber: int | None = None
    largest = math.inf

    def evaluate(variables: np.ndarray) -> np.ndarray:
        nonlocal climber, largest
        moving = variables.reshape(count, *start.shape)
        compute(moving)
        forces = np.stack([r.forces for r in results])
        if movable is not None:
            forces = forces * movable
        energies[1:-1] = [r.energy for r in results]
        climber = 1 + int(np.argmax(energies[1:-1])) if climb else None
        full = np.concatenate([start[None], moving, end[None]])
        out = _band_forces(full, energies, forces, spring, climber)
        if movable is not None:
            out = out * movable
        largest = _largest(out)
        return out.reshape(-1, 3)

    inverse_mass = np.tile((_FTM2V / masses)[:, None], (count, 1))
    writer = _Log(log)

    def band_now() -> tuple[System, ...]:
        states = (first.attach(initial.with_columns(positions=start)),)
        middle = tuple(r.attach(s) for r, s in zip(results, systems))
        return states + middle + (last.attach(initial.with_columns(positions=end)),)

    def write() -> None:
        top = int(np.argmax(energies))
        writer.row(fire.step, largest, float(energies[top] - energies[0]), top)

    variables = band[1:-1].reshape(-1, 3).copy()
    fire = _Fire(variables, evaluate(variables), inverse_mass, timestep, max_step)
    try:
        if writer.on:
            writer.title("nudged elastic band (FIRE 2.0)")
            writer.comment(
                f"{len(initial)} atoms; {count} images between the states, "
                f"{'with' if climb else 'without'} a climbing image; "
                f"interpolation {interpolation}",
                f"potential: {calculators[0]!r}",
                f"spring {spring!r} eV/Å², max_force {max_force!r} eV/Å, "
                f"max_steps {max_steps}",
                "",
            )
            writer.columns(_LOG_COLUMNS)
            write()
        while largest >= max_force and fire.step < max_steps:
            fire.advance(evaluate)
            if fire.step % log_every == 0:
                write()
            if callback is not None and callback(
                fire.step, band_now(), energies.copy(), largest
            ):
                break
        if writer.on:
            if fire.step % log_every != 0:
                write()
            how = (
                "converged"
                if largest < max_force
                else (
                    "stopped by the callback"
                    if fire.step < max_steps
                    else "reached max_steps"
                )
            )
            writer.comment(f"{how} after {fire.step} steps, {writer.wall:.2f} s")
    finally:
        writer.close()
        if pool is not None:
            pool.shutdown()

    full = band_now()
    positions = np.stack([image["positions"] for image in full])
    steps = positions[1:] - positions[:-1]
    distances = np.concatenate(
        [[0.0], np.cumsum(np.sqrt((steps * steps).sum(axis=(1, 2))))]
    )
    top = int(np.argmax(energies))
    return NEBResult(
        images=full,
        energies=energies.copy(),
        distances=distances,
        barrier=float(energies[top] - energies[0]),
        highest=top,
        converged=largest < max_force,
        steps=fire.step,
        max_force=largest,
    )
