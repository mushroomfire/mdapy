# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The melting point by the coexistence of a solid and a liquid in one box."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from mdapy.analysis._steinhardt import SteinhardtParameters
from mdapy.calculators._lammps import LAMMPS
from mdapy.core.box import Box
from mdapy.core.result import Result
from mdapy.core.system import System
from mdapy.core.threads import CancelToken
from mdapy.errors import InvalidParameter, MissingProperty
from mdapy.neighbor import NeighborList
from mdapy.simulation._dynamics import MolecularDynamics
from mdapy.simulation._free_energy import _mean_box, _Progress, _run
from mdapy.transform import replicate

# outcomes of a run
_MELTED, _FROZEN, _COEXISTING = "melted", "frozen", "coexisting"
# below and above these fractions of solid atoms, one phase has taken over
_ONE_PHASE = 0.1
# a run with less than this much of either phase is run again from its temperature
_ENOUGH = 0.2
# how far a first guess moves when it is all that is known
_STEP = 0.15
# how much hotter than the temperature the liquid half is melted
_HOT = 2.0
# times the melting of the liquid half is tried, each this much hotter
_MELT_TRIES = 5
_HOTTER = 1.5
# the first shell of a crystal ends where the next distance is this much larger
_SHELL_GAP = 1.2
_MOST_NEIGHBORS = 20
# an atom is solid-like with this share of its neighbors in solid-like bonds
_SOLID_BONDS = 0.6
# stretches of coexistence.time in one box, while the temperature still drifts
_SEGMENTS = 5
# parts of a run, after each of which one phase having taken over ends it
_PIECES = 10
# blocks for the standard error of the mean temperature
_BLOCKS = 5


@dataclass(frozen=True)
class MeltingAttempt:
    """One stretch of a solid and a liquid in one box.

    Attributes
    ----------
    start : float
        The temperature in K at which the run started.
    outcome : str
        "melted", "frozen" or "coexisting".
    temperature : float
        The mean temperature in K over its second half.
    fraction : float
        The fraction of solid atoms at its end.
    drift : float
        How much in K the mean temperature changed from the third quarter of
        the run to the last.
    """

    start: float
    outcome: str
    temperature: float
    fraction: float
    drift: float


@dataclass(frozen=True)
class MeltingPointResult(Result):
    """The melting point of a crystal.

    Attributes
    ----------
    temperature : float
        The melting point in K: the temperature at which the two phases
        coexist, averaged over the second half of the last stretch.
    error : float
        The standard error of that mean in K, from blocks of the run.
    pressure : float
        The pressure in GPa.
    attempts : tuple of MeltingAttempt
        Each stretch of each run, in order.
    time : numpy.ndarray
        (S,) the time in ps along the last stretch.
    temperatures : numpy.ndarray
        (S,) the temperature in K along the last stretch.
    pressures : numpy.ndarray
        (S,) the pressure in GPa along the last stretch.
    fractions : numpy.ndarray
        (F,) the fraction of solid atoms along the last stretch.
    fraction_time : numpy.ndarray
        (F,) the time in ps of the fractions.
    system : System
        The atoms at the end of the last run, with a boolean column ``solid``.
    """

    temperature: float
    error: float
    pressure: float
    attempts: tuple[MeltingAttempt, ...]
    time: np.ndarray
    temperatures: np.ndarray
    pressures: np.ndarray
    fractions: np.ndarray
    fraction_time: np.ndarray
    system: System


@dataclass(frozen=True)
class _Run:
    """What one run gives, with an attempt for each of its segments; its last
    segment for the rest."""

    attempts: tuple[MeltingAttempt, ...]
    error: float
    time: np.ndarray
    temperatures: np.ndarray
    pressures: np.ndarray
    fractions: np.ndarray
    fraction_time: np.ndarray
    system: System


def _shell(crystal: System) -> int:
    """How many neighbors of the first atoms of a perfect crystal are in the
    shell that the next distance leaves far behind: 12 in fcc and hcp, 14 in
    bcc, 4 in diamond."""
    distances = NeighborList.build(
        crystal, num_neighbors=_MOST_NEIGHBORS
    ).distances.reshape(len(crystal), _MOST_NEIGHBORS)
    distances = np.sort(distances, axis=1).max(axis=0)
    jumps = np.flatnonzero(distances[1:] / distances[:-1] > _SHELL_GAP)
    return int(jumps[0]) + 1 if len(jumps) else _MOST_NEIGHBORS


def _solid_like(system: System, neighbors: int) -> np.ndarray:
    """(N,) whether each atom shares the order of its neighbors, as in a crystal."""
    calculator = SteinhardtParameters(
        degrees=(6,),
        num_neighbors=neighbors,
        solid=True,
        min_solid_bonds=round(_SOLID_BONDS * neighbors),
    )
    return calculator.compute(system).solid


def _travel(start: System, end: System, first: int) -> float:
    """The mean square distance in Å² the atoms from this index on have gone,
    in a box that stayed the same."""
    shifts = (
        end["positions"]
        - start["positions"]
        + (end["image"] - start["image"]) @ start.box.matrix
    )[first:]
    shifts = shifts - shifts.mean(axis=0)
    return float((shifts**2).sum(axis=1).mean())


def _block_error(values: np.ndarray) -> float:
    """The standard error of the mean of correlated values, from blocks."""
    blocks = [block.mean() for block in np.array_split(values, _BLOCKS)]
    return float(np.std(blocks, ddof=1) / np.sqrt(_BLOCKS))


def _supercell(crystal: System, atoms: int) -> System:
    """The crystal repeated to about this many atoms, in a box three times as
    long along its third vector as along the others."""
    lengths = crystal.box.thickness
    volume_per_atom = crystal.box.volume / len(crystal)
    edge = (atoms * volume_per_atom / 3) ** (1 / 3)
    repeats = np.maximum(
        np.round(np.array([edge, edge, 3 * edge]) / lengths).astype(int), 1
    )
    repeats[2] += repeats[2] % 2  # the two halves hold whole cells
    return replicate(crystal, repeats)


def _one_run(
    start: System,
    potential: LAMMPS,
    temperature: float,
    pressure: float,
    barostat_damping: float | None,
    timestep: float,
    equilibrium: int,
    coexistence: int,
    neighbors: int,
    tolerance: float,
    seed: int,
    counter: _Progress,
) -> _Run:
    """Melts one half of the crystal, joins the halves at this temperature,
    and lets them find their common temperature at constant enthalpy."""
    every = max(equilibrium // 25, 1)
    with MolecularDynamics(start, potential, timestep=timestep) as md:
        md.velocities(2 * temperature, seed=seed)
        done = md.step
        md.run(
            equilibrium,
            ensemble="npt",
            temperature=temperature,
            pressure=pressure,
            barostat="aniso",
            barostat_damping=barostat_damping,
            thermo=equilibrium,
            frames=every,
            cancel=counter.cancel,
        )
        counter.add(md.step - done)
        box: Box = _mean_box(md.frames)
    crystal = start.with_box(box, scale_positions=True)
    third = box.to_fractional(crystal["positions"])[:, 2]
    third -= np.floor(third)
    crystal = crystal.gather(np.argsort(third, kind="stable"))
    half = len(crystal) // 2
    hot = _HOT * temperature

    def heating(first: float, last: float) -> str:
        # at the density of the crystal, so that the half cannot fly apart; the
        # barostat of the run that follows lets it expand
        return (
            f"fix mdapy_heat mdapy_liquid nvt temp {first!r} {last!r} "
            f"{100 * timestep!r}"
        )

    with MolecularDynamics(crystal, potential, timestep=timestep) as md:
        md.command(
            "\n".join(
                [
                    f"group mdapy_solid id 1:{half}",
                    "group mdapy_liquid subtract all mdapy_solid",
                    "velocity mdapy_solid set 0 0 0",
                    f"velocity mdapy_liquid create {hot!r} {seed + 1} mom yes rot no",
                    "fix mdapy_hold mdapy_solid setforce 0 0 0",
                    heating(hot, hot),
                ]
            )
        )
        # hot until the atoms have left their places, by about the distance
        # between them, hotter and longer as long as they have not
        before = md.system
        distance = (crystal.box.volume / len(crystal)) ** (2 / 3)
        for _ in range(_MELT_TRIES):
            _run(md, equilibrium, counter)
            if _travel(before, md.system, half) > distance:
                break
            hot *= _HOTTER
            md.command(f"unfix mdapy_heat\n{heating(hot, hot)}")
        else:
            raise InvalidParameter(
                f"one half of the crystal does not melt, even at {hot:g} K"
            )
        md.command(f"unfix mdapy_heat\n{heating(hot, temperature)}")
        _run(md, equilibrium, counter)
        # the liquid settles at the temperature: cooled fast, it holds extra energy
        md.command(f"unfix mdapy_heat\n{heating(temperature, temperature)}")
        _run(md, equilibrium, counter)
        # the crystal starts at rest in its places: half of this goes into them
        md.command(
            "\n".join(
                [
                    "unfix mdapy_heat",
                    "unfix mdapy_hold",
                    f"velocity mdapy_solid create {2 * temperature!r} {seed + 2} "
                    "mom yes rot no",
                ]
            )
        )
        joined = equilibrium // 2
        done = md.step
        md.run(
            joined,
            ensemble="npt",
            temperature=temperature,
            pressure=pressure,
            barostat="aniso",
            barostat_damping=barostat_damping,
            thermo=joined,
            cancel=counter.cancel,
        )
        counter.add(md.step - done)
        solid_like = _solid_like(md.system, neighbors)
        ordered = float(solid_like[:half].mean())
        if ordered < _ENOUGH:  # the crystal melted by itself: far too hot
            return _Run(
                attempts=(
                    MeltingAttempt(temperature, _MELTED, temperature, ordered / 2, 0.0),
                ),
                error=0.0,
                time=np.zeros(0),
                temperatures=np.zeros(0),
                pressures=np.zeros(0),
                fractions=np.zeros(0),
                fraction_time=np.zeros(0),
                system=md.system,
            )
        begin_step = md.step
        attempts = []
        for segment in range(_SEGMENTS):
            begin = len(md.thermo)
            times, fractions, outcome = [], [], _COEXISTING
            piece = max(coexistence // _PIECES, 1)
            for _ in range(_PIECES):
                counter.check()
                done = md.step
                md.run(
                    piece,
                    ensemble="nph",
                    pressure=pressure,
                    barostat="aniso",
                    barostat_damping=barostat_damping,
                    thermo=max(piece // 50, 1),
                    cancel=counter.cancel,
                )
                counter.add(md.step - done)
                solid_like = _solid_like(md.system, neighbors)
                fractions.append(float(solid_like.mean()))
                times.append((md.step - begin_step) * timestep)
                if fractions[-1] < _ONE_PHASE:
                    outcome = _MELTED
                    break
                if fractions[-1] > 1 - _ONE_PHASE:
                    outcome = _FROZEN
                    break
            rows = md.thermo[begin:]
            temperatures = rows["temperature"].to_numpy()
            second = temperatures[len(temperatures) // 2 :]
            third, fourth = np.array_split(second, 2)
            error = _block_error(second)
            drift = float(fourth.mean() - third.mean())
            attempts.append(
                MeltingAttempt(
                    start=temperature,
                    outcome=outcome,
                    temperature=float(second.mean()),
                    fraction=fractions[-1],
                    drift=drift,
                )
            )
            enough = _ENOUGH <= fractions[-1] <= 1 - _ENOUGH
            if outcome != _COEXISTING or not enough:
                break
            if abs(drift) <= max(tolerance, 2 * error):
                break
        final = md.system.with_columns(solid=solid_like)
    return _Run(
        attempts=tuple(attempts),
        error=error,
        time=(rows["step"].to_numpy() - begin_step) * timestep,
        temperatures=temperatures,
        pressures=rows["pressure"].to_numpy(),
        fractions=np.array(fractions),
        fraction_time=np.array(times),
        system=final,
    )


def _is_answer(last: MeltingAttempt, error: float, tolerance: float) -> bool:
    """Whether a run ended with enough of both phases and a steady temperature."""
    return (
        last.outcome == _COEXISTING
        and _ENOUGH <= last.fraction <= 1 - _ENOUGH
        and abs(last.drift) <= max(tolerance, 2 * error)
    )


def _next_temperature(attempts: list[MeltingAttempt]) -> float:
    """The temperature to try next after a run that gave no answer. A run that
    melted is too hot and one that froze too cold, whatever its temperature
    became; between such bounds the next try is halfway. A run with both phases
    goes on from where its temperature was going."""
    last = attempts[-1]
    if last.outcome == _COEXISTING:
        return last.temperature + last.drift
    hot = [a.start for a in attempts if a.outcome == _MELTED]
    cold = [a.start for a in attempts if a.outcome == _FROZEN]
    if hot and cold:
        return 0.5 * (min(hot) + max(cold))
    if hot:
        return min(hot) * (1 - _STEP)
    return max(cold) * (1 + _STEP)


def melting_point(
    crystal: System,
    potential: LAMMPS,
    temperature: float,
    *,
    pressure: float = 0.0,
    atoms: int = 4000,
    repeat: ArrayLike | None = None,
    timestep: float = 0.001,
    equilibration: float = 10.0,
    coexistence: float = 100.0,
    tolerance: float = 3.0,
    max_runs: int = 8,
    barostat_damping: float | None = None,
    seed: int = 1,
    progress: Callable[[int, int], None] | None = None,
    cancel: CancelToken | None = None,
) -> MeltingPointResult:
    """The melting point of a crystal, from a first guess, by the two-phase
    method: a solid and a liquid share a box at constant enthalpy, where they
    reach their common temperature while the interface between them moves. A
    run in which one phase takes over gives a new guess, until the two last.

    Parameters
    ----------
    crystal : System
        The crystal, relaxed, with an ``element`` column, in a box periodic in
        every direction; the interface between the phases is across its third
        vector.
    potential : LAMMPS
        The interaction.
    temperature : float
        The first guess in K, positive.
    pressure : float, optional
        Pressure in GPa. Default 0.
    atoms : int, optional
        About how many atoms the box holds, three times as long along its
        third vector as along the others. Default 4000.
    repeat : array_like, optional
        (3,) repeats of crystal along its vectors, instead of ``atoms``; the
        third is made even.
    timestep : float, optional
        Time step in ps. Default 0.001.
    equilibration : float, optional
        Time in ps of each stage before the two phases meet. Default 10.
    coexistence : float, optional
        Time in ps of each stretch of the two phases, of which a run has up to
        five while the temperature drifts. Default 100.
    tolerance : float, optional
        How much in K the temperature may still drift, from the third quarter
        of a stretch of coexistence to the last, for it to be the answer; twice
        its statistical error, if that is larger. Default 3.
    max_runs : int, optional
        Largest number of runs. Default 8.
    barostat_damping : float, optional
        Damping time of the barostat in ps, as in
        :meth:`MolecularDynamics.run`. Default 1000 time steps.
    seed : int, optional
        Seed of the random numbers, positive. Default 1.
    progress : callable, optional
        Called as ``progress(done, total)`` with the steps run, of a total
        that takes every run to be the last.
    cancel : CancelToken, optional
        Stops the calculation, which raises :class:`mdapy.errors.Cancelled`.

    Returns
    -------
    MeltingPointResult
        The melting point, its error, the runs, and the temperature, pressure
        and fraction of solid atoms along the last run.

    Raises
    ------
    InvalidParameter
        For invalid options, a box not periodic in every direction, if half of
        the crystal does not melt, if no run ends with two phases within
        ``max_runs``, or if LAMMPS fails.
    MissingProperty
        If the system has no ``element`` column.
    Cancelled
        If cancelled.
    """
    if not isinstance(potential, LAMMPS):
        raise InvalidParameter("potential must be a mdapy.calculators.LAMMPS")
    if "element" not in crystal.table.columns:
        raise MissingProperty("the melting point needs the element column")
    if not crystal.box.pbc.all():
        raise InvalidParameter(
            "the melting point needs a box periodic in every direction"
        )
    if not temperature > 0:
        raise InvalidParameter(f"temperature must be positive, not {temperature!r}")
    if not min(timestep, equilibration, coexistence, tolerance) > 0:
        raise InvalidParameter(
            "timestep, equilibration, coexistence and tolerance must be positive"
        )
    if int(atoms) < 1 or int(max_runs) < 1 or int(seed) < 1:
        raise InvalidParameter("atoms, max_runs and seed must be positive")
    if repeat is None:
        start = _supercell(crystal, int(atoms))
    else:
        counts = np.array(repeat)
        if (
            counts.shape != (3,)
            or not np.issubdtype(counts.dtype, np.integer)
            or (counts < 1).any()
        ):
            raise InvalidParameter(
                f"repeat must be 3 positive integers, not {repeat!r}"
            )
        counts[2] += counts[2] % 2
        start = replicate(crystal, counts)
    neighbors = _shell(start)
    equilibrium = max(int(round(equilibration / timestep)), 2)
    together = max(int(round(coexistence / timestep)), _PIECES)
    per_run = 4 * equilibrium + equilibrium // 2 + _SEGMENTS * together
    counter = _Progress(per_run * int(max_runs), progress, cancel)
    attempts: list[MeltingAttempt] = []
    guess = float(temperature)
    run = None
    for k in range(int(max_runs)):
        run = _one_run(
            start,
            potential,
            guess,
            float(pressure),
            barostat_damping,
            timestep,
            equilibrium,
            together,
            neighbors,
            float(tolerance),
            int(seed) + 10 * k,
            counter,
        )
        attempts.extend(run.attempts)
        if _is_answer(attempts[-1], run.error, float(tolerance)):
            return MeltingPointResult(
                temperature=attempts[-1].temperature,
                error=run.error,
                pressure=float(pressure),
                attempts=tuple(attempts),
                time=run.time,
                temperatures=run.temperatures,
                pressures=run.pressures,
                fractions=run.fractions,
                fraction_time=run.fraction_time,
                system=run.system,
            )
        guess = _next_temperature(attempts)
    outcomes = ", ".join(f"{a.start:.0f} K {a.outcome}" for a in attempts)
    raise InvalidParameter(f"no melting point in {int(max_runs)} runs: {outcomes}")
