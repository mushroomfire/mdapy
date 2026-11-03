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
from mdapy.simulation._dynamics import MolecularDynamics
from mdapy.simulation._free_energy import _Progress, _mean_box, _run
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
# parts of a run, after each of which one phase having taken over ends it
_PIECES = 10
# blocks for the standard error of the mean temperature
_BLOCKS = 5


@dataclass(frozen=True)
class MeltingAttempt:
    """One run of a solid and a liquid in one box.

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
    """

    start: float
    outcome: str
    temperature: float
    fraction: float


@dataclass(frozen=True)
class MeltingPointResult(Result):
    """The melting point of a crystal.

    Attributes
    ----------
    temperature : float
        The melting point in K: the temperature at which the two phases
        coexist, averaged over the second half of the last run.
    error : float
        The standard error of that mean in K, from blocks of the run.
    pressure : float
        The pressure in GPa.
    attempts : tuple of MeltingAttempt
        The runs, in order.
    time : numpy.ndarray
        (S,) the time in ps along the last run.
    temperatures : numpy.ndarray
        (S,) the temperature in K along the last run.
    pressures : numpy.ndarray
        (S,) the pressure in GPa along the last run.
    fractions : numpy.ndarray
        (F,) the fraction of solid atoms along the last run.
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
    """What one run gives, with the attempt it counts as."""

    attempt: MeltingAttempt
    error: float
    time: np.ndarray
    temperatures: np.ndarray
    pressures: np.ndarray
    fractions: np.ndarray
    fraction_time: np.ndarray
    system: System


def _order(system: System) -> np.ndarray:
    """(N,) the averaged q6 of each atom, high in a crystal and low in a liquid."""
    calculator = SteinhardtParameters(degrees=(6,), num_neighbors=12, average=True)
    return calculator.compute(system)[6]


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

    with MolecularDynamics(crystal, potential, timestep=timestep) as md:
        md.command(
            "\n".join(
                [
                    f"group mdapy_solid id 1:{half}",
                    "group mdapy_liquid subtract all mdapy_solid",
                    "velocity mdapy_solid set 0 0 0",
                    f"velocity mdapy_liquid create {hot!r} {seed + 1} mom yes rot no",
                    "fix mdapy_hold mdapy_solid setforce 0 0 0",
                    "fix mdapy_heat mdapy_liquid nvt temp "
                    f"{hot!r} {hot!r} {100 * timestep!r}",
                ]
            )
        )
        _run(md, equilibrium, counter)
        md.command(
            "\n".join(
                [
                    "unfix mdapy_heat",
                    "fix mdapy_heat mdapy_liquid nvt temp "
                    f"{hot!r} {temperature!r} {100 * timestep!r}",
                ]
            )
        )
        _run(md, equilibrium, counter)
        md.command(
            "\n".join(
                [
                    "unfix mdapy_heat",
                    "unfix mdapy_hold",
                    f"velocity mdapy_solid create {temperature!r} {seed + 2} "
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
        order = _order(md.system)
        solid, liquid = order[:half].mean(), order[half:].mean()
        if not solid > liquid:
            raise InvalidParameter(
                f"the two halves are alike at {temperature:g} K: the order of the "
                f"atoms is {solid:.3f} and {liquid:.3f}"
            )
        threshold = 0.5 * (solid + liquid)
        begin, begin_step = len(md.thermo), md.step
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
            now = md.system
            order = _order(now)
            fractions.append(float((order > threshold).mean()))
            times.append((md.step - begin_step) * timestep)
            if fractions[-1] < _ONE_PHASE:
                outcome = _MELTED
                break
            if fractions[-1] > 1 - _ONE_PHASE:
                outcome = _FROZEN
                break
        rows = md.thermo[begin:]
        final = md.system.with_columns(solid=order > threshold)
    temperatures = rows["temperature"].to_numpy()
    second = temperatures[len(temperatures) // 2 :]
    return _Run(
        attempt=MeltingAttempt(
            start=temperature,
            outcome=outcome,
            temperature=float(second.mean()),
            fraction=fractions[-1],
        ),
        error=_block_error(second),
        time=(rows["step"].to_numpy() - begin_step) * timestep,
        temperatures=temperatures,
        pressures=rows["pressure"].to_numpy(),
        fractions=np.array(fractions),
        fraction_time=np.array(times),
        system=final,
    )


def _next_temperature(
    attempts: list[MeltingAttempt], tolerance: float
) -> float | None:
    """The temperature to try next, or None when the last run is the answer: it
    kept enough of both phases and stayed near where it started. A run that
    melted is too hot and one that froze too cold, whatever its temperature
    became; between such bounds the next try is halfway."""
    last = attempts[-1]
    if last.outcome == _COEXISTING:
        enough = _ENOUGH <= last.fraction <= 1 - _ENOUGH
        if enough and abs(last.temperature - last.start) <= tolerance:
            return None
        return last.temperature
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
    tolerance: float = 10.0,
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
        Time in ps of each run of the two phases. Default 100.
    tolerance : float, optional
        How far in K the mean temperature of a run may be from its start, for
        that run to be the answer. Default 10.
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
        For invalid options, a box not periodic in every direction, if the
        melted half stays as ordered as the crystal, if no run ends with two
        phases within ``max_runs``, or if LAMMPS fails.
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
            raise InvalidParameter(f"repeat must be 3 positive integers, not {repeat!r}")
        counts[2] += counts[2] % 2
        start = replicate(crystal, counts)
    equilibrium = max(int(round(equilibration / timestep)), 2)
    together = max(int(round(coexistence / timestep)), _PIECES)
    per_run = 3 * equilibrium + equilibrium // 2 + together
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
            int(seed) + 10 * k,
            counter,
        )
        attempts.append(run.attempt)
        following = _next_temperature(attempts, float(tolerance))
        if following is None:
            return MeltingPointResult(
                temperature=run.attempt.temperature,
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
        guess = following
    outcomes = ", ".join(f"{a.start:.0f} K {a.outcome}" for a in attempts)
    raise InvalidParameter(f"no melting point in {int(max_runs)} runs: {outcomes}")
