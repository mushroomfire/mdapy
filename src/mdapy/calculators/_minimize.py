# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Energy minimization of the positions with FIRE 2.0, as LAMMPS min_style fire."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from mdapy import units
from mdapy.calculators._base import Calculator, CalculatorResult
from mdapy.core.system import System
from mdapy.elements import atomic_masses
from mdapy.errors import InvalidParameter, MissingProperty
from mdapy.neighbor import NeighborList

# Neighbors are searched this much farther than the cutoff of the calculator,
# in Å, so that one search serves the steps until an atom has moved half as far.
_SKIN = 1.0

# The default settings of LAMMPS min_style fire (min_modify), FIRE 2.0 of
# Guénolé et al. (2020): steps after a reset before the time step grows, its
# growth and shrink factors, the mixing parameter and its decay, and the
# largest and smallest time steps as multiples of the first.
_DELAY = 20
_GROW, _SHRINK = 1.1, 0.5
_ALPHA0, _ALPHA_SHRINK = 0.25, 0.99
_TMAX, _TMIN = 10.0, 0.02

# Velocity in Å/ps of a force of 1 eV/Å acting on 1 amu for 1 ps.
_FTM2V = 1.0 / units.AMU_A2_PER_PS2_TO_EV


@dataclass(frozen=True)
class MinimizeResult:
    """The outcome of :func:`minimize`.

    Attributes
    ----------
    system : System
        The last configuration, with ``forces`` and ``atomic_energies``
        columns and ``energy`` in ``info``.
    converged : bool
        Whether every force fell below ``max_force``.
    steps : int
        Number of steps taken.
    energy : float
        Energy of the last configuration in eV.
    max_force : float
        Largest force of the last configuration in eV/Å.
    frames : tuple of System or None
        With ``save_every``, the configurations at step 0, every
        ``save_every`` steps and the last step, with ``forces`` and
        ``atomic_energies`` columns and ``energy`` and the step number as
        ``timestep`` in ``info``; None otherwise.
    """

    system: System
    converged: bool
    steps: int
    energy: float
    max_force: float
    frames: tuple[System, ...] | None = None


def _masses(system: System) -> np.ndarray:
    table = system.table
    if "mass" in table.columns:
        return table["mass"].to_numpy().astype(float)
    if "element" in table.columns:
        return atomic_masses(table["element"].to_physical().to_numpy())
    raise MissingProperty("minimize needs the masses: a mass or an element column")


def minimize(
    system: System,
    calculator: Calculator,
    *,
    max_force: float = 1e-3,
    max_steps: int = 1000,
    max_step: float = 0.1,
    timestep: float = 0.001,
    save_every: int | None = None,
    callback: Any = None,
) -> MinimizeResult:
    """Minimizes the energy over the positions, in a fixed box, with FIRE 2.0.

    Parameters
    ----------
    system : System
        The starting configuration, with a ``mass`` or an ``element`` column.
    calculator : Calculator
        Computes energies and forces.
    max_force : float, optional
        Convergence: the largest force on any atom, in eV/Å. Default 1e-3.
    max_steps : int, optional
        Largest number of steps. Default 1000.
    max_step : float, optional
        Largest displacement of an atom along x, y or z in one step, in Å.
        Default 0.1.
    timestep : float, optional
        First time step of the dynamics, in ps. Default 0.001.
    save_every : int, optional
        Keeps the configuration every this many steps, with the first and
        the last, in ``frames`` of the result. Default None, keeping none.
    callback : callable, optional
        Called after every step as ``callback(step, system, energy,
        max_force)``; returning True stops the minimization.

    Returns
    -------
    MinimizeResult
        The last configuration and whether it converged.

    Raises
    ------
    InvalidParameter
        For invalid options, or a calculator without energies and forces.
    MissingProperty
        If the system has neither masses nor elements.
    """
    if not max_force > 0:
        raise InvalidParameter("max_force must be positive")
    if max_steps < 0 or not max_step > 0 or not timestep > 0:
        raise InvalidParameter(
            "max_steps must be at least 0, and max_step and timestep positive"
        )
    if save_every is not None and (
        isinstance(save_every, bool)
        or not isinstance(save_every, int)
        or save_every < 1
    ):
        raise InvalidParameter(
            f"save_every must be a positive integer, not {save_every!r}"
        )
    if not {"energy", "forces"} <= calculator.implemented:
        raise InvalidParameter(
            f"{type(calculator).__name__} does not compute both energies and forces"
        )
    properties = ("energy", "forces") + (
        ("atomic_energies",) if "atomic_energies" in calculator.implemented else ()
    )
    inverse_mass = (_FTM2V / _masses(system))[:, None]

    cutoff = float(calculator.cutoff)
    wide: NeighborList | None = None

    def evaluate(current: System) -> tuple[CalculatorResult, float]:
        # the list of the calculator's cutoff, as a new search would give it
        nonlocal wide
        neighbors = None
        if 0.0 < cutoff < np.inf:
            neighbors = wide._follow(current, cutoff) if wide is not None else None
            if neighbors is None:
                wide = NeighborList.build(current, cutoff + _SKIN)
                neighbors = wide._follow(current, cutoff)
        result = calculator.compute(current, properties, neighbors=neighbors)
        forces = result.forces
        largest = float(np.sqrt((forces * forces).sum(axis=1)).max(initial=0.0))
        return result, largest

    dt, alpha = timestep, _ALPHA0
    dt_max, dt_min = _TMAX * timestep, _TMIN * timestep
    positions = system.positions.copy()
    velocities = np.zeros_like(positions)
    current = system
    result, largest = evaluate(current)
    step, last_negative, reset = 0, 0, True
    frames: list[System] = []

    def save() -> None:
        frames.append(result.attach(current).with_info(timestep=step))

    if save_every is not None:
        save()
    while not largest < max_force and step < max_steps:
        step += 1
        forces = result.forces
        power = float(np.vdot(velocities, forces))
        if power > 0.0:
            # mix the velocities toward the forces
            keep = 1.0 - alpha
            ff = float(np.vdot(forces, forces))
            vv = float(np.vdot(velocities, velocities))
            turn = alpha * np.sqrt(vv / ff) if ff > 1e-20 else 0.0
            if step - last_negative > _DELAY:
                dt = min(dt * _GROW, dt_max)
                alpha *= _ALPHA_SHRINK
        else:
            # uphill: shrink the time step after the first steps, go back
            # half a step and stop
            last_negative = step
            if step >= _DELAY:
                alpha = _ALPHA0
                if dt * _SHRINK >= dt_min:
                    dt *= _SHRINK
            if velocities.any():
                positions = positions - 0.5 * dt * velocities
                current = current.with_positions(positions)
                result, largest = evaluate(current)
                forces = result.forces
            velocities[:] = 0.0
            reset = True
        # no atom moves more than max_step along an axis
        estimate = dt * inverse_mass * forces if reset else velocities
        fastest = float(np.abs(estimate).max(initial=0.0))
        dtv = max_step / fastest if dt * fastest > max_step else dt
        velocities += dtv * inverse_mass * forces
        if power > 0.0:
            velocities = keep * velocities + turn * forces
        positions = positions + dtv * velocities
        current = current.with_positions(positions)
        result, largest = evaluate(current)
        reset = False
        if save_every is not None and step % save_every == 0:
            save()
        if callback is not None and callback(step, current, result.energy, largest):
            break
    if save_every is not None and frames[-1].info["timestep"] != step:
        save()
    return MinimizeResult(
        system=result.attach(current),
        converged=largest < max_force,
        steps=step,
        energy=float(result.energy),
        max_force=largest,
        frames=tuple(frames) if save_every is not None else None,
    )
