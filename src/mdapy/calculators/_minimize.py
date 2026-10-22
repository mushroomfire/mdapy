# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Energy minimization of the positions with FIRE."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from mdapy.calculators._base import Calculator
from mdapy.core.system import System
from mdapy.errors import InvalidParameter
from mdapy.neighbor import NeighborList

# Neighbors are searched this much farther than the cutoff of the calculator,
# in Å, so that one search serves the steps until an atom has moved half as far.
_SKIN = 1.0


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
    """

    system: System
    converged: bool
    steps: int
    energy: float
    max_force: float


def minimize(
    system: System,
    calculator: Calculator,
    *,
    max_force: float = 1e-3,
    max_steps: int = 1000,
    max_step: float = 0.2,
    callback: Any = None,
) -> MinimizeResult:
    """Minimizes the energy over the positions, in a fixed box, with FIRE.

    Parameters
    ----------
    system : System
        The starting configuration.
    calculator : Calculator
        Computes energies and forces.
    max_force : float, optional
        Convergence: the largest force on any atom, in eV/Å. Default 1e-3.
    max_steps : int, optional
        Largest number of steps. Default 1000.
    max_step : float, optional
        Largest distance an atom moves in one step, in Å. Default 0.2.
    callback : callable, optional
        Called after every step as ``callback(step, system, energy,
        max_force)``; returning True stops the minimization.

    Returns
    -------
    MinimizeResult
        The last configuration and whether it converged.
    """
    if not max_force > 0:
        raise InvalidParameter("max_force must be positive")
    if max_steps < 0 or not max_step > 0:
        raise InvalidParameter("max_steps must be at least 0 and max_step positive")
    if not {"energy", "forces"} <= calculator.implemented:
        raise InvalidParameter(
            f"{type(calculator).__name__} does not compute both energies and forces"
        )
    properties = ("energy", "forces") + (
        ("atomic_energies",) if "atomic_energies" in calculator.implemented else ()
    )

    # FIRE parameters of Bitzek et al. (2006), as in ASE.
    dt, dt_max, n_min = 0.1, 1.0, 5
    f_inc, f_dec, alpha_start, f_alpha = 1.1, 0.5, 0.1, 0.99
    alpha, positive = alpha_start, 0

    positions = system.positions.copy()
    velocities = np.zeros_like(positions)
    current = system
    step = 0
    cutoff = float(calculator.cutoff)
    wide = None
    while True:
        # the list of the calculator's cutoff, as a new search would give it
        neighbors = None
        if 0.0 < cutoff < np.inf:
            neighbors = wide._follow(current, cutoff) if wide is not None else None
            if neighbors is None:
                wide = NeighborList.build(current, cutoff + _SKIN)
                neighbors = wide._follow(current, cutoff)
        result = calculator.compute(current, properties, neighbors=neighbors)
        forces = result.forces
        largest = (
            float(np.sqrt((forces * forces).sum(axis=1)).max()) if len(forces) else 0.0
        )
        if callback is not None and step > 0:
            if callback(step, current, result.energy, largest):
                break
        if largest < max_force or step >= max_steps:
            break
        power = float(np.vdot(forces, velocities))
        if power > 0.0:
            norm_f = np.sqrt(np.vdot(forces, forces))
            norm_v = np.sqrt(np.vdot(velocities, velocities))
            velocities = (1.0 - alpha) * velocities + alpha * forces / norm_f * norm_v
            if positive > n_min:
                dt = min(dt * f_inc, dt_max)
                alpha *= f_alpha
            positive += 1
        else:
            velocities[:] = 0.0
            alpha = alpha_start
            dt *= f_dec
            positive = 0
        velocities += dt * forces
        move = dt * velocities
        longest = np.sqrt((move * move).sum(axis=1)).max()
        if longest > max_step:
            move *= max_step / longest
        positions = positions + move
        current = current.with_positions(positions)
        step += 1
    return MinimizeResult(
        system=result.attach(current),
        converged=largest < max_force,
        steps=step,
        energy=float(result.energy),
        max_force=largest,
    )
