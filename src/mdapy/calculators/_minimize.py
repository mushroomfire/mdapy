# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Energy minimization of the positions, and optionally the box, with FIRE."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from mdapy import units
from mdapy.calculators._base import Calculator
from mdapy.core.box import Box
from mdapy.core.system import System
from mdapy.errors import InvalidParameter

#: How the box may change: not at all, by one scale factor, along its
#: Cartesian axes, or by any homogeneous strain.
CELL_MODES = (None, "isotropic", "axes", "full")


@dataclass(frozen=True)
class MinimizeResult:
    """The outcome of :func:`minimize`.

    Attributes
    ----------
    system : System
        The last configuration, with ``forces`` and ``atomic_energies``
        columns and ``energy`` in ``info``.
    converged : bool
        Whether every force fell below ``max_force`` and, when the box was
        relaxed, every relaxed stress component within ``max_stress`` of the
        target.
    steps : int
        Number of steps taken.
    energy : float
        Energy of the last configuration in eV.
    max_force : float
        Largest force of the last configuration in eV/Å.
    stress : numpy.ndarray or None
        (3, 3) stress of the last configuration in GPa when the box was
        relaxed.
    """

    system: System
    converged: bool
    steps: int
    energy: float
    max_force: float
    stress: np.ndarray | None = None


def _projector(cell: str | None, pbc: np.ndarray):
    """The projection of a (3, 3) tensor onto the strains the mode allows."""
    periodic = np.outer(pbc, pbc)
    if cell == "full":
        return lambda m: np.where(periodic, m, 0.0)
    if cell == "axes":
        return lambda m: np.where(np.eye(3, dtype=bool) & periodic, m, 0.0)

    def isotropic(m):
        mean = np.trace(m * np.diag(pbc)) / max(pbc.sum(), 1)
        return np.diag(np.where(pbc, mean, 0.0))

    return isotropic


def minimize(
    system: System,
    calculator: Calculator,
    *,
    max_force: float = 1e-3,
    max_steps: int = 1000,
    max_step: float = 0.2,
    cell: str | None = None,
    pressure: float = 0.0,
    max_stress: float = 1e-2,
    callback: Any = None,
) -> MinimizeResult:
    """Minimizes the energy over the positions, and optionally the box, with
    FIRE.

    Parameters
    ----------
    system : System
        The starting configuration.
    calculator : Calculator
        Computes energies and forces, and the virial when the box relaxes.
    max_force : float, optional
        Convergence: the largest force on any atom, in eV/Å. Default 1e-3.
    max_steps : int, optional
        Largest number of steps. Default 1000.
    max_step : float, optional
        Largest distance an atom moves in one step, in Å. Default 0.2.
    cell : {None, "isotropic", "axes", "full"}, optional
        How the box may change along its periodic directions: not at all
        (default), by one scale factor, by a scale factor along each
        Cartesian axis, or by any homogeneous strain.
    pressure : float, optional
        External hydrostatic pressure in GPa when the box relaxes; the
        enthalpy is minimized. Default 0.
    max_stress : float, optional
        Convergence when the box relaxes: the largest deviation of a relaxed
        stress component from the target, in GPa. Default 0.01.
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
    if cell not in CELL_MODES:
        raise InvalidParameter(f"cell must be one of {CELL_MODES}, not {cell!r}")
    if cell is not None and not max_stress > 0:
        raise InvalidParameter("max_stress must be positive")
    needed = {"energy", "forces"} | ({"virial"} if cell else set())
    if not needed <= calculator.implemented:
        raise InvalidParameter(
            f"{type(calculator).__name__} does not compute {sorted(needed)}"
        )
    properties = tuple(sorted(needed)) + (
        ("atomic_energies",) if "atomic_energies" in calculator.implemented else ()
    )

    # FIRE parameters of Bitzek et al. (2006), as in ASE.
    dt, dt_max, n_min = 0.1, 1.0, 5
    f_inc, f_dec, alpha_start, f_alpha = 1.1, 0.5, 0.1, 0.99
    alpha, positive = alpha_start, 0

    # The box relaxes by a deformation gradient F of the starting box; the
    # atoms move with it, so the variables are the undeformed positions u
    # (x = origin + u F^T) and F scaled by the number of atoms.
    n = len(system)
    box0, origin = system.box, system.box.origin
    project = _projector(cell, box0.pbc)
    scale = float(max(n, 1))
    deformation = np.eye(3)
    variables = system.positions - origin
    if cell:
        variables = np.vstack([variables, scale * deformation])
    velocities = np.zeros_like(variables)
    current = system
    stress = None
    step = 0
    while True:
        result = calculator.compute(current, properties)
        forces = result.forces
        largest = float(np.sqrt((forces * forces).sum(axis=1)).max()) if n else 0.0
        generalized = forces
        deviation = 0.0
        if cell:
            volume = current.box.volume
            stress = -result.virial / volume * units.EV_PER_A3_TO_GPA
            deviation = float(np.abs(project(stress + pressure * np.eye(3))).max())
            target = (
                result.virial - pressure * units.GPA_TO_EV_PER_A3 * volume * np.eye(3)
            )
            generalized = np.vstack(
                [
                    forces @ deformation,
                    project(target) @ np.linalg.inv(deformation).T / scale,
                ]
            )
        converged = largest < max_force and deviation < max_stress
        if callback is not None and step > 0:
            if callback(step, current, result.energy, largest):
                break
        if converged or step >= max_steps:
            break
        power = float(np.vdot(generalized, velocities))
        if power > 0.0:
            norm_f = np.sqrt(np.vdot(generalized, generalized))
            norm_v = np.sqrt(np.vdot(velocities, velocities))
            velocities = (
                1.0 - alpha
            ) * velocities + alpha * generalized / norm_f * norm_v
            if positive > n_min:
                dt = min(dt * f_inc, dt_max)
                alpha *= f_alpha
            positive += 1
        else:
            velocities[:] = 0.0
            alpha = alpha_start
            dt *= f_dec
            positive = 0
        velocities += dt * generalized
        move = dt * velocities
        longest = np.sqrt((move * move).sum(axis=1)).max()
        if longest > max_step:
            move *= max_step / longest
        variables = variables + move
        if cell:
            deformation = variables[n:] / scale
            box = Box(box0.matrix @ deformation.T, origin=origin, pbc=box0.pbc)
            current = current.with_box(box).with_positions(
                origin + variables[:n] @ deformation.T
            )
        else:
            current = current.with_positions(origin + variables)
        step += 1
    return MinimizeResult(
        system=result.attach(current),
        converged=converged,
        steps=step,
        energy=float(result.energy),
        max_force=largest,
        stress=stress,
    )
