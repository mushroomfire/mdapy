# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Energy minimization of the positions, and optionally the box, with FIRE 2.0
as LAMMPS min_style fire."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from mdapy import units
from mdapy.calculators._base import Calculator
from mdapy.core.box import Box
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

#: How the box may change: not at all, by one scale factor, along its
#: Cartesian axes, or by any strain without rotation.
CELL_MODES = (None, "isotropic", "axes", "full")


@dataclass(frozen=True)
class MinimizeResult:
    """The outcome of :func:`minimize`.

    Attributes
    ----------
    system : System
        The last configuration, with ``forces`` and ``atomic_energies``
        columns and ``energy``, and with a virial ``virial`` and ``stress``,
        in ``info``.
    converged : bool
        Whether every force fell below ``max_force`` and, when the box
        relaxed, every relaxed stress component came within ``max_stress``
        of the target.
    steps : int
        Number of steps taken.
    energy : float
        Energy of the last configuration in eV.
    max_force : float
        Largest force of the last configuration in eV/Å.
    stress : numpy.ndarray or None
        (3, 3) stress of the last configuration in GPa, positive under
        tension; None when the calculator computes no virial.
    frames : tuple of System or None
        With ``save_every``, the configurations at step 0, every
        ``save_every`` steps and the last step, with the columns and
        information of ``system`` and the step number as ``timestep`` in
        ``info``; None otherwise.
    """

    system: System
    converged: bool
    steps: int
    energy: float
    max_force: float
    stress: np.ndarray | None = None
    frames: tuple[System, ...] | None = None


def _masses(system: System) -> np.ndarray:
    table = system.table
    if "mass" in table.columns:
        return table["mass"].to_numpy().astype(float)
    if "element" in table.columns:
        return atomic_masses(table["element"].to_physical().to_numpy())
    raise MissingProperty("minimize needs the masses: a mass or an element column")


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
    max_step: float = 0.1,
    timestep: float = 0.001,
    cell: str | None = None,
    pressure: float = 0.0,
    max_stress: float = 1e-2,
    save_every: int | None = None,
    callback: Any = None,
) -> MinimizeResult:
    """Minimizes the energy over the positions, and optionally the box, with
    FIRE 2.0.

    Parameters
    ----------
    system : System
        The starting configuration, with a ``mass`` or an ``element`` column;
        a ``movable`` column, (N, 3) bool, keeps each atom still along the
        directions where it is False.
    calculator : Calculator
        Computes energies and forces, and the virial when the box relaxes.
    max_force : float, optional
        Convergence: the largest force on any atom, in eV/Å. Default 1e-3.
    max_steps : int, optional
        Largest number of steps. Default 1000.
    max_step : float, optional
        Largest displacement of an atom along x, y or z in one step, in Å.
        Default 0.1.
    timestep : float, optional
        First time step of the dynamics, in ps. Default 0.001.
    cell : {None, "isotropic", "axes", "full"}, optional
        How the box may change along its periodic directions: not at all
        (default), by one scale factor, by a scale factor along each
        Cartesian axis, or by any strain without rotation.
    pressure : float, optional
        External hydrostatic pressure in GPa when the box relaxes; the
        enthalpy is minimized. Default 0.
    max_stress : float, optional
        Convergence when the box relaxes: the largest deviation of a relaxed
        stress component from the target, in GPa. Default 0.01.
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
        For invalid options, or a calculator without the properties needed.
    MissingProperty
        If the system has neither masses nor elements.
    """
    if not max_force > 0:
        raise InvalidParameter("max_force must be positive")
    if max_steps < 0 or not max_step > 0 or not timestep > 0:
        raise InvalidParameter(
            "max_steps must be at least 0, and max_step and timestep positive"
        )
    if cell not in CELL_MODES:
        raise InvalidParameter(f"cell must be one of {CELL_MODES}, not {cell!r}")
    if cell is not None and not max_stress > 0:
        raise InvalidParameter("max_stress must be positive")
    if save_every is not None and (
        isinstance(save_every, bool)
        or not isinstance(save_every, int)
        or save_every < 1
    ):
        raise InvalidParameter(
            f"save_every must be a positive integer, not {save_every!r}"
        )
    needed = {"energy", "forces"} | ({"virial"} if cell else set())
    if not needed <= calculator.implemented:
        raise InvalidParameter(
            f"{type(calculator).__name__} does not compute {sorted(needed)}"
        )
    properties = tuple(
        p
        for p in ("energy", "forces", "virial", "atomic_energies")
        if p in calculator.implemented
    )
    masses = _masses(system)
    # the directions in which each atom may move, as VASP's selective dynamics
    movable = (
        system.table["movable"].to_numpy().astype(bool)
        if "movable" in system.table.columns
        else None
    )

    # The box relaxes by a symmetric deformation gradient F of the starting
    # box, a strain without rotation; the atoms move with it, so the
    # variables are the positions u in the starting box (x = origin + u F^T)
    # and s F. With s = sqrt(N) (V/N)^(1/3) in Å, the rows of s F move like
    # atoms of the mean mass, about as stiff as the atoms whatever the size
    # of the system.
    n = len(system)
    box0, origin = system.box, system.box.origin
    project = _projector(cell, box0.pbc)
    scale = float(max(n, 1)) ** (1 / 6) * float(box0.volume) ** (1 / 3)
    if cell:
        cell_mass = float(masses.mean()) if n else 1.0
        masses = np.concatenate([masses, np.full(3, cell_mass)])
    inverse_mass = (_FTM2V / masses)[:, None]

    def place(variables: np.ndarray) -> System:
        if not cell:
            return system.with_columns(positions=variables)
        deformation = variables[n:] / scale
        box = Box(box0.matrix @ deformation.T, origin=origin, pbc=box0.pbc)
        return system.with_box(box).with_columns(
            positions=origin + variables[:n] @ deformation.T
        )

    cutoff = float(calculator.cutoff)
    wide: NeighborList | None = None

    def evaluate(variables: np.ndarray):
        # the result, the largest force, the stress, the forces on the
        # variables and the largest deviation of the stress from its target
        nonlocal wide
        current = place(variables)
        neighbors = None
        if 0.0 < cutoff < np.inf:
            neighbors = wide._follow(current, cutoff) if wide is not None else None
            if neighbors is None:
                wide = NeighborList.build(current, cutoff + _SKIN)
                neighbors = wide._follow(current, cutoff)
        result = calculator.compute(current, properties, neighbors=neighbors)
        forces = result.forces if movable is None else result.forces * movable
        largest = float(np.sqrt((forces * forces).sum(axis=1)).max(initial=0.0))
        stress = result.stress(current) if "virial" in properties else None
        generalized, deviation = forces, 0.0
        if cell:
            deformation = variables[n:] / scale
            volume = current.box.volume
            deviation = float(np.abs(project(stress + pressure * np.eye(3))).max())
            target = (
                result.virial - pressure * units.GPA_TO_EV_PER_A3 * volume * np.eye(3)
            )
            on_box = project(target) @ np.linalg.inv(deformation).T / scale
            generalized = np.vstack([forces @ deformation, 0.5 * (on_box + on_box.T)])
        return current, result, largest, stress, generalized, deviation

    variables = system["positions"].copy()
    if cell:
        variables = np.vstack([variables - origin, scale * np.eye(3)])
    velocities = np.zeros_like(variables)
    current, result, largest, stress, forces, deviation = evaluate(variables)
    dt, alpha = timestep, _ALPHA0
    dt_max, dt_min = _TMAX * timestep, _TMIN * timestep
    step, last_negative, reset = 0, 0, True
    frames: list[System] = []

    def frame() -> System:
        out = result.attach(current)
        return out if stress is None else out.with_info(stress=stress)

    def save() -> None:
        frames.append(frame().with_info(timestep=step))

    def done() -> bool:
        return largest < max_force and deviation < max_stress

    if save_every is not None:
        save()
    while not done() and step < max_steps:
        step += 1
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
                variables = variables - 0.5 * dt * velocities
                current, result, largest, stress, forces, deviation = evaluate(
                    variables
                )
            velocities[:] = 0.0
            reset = True
        # no variable moves more than max_step in one step
        estimate = dt * inverse_mass * forces if reset else velocities
        fastest = float(np.abs(estimate).max(initial=0.0))
        dtv = max_step / fastest if dt * fastest > max_step else dt
        velocities += dtv * inverse_mass * forces
        if power > 0.0:
            velocities = keep * velocities + turn * forces
        variables = variables + dtv * velocities
        current, result, largest, stress, forces, deviation = evaluate(variables)
        reset = False
        if save_every is not None and step % save_every == 0:
            save()
        if callback is not None and callback(step, current, result.energy, largest):
            break
    if save_every is not None and frames[-1].info["timestep"] != step:
        save()
    return MinimizeResult(
        system=frame(),
        converged=done(),
        steps=step,
        energy=float(result.energy),
        max_force=largest,
        stress=stress,
        frames=tuple(frames) if save_every is not None else None,
    )
