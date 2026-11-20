# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Energy minimization of the positions, and optionally the box, with FIRE 2.0
as LAMMPS min_style fire."""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from mdapy import units
from mdapy.calculators._base import Calculator
from mdapy.core.box import Box
from mdapy.core.result import Result
from mdapy.core.system import System
from mdapy.elements import atomic_masses
from mdapy.errors import CalculationFailed, InvalidParameter, MissingProperty
from mdapy.neighbor import NeighborList
from mdapy.simulation._log import _Log, _machine, _on_gpu

# Neighbors are searched this much farther than the cutoff of the calculator,
# in Å, so that one search serves the steps until an atom has moved half as far.
_SKIN = 1.0

# the columns of the log: name, format, width
_LOG_COLUMNS = (
    ("step", "{:d}", 10),
    ("energy", "{:.8f}", 22),
    ("max_force", "{:.6e}", 14),
    ("max_stress", "{:.6f}", 14),
)
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
class MinimizeResult(Result):
    """The outcome of ``minimize``.

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
    steps : int32
        Number of steps taken.
    energy : float64
        Energy of the last configuration in eV.
    max_force : float64
        Largest force of the last configuration in eV/Å.
    stress : numpy.ndarray or None
        (3, 3) float64 stress of the last configuration in GPa, positive under
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


# Not with BLAS: after a product of large arrays the threads of OpenBLAS spin for
# milliseconds, and between the steps they take the cores from the kernels (seven
# times slower for 4000 atoms on 16 threads).
def _dot(a: np.ndarray, b: np.ndarray) -> float:
    """The sum of the products of the elements of two arrays of rows."""
    return float(np.einsum("ij,ij->", a, b))


def _rows_times(rows: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """``rows @ matrix``, for rows of positions and a (3, 3) matrix."""
    return np.einsum("ij,kj->ik", rows, matrix)


class _Neighbors:
    """The neighbor lists of a calculator for configurations that change
    little from one to the next: one search, with a skin, serves until an
    atom has moved half as far."""

    def __init__(self, cutoff: float):
        self.__cutoff = float(cutoff)
        self.__wide: NeighborList | None = None

    def of(self, system: System) -> NeighborList | None:
        """The list of system within the cutoff, or None for a calculator
        without a finite cutoff."""
        cutoff = self.__cutoff
        if not 0.0 < cutoff < np.inf:
            return None
        wide = self.__wide
        neighbors = wide._follow(system, cutoff) if wide is not None else None
        if neighbors is None:
            self.__wide = NeighborList.build(system, cutoff + _SKIN)
            neighbors = self.__wide._follow(system, cutoff)
        return neighbors


class _Fire:
    """FIRE 2.0 as LAMMPS min_style fire, a step at a time, over rows of
    variables with their masses.

    ``advance(evaluate)`` takes a step; ``evaluate(variables)`` gives the
    forces on new variables, once in a step, or twice after an uphill one.
    """

    def __init__(
        self,
        variables: np.ndarray,
        forces: np.ndarray,
        inverse_mass: np.ndarray,
        timestep: float,
        max_step: float,
    ):
        self.variables = variables
        self.forces = forces
        self.velocities = np.zeros_like(variables)
        self.step = 0
        self.__inverse_mass = inverse_mass
        self.__max_step = max_step
        self.__dt, self.__alpha = timestep, _ALPHA0
        self.__dt_max, self.__dt_min = _TMAX * timestep, _TMIN * timestep
        self.__last_negative, self.__reset = 0, True

    def advance(self, evaluate: Any) -> None:
        self.step += 1
        step = self.step
        velocities, forces = self.velocities, self.forces
        power = _dot(velocities, forces)
        if power > 0.0:
            # mix the velocities toward the forces
            keep = 1.0 - self.__alpha
            ff = _dot(forces, forces)
            vv = _dot(velocities, velocities)
            turn = self.__alpha * np.sqrt(vv / ff) if ff > 1e-20 else 0.0
            if step - self.__last_negative > _DELAY:
                self.__dt = min(self.__dt * _GROW, self.__dt_max)
                self.__alpha *= _ALPHA_SHRINK
        else:
            # uphill: shrink the time step after the first steps, go back
            # half a step and stop
            self.__last_negative = step
            if step >= _DELAY:
                self.__alpha = _ALPHA0
                if self.__dt * _SHRINK >= self.__dt_min:
                    self.__dt *= _SHRINK
            if velocities.any():
                self.variables = self.variables - 0.5 * self.__dt * velocities
                forces = self.forces = evaluate(self.variables)
            velocities[:] = 0.0
            self.__reset = True
        # no variable moves more than max_step in one step
        dt, inverse_mass = self.__dt, self.__inverse_mass
        estimate = dt * inverse_mass * forces if self.__reset else velocities
        fastest = float(np.abs(estimate).max(initial=0.0))
        dtv = self.__max_step / fastest if dt * fastest > self.__max_step else dt
        velocities += dtv * inverse_mass * forces
        if power > 0.0:
            velocities = keep * velocities + turn * forces
        self.velocities = velocities
        self.variables = self.variables + dtv * velocities
        self.forces = evaluate(self.variables)
        self.__reset = False


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
    log: Any = None,
    log_every: int = 10,
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
    max_force : float64, optional
        Convergence: the largest force on any atom, in eV/Å. Default 1e-3.
    max_steps : int32, optional
        Largest number of steps. Default 1000.
    max_step : float64, optional
        Largest displacement of an atom along x, y or z in one step, in Å.
        Default 0.1.
    timestep : float64, optional
        First time step of the dynamics, in ps. Default 0.001.
    cell : {None, "isotropic", "axes", "full"}, optional
        How the box may change along its periodic directions: not at all
        (default), by one scale factor, by a scale factor along each
        Cartesian axis, or by any strain without rotation.
    pressure : float64, optional
        External hydrostatic pressure in GPa when the box relaxes; the
        enthalpy is minimized. Default 0.
    max_stress : float64, optional
        Convergence when the box relaxes: the largest deviation of a relaxed
        stress component from the target, in GPa. Default 0.01.
    save_every : int32, optional
        Keeps the configuration every this many steps, with the first and
        the last, in ``frames`` of the result. Default None, keeping none.
    callback : callable, optional
        Called after every step as ``callback(step, system, energy,
        max_force)``; returning True stops the minimization.
    log : str, os.PathLike or stream, optional
        Writes the minimization to this file, which it overwrites, or to
        this stream, such as ``sys.stdout``: a heading, the energy, the
        largest force and, when the box relaxes, the largest deviation of
        the stress from its target every ``log_every`` steps and at the
        last, with the seconds since the start, and how it ended. Lines that
        are not rows of numbers start with #, so that ``numpy.loadtxt``
        reads the rows. Default None, writing none.
    log_every : int32, optional
        Writes a row to the log every this many steps. Default 10.

    Returns
    -------
    MinimizeResult
        The last configuration and whether it converged.

    Raises
    ------
    InvalidParameter
        For invalid options, a calculator without the properties needed,
        or a log that cannot be written.
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
    if isinstance(log_every, bool) or not isinstance(log_every, int) or log_every < 1:
        raise InvalidParameter(
            f"log_every must be a positive integer, not {log_every!r}"
        )
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
            positions=origin + _rows_times(variables[:n], deformation.T)
        )

    lists = _Neighbors(calculator.cutoff)

    def evaluate(variables: np.ndarray):
        # the result, the largest force, the stress, the forces on the
        # variables and the largest deviation of the stress from its target
        current = place(variables)
        result = calculator.compute(current, properties, neighbors=lists.of(current))
        if not (np.isfinite(result.energy) and np.isfinite(result.forces).all()):
            raise CalculationFailed(
                "the energy or the forces are not finite: atoms may overlap"
            )
        forces = result.forces if movable is None else result.forces * movable
        largest = float(np.sqrt((forces * forces).sum(axis=1)).max(initial=0.0))
        stress = result.stress if "virial" in properties else None
        generalized, deviation = forces, 0.0
        if cell:
            deformation = variables[n:] / scale
            volume = current.box.volume
            deviation = float(np.abs(project(stress + pressure * np.eye(3))).max())
            target = (
                result.virial - pressure * units.GPA_TO_EV_PER_A3 * volume * np.eye(3)
            )
            on_box = project(target) @ np.linalg.inv(deformation).T / scale
            generalized = np.vstack(
                [_rows_times(forces, deformation), 0.5 * (on_box + on_box.T)]
            )
        return current, result, largest, stress, generalized, deviation

    variables = system["positions"].copy()
    if cell:
        variables = np.vstack([variables - origin, scale * np.eye(3)])
    current, result, largest, stress, forces, deviation = evaluate(variables)

    def forces_at(variables: np.ndarray) -> np.ndarray:
        nonlocal current, result, largest, stress, deviation
        current, result, largest, stress, forces, deviation = evaluate(variables)
        return forces

    fire = _Fire(variables, forces, inverse_mass, timestep, max_step)
    step = 0
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
    writer = _Log(log)
    if writer.on:
        lengths = " ".join(f"{v:.4f}" for v in np.linalg.norm(box0.matrix, axis=1))
        relax = (
            f"the box relaxes ({cell}) at {pressure!r} GPa, max_stress {max_stress!r} GPa"
            if cell
            else "the box is kept"
        )
        writer.title("energy minimization (FIRE 2.0)")
        writer.comment(
            f"{n} atoms; box vectors {lengths} Å long",
            f"potential: {calculator!r}",
            *([f"GPU: {on}"] if (on := _on_gpu(calculator)) is not None else []),
            f"machine: {_machine()}",
            f"max_force {max_force!r} eV/Å, max_steps {max_steps}, {relax}",
            "",
        )
        writer.columns(_LOG_COLUMNS)

    def write() -> None:
        writer.row(step, float(result.energy), largest, deviation if cell else math.nan)

    try:
        write()
        while not done() and step < max_steps:
            fire.advance(forces_at)
            step = fire.step
            if save_every is not None and step % save_every == 0:
                save()
            if step % log_every == 0:
                write()
            if callback is not None and callback(step, current, result.energy, largest):
                break
        if step % log_every != 0:
            write()
        if writer.on:
            how = (
                "converged"
                if done()
                else (
                    "stopped by the callback"
                    if step < max_steps
                    else "reached max_steps"
                )
            )
            writer.comment(f"{how} after {step} steps, {writer.wall:.2f} s")
    finally:
        writer.close()
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
