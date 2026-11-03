# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The free energy of a stacking fault with the temperature."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from mdapy import units
from mdapy.build._defects import _axis, slip
from mdapy.calculators._lammps import LAMMPS
from mdapy.core.result import Result
from mdapy.core.system import System
from mdapy.core.threads import CancelToken
from mdapy.errors import InvalidParameter, MissingProperty
from mdapy.simulation._defect_energies import _options, _relaxed
from mdapy.simulation._free_energy import (
    _frenkel_ladd,
    _Progress,
    _solid_state,
    _spring_constants,
)


@dataclass(frozen=True)
class FaultFreeEnergyResult(Result):
    """The free energy of a stacking fault per area.

    Attributes
    ----------
    temperatures : numpy.ndarray
        (T,) the temperatures in K.
    energy : numpy.ndarray
        (T,) the free energy per area in J/m².
    error : numpy.ndarray
        (T,) its standard error in J/m², from the repeats; zero for one repeat.
    area : float
        The area of the fault at the first temperature in Å².
    volume : numpy.ndarray
        (T,) the volume per atom of the crystal in Å³.
    dissipation : numpy.ndarray
        (T,) half the difference in eV per atom of the work there and back, the
        largest over the runs: near zero when the switching is slow enough.
    """

    temperatures: np.ndarray
    energy: np.ndarray
    error: np.ndarray
    area: float
    volume: np.ndarray
    dissipation: np.ndarray


def _between_layers(crystal: System, k: int) -> float:
    """The fractional height along box vector k in the widest gap between atoms
    nearest the middle: a plane through a layer would divide it by rounding."""
    heights = np.sort(crystal.box.to_fractional(crystal["positions"])[:, k] % 1.0)
    middles = 0.5 * (heights[1:] + heights[:-1])
    gaps = np.diff(heights)
    wide = gaps > 0.5 * gaps.max()
    return float(middles[wide][np.abs(middles[wide] - 0.5).argmin()])


def _free_energies(
    crystal: System,
    potential: LAMMPS,
    temperature: float,
    timestep: float,
    settle: int,
    switch: int,
    repeats: int,
    seed: int,
    counter: _Progress,
) -> tuple[np.ndarray, float]:
    """The free energy of each repeat in eV, of all atoms of a crystal at constant
    volume, and the largest dissipation per atom."""
    springs = _spring_constants(
        crystal, potential, temperature, timestep, settle, seed, counter
    )
    runs = [
        _frenkel_ladd(
            crystal,
            potential,
            temperature,
            springs,
            timestep,
            settle,
            switch,
            seed + 1 + k,
            counter,
        )
        for k in range(repeats)
    ]
    values = np.array([value for value, _ in runs]) * len(crystal)
    return values, max(abs(lost) for _, lost in runs)


def stacking_fault_free_energy(
    crystal: System,
    potential: LAMMPS,
    direction: ArrayLike,
    temperatures: float | ArrayLike,
    *,
    height: float | None = None,
    axis: int | str = 2,
    pressure: float = 0.0,
    timestep: float = 0.001,
    equilibration: float = 10.0,
    switching: float = 50.0,
    repeats: int = 4,
    max_force: float = 1e-4,
    max_steps: int = 10000,
    seed: int = 1,
    progress: Callable[[int, int], None] | None = None,
    cancel: CancelToken | None = None,
) -> FaultFreeEnergyResult:
    """The free energy of a stacking fault at each temperature: the difference of
    the free energies of the crystal with and without it, in one box of the size
    the crystal has at that temperature and pressure.

    Parameters
    ----------
    crystal : System
        The crystal, relaxed, with an ``element`` column, periodic in every
        direction, with the plane of the fault normal to the Cartesian axis of
        axis, such as fcc with z along [111]. The wider the plane the smaller the
        error, as the error of the free energy of the atoms grows more slowly than
        the area does.
    potential : LAMMPS
        The interaction.
    direction : array_like
        (3,) the slip vector in Å, in the plane, such as a/6 [11-2].
    temperatures : float or array_like
        Temperatures in K, positive.
    height : float, optional
        The fault, at this fractional coordinate along the box vector of axis.
        Default between the two layers of atoms nearest the middle of the box.
    axis : int or str, optional
        The Cartesian axis normal to the fault. Default z.
    pressure : float, optional
        Pressure in GPa. Default 0.
    timestep : float, optional
        Time step in ps. Default 0.001.
    equilibration : float, optional
        Time in ps of each equilibration. Default 10.
    switching : float, optional
        Time in ps of each switching between the crystal and the springs.
        Default 50.
    repeats : int, optional
        Independent free energies of each crystal at each temperature. Default 4.
    max_force : float, optional
        Convergence in eV/Å of the relaxation of the fault at each size of the box.
        Default 1e-4.
    max_steps : int, optional
        Largest number of steps of each relaxation. Default 10000.
    seed : int, optional
        Seed of the random numbers, positive. Default 1.
    progress : callable, optional
        Called as ``progress(done, total)`` with the steps run.
    cancel : CancelToken, optional
        Stops the calculation, which raises :class:`mdapy.errors.Cancelled`.

    Returns
    -------
    FaultFreeEnergyResult
        The free energy of the fault per area, and its error.

    Raises
    ------
    InvalidParameter
        For invalid options, a plane not normal to the axis, a box not periodic in
        every direction, or if LAMMPS fails.
    MissingProperty
        If the system has no ``element`` column.
    Cancelled
        If cancelled.
    """
    if not isinstance(potential, LAMMPS):
        raise InvalidParameter("potential must be a mdapy.calculators.LAMMPS")
    if "element" not in crystal.table.columns:
        raise MissingProperty("the free energy needs the element column")
    if not crystal.box.pbc.all():
        raise InvalidParameter(
            "the free energy needs a box periodic in every direction"
        )
    wanted = np.atleast_1d(np.asarray(temperatures, dtype=float))
    if wanted.ndim != 1 or not (wanted > 0).all():
        raise InvalidParameter("temperatures must be positive")
    if not min(timestep, equilibration, switching) > 0:
        raise InvalidParameter("timestep, equilibration and switching must be positive")
    if int(repeats) < 1 or int(seed) < 1:
        raise InvalidParameter("repeats and seed must be positive")
    k = _axis(axis)
    others = [i for i in range(3) if i != k]
    normal = np.cross(crystal.box.matrix[others[0]], crystal.box.matrix[others[1]])
    if np.linalg.norm(np.delete(normal, k)) > 1e-9 * np.linalg.norm(normal):
        raise InvalidParameter(
            "the plane of the fault must be normal to the Cartesian axis"
        )
    vector = np.asarray(direction, dtype=float)
    if vector.shape != (3,):
        raise InvalidParameter("direction must have three components")
    settle = max(int(round(equilibration / timestep)), 1)
    switch = max(int(round(switching / timestep)), 1)
    repeats, seed = int(repeats), int(seed)
    total = len(wanted) * (3 * settle + 4 * repeats * (settle + switch))
    counter = _Progress(total, progress, cancel)
    n = len(crystal)
    movable = np.zeros((n, 3), dtype=bool)
    movable[:, k] = True
    options = _options(max_force, max_steps)
    energies, errors, volumes, lost = [], [], [], []
    area = None
    for j, temperature in enumerate(wanted):
        temperature = float(temperature)
        start = seed + 1000 * j
        perfect, _, volume = _solid_state(
            crystal,
            potential,
            temperature,
            pressure,
            "aniso",
            timestep,
            settle,
            start,
            counter,
        )
        plane = _between_layers(perfect, k) if height is None else height
        faulted = slip(perfect, vector, height=plane, axis=k)
        _, relaxed = _relaxed(faulted.with_columns(movable=movable), potential, options)
        relaxed = relaxed.drop("movable")
        plain, wasted = _free_energies(
            perfect,
            potential,
            temperature,
            timestep,
            settle,
            switch,
            repeats,
            start + 10,
            counter,
        )
        defect, wasted_too = _free_energies(
            relaxed,
            potential,
            temperature,
            timestep,
            settle,
            switch,
            repeats,
            start + 40,
            counter,
        )
        size = float(
            np.linalg.norm(
                np.cross(perfect.box.matrix[others[0]], perfect.box.matrix[others[1]])
            )
        )
        area = size if area is None else area
        scale = units.EV_PER_A2_TO_J_PER_M2 / size
        energies.append((defect.mean() - plain.mean()) * scale)
        spread = (
            (plain.var(ddof=1) + defect.var(ddof=1)) / repeats if repeats > 1 else 0.0
        )
        errors.append(float(np.sqrt(spread)) * scale)
        volumes.append(volume)
        lost.append(max(wasted, wasted_too))
    return FaultFreeEnergyResult(
        temperatures=wanted,
        energy=np.array(energies),
        error=np.array(errors),
        area=float(area),
        volume=np.array(volumes),
        dissipation=np.array(lost),
    )
