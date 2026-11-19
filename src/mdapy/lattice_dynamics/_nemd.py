# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The thermal conductivity by nonequilibrium molecular dynamics (NEMD): the
heat that a hot and a cold bath drive through a crystal, over the gradient
of temperature between them."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from mdapy import units
from mdapy.calculators._base import Calculator
from mdapy.core.system import System
from mdapy.core.threads import CancelToken
from mdapy.errors import InvalidParameter
from mdapy.lattice_dynamics._green_kubo import _W_PER_M_K
from mdapy.lattice_dynamics._phonons import _masses
from mdapy.simulation._controls import Langevin
from mdapy.simulation._dynamics import MolecularDynamics


@dataclass(frozen=True)
class NEMD:
    """The thermal conductivity of nonequilibrium molecular dynamics.

    Attributes
    ----------
    temperature : float64
        The mean temperature of the atoms during the measurement in K.
    kappa : float64
        The conductivity along the axis in W/(m K).
    error : float64
        Its standard error in W/(m K), from blocks of the measurement.
    flux : float64
        The heat flux through each half of the box in eV/(ps Å²).
    gradient : float64
        The gradient of temperature in K/Å, the mean of the two halves.
    positions : numpy.ndarray
        (B,) float64 the middle of each slab along the axis in Å, from the
        middle of the cold bath.
    temperatures : numpy.ndarray
        (B,) float64 the mean temperature of each slab in K.
    times : numpy.ndarray
        (S,) float64 the time in ps of each record of the measurement.
    hot : numpy.ndarray
        (S,) float64 the heat in eV the hot bath gave since the baths began.
    cold : numpy.ndarray
        (S,) float64 the heat in eV the cold bath gave, negative.
    length : float64
        The length of the box along the axis, normal to the slabs, in Å.
    area : float64
        The area of the slabs in Å².
    """

    temperature: float
    kappa: float
    error: float
    flux: float
    gradient: float
    positions: np.ndarray
    temperatures: np.ndarray
    times: np.ndarray
    hot: np.ndarray
    cold: np.ndarray
    length: float
    area: float

    def __post_init__(self):
        for array in (
            self.positions,
            self.temperatures,
            self.times,
            self.hot,
            self.cold,
        ):
            array.flags.writeable = False

    def __repr__(self) -> str:
        return (
            f"NEMD({self.kappa:.4g} ± {self.error:.2g} W/(m K) at "
            f"{self.temperature:.6g} K over {self.length:.4g} Å)"
        )


def _slopes(times: np.ndarray, values: np.ndarray) -> float:
    """The slope of the least-squares line of values over times."""
    return float(np.polyfit(times, values, 1)[0])


def _gradient(profile: np.ndarray, positions: np.ndarray) -> float:
    """The mean of the gradients of the two halves between the baths, of
    their slabs but the ones next to the baths."""
    bins = len(profile)
    half = bins // 2
    left = np.arange(2, half - 1)
    right = np.arange(half + 2, bins - 1)
    up = np.polyfit(positions[left], profile[left], 1)[0]
    down = np.polyfit(positions[right], profile[right], 1)[0]
    return 0.5 * (float(up) - float(down))


def nemd(
    system: System,
    potential: Calculator,
    temperature: float,
    *,
    difference: float | None = None,
    axis: int = 2,
    bins: int = 20,
    damping: float = 0.1,
    timestep: float = 0.001,
    equilibration: int = 10000,
    relaxation: int = 50000,
    production: int = 200000,
    every: int = 100,
    blocks: int = 5,
    seed: int = 1,
    progress: Callable[[int, int], None] | None = None,
    cancel: CancelToken | None = None,
) -> NEMD:
    """The thermal conductivity of a crystal along a box vector by
    nonequilibrium molecular dynamics (NEMD): the box cut into slabs across
    the vector, Langevin baths in the slab at its start, cold, and in the
    middle one, hot, the rest at constant energy; the heat the baths drive
    through each half over the gradient of temperature there.

    Parameters
    ----------
    system : System
        The crystal, with an ``element`` column, in a box periodic along
        ``axis``, of its size at the temperature, long along the axis.
    potential : Calculator
        The interaction: any calculator of energies and forces; the dynamics
        run in MDAPY, on the processor or the GPU, or in LAMMPS for a LAMMPS
        calculator.
    temperature : float64
        The mean temperature in K, positive.
    difference : float64, optional
        The difference of the temperatures of the baths in K, positive and
        less than twice the temperature. Default a fifth of the temperature.
    axis : int32, optional
        The box vector along which heat flows, 0, 1 or 2. Default 2.
    bins : int32, optional
        The slabs across it, even and at least 10, so that each half has
        two slabs that are not next to a bath. Default 20.
    damping : float64, optional
        The damping time of the baths in ps. Default 0.1.
    timestep : float64, optional
        Time step in ps. Default 0.001.
    equilibration : int32, optional
        Steps at the temperature with a Nosé-Hoover thermostat first.
        Default 10000.
    relaxation : int32, optional
        Steps with the baths before the measurement, for the steady state.
        Default 50000.
    production : int32, optional
        Steps of the measurement. Default 200000.
    every : int32, optional
        Steps between the samples of the temperatures of the slabs and of
        the heat of the baths. Default 100.
    blocks : int32, optional
        Blocks of the measurement for the standard error, at least 2.
        Default 5.
    seed : int32, optional
        Seed of the starting velocities and of the baths, positive. Default 1.
    progress : callable, optional
        Called as ``progress(done, total)`` with the steps run.
    cancel : CancelToken, optional
        Stops the calculation, which raises ``mdapy.errors.Cancelled``.

    Returns
    -------
    NEMD
        The conductivity, the flux, the gradient and the profile of the
        temperature.

    Raises
    ------
    InvalidParameter
        For a box not periodic along the axis, a temperature, difference,
        axis, number of slabs or of steps that are not valid, slabs without
        atoms, or the errors of ``MolecularDynamics``.
    Cancelled
        If cancelled.
    """
    if not isinstance(potential, Calculator):
        raise InvalidParameter("potential must be a mdapy calculator")
    if axis not in (0, 1, 2) or isinstance(axis, bool):
        raise InvalidParameter(f"axis must be 0, 1 or 2, not {axis!r}")
    if not system.box.pbc[axis]:
        raise InvalidParameter("NEMD needs a box periodic along its axis")
    if not temperature > 0:
        raise InvalidParameter(f"temperature must be positive, not {temperature!r}")
    difference = 0.2 * temperature if difference is None else float(difference)
    if not 0 < difference < 2 * temperature:
        raise InvalidParameter(
            "difference must be positive and less than twice the temperature, "
            f"not {difference!r}"
        )
    if isinstance(bins, bool) or int(bins) != bins or bins < 10 or bins % 2:
        raise InvalidParameter(f"bins must be even and at least 10, not {bins!r}")
    bins = int(bins)
    steps = (int(equilibration), int(relaxation), int(production), int(every))
    if min(steps) < 1:
        raise InvalidParameter(
            "equilibration, relaxation, production and every must be positive"
        )
    if int(blocks) < 2 or int(production) // int(every) < 2 * int(blocks):
        raise InvalidParameter("blocks must be at least 2, with two samples each")
    if not damping > 0 or int(seed) < 1:
        raise InvalidParameter("damping and seed must be positive")
    box = system.box
    others = [k for k in range(3) if k != axis]
    area = float(np.linalg.norm(np.cross(box.matrix[others[0]], box.matrix[others[1]])))
    length = float(box.volume / area)
    slab = np.floor(box.to_fractional(system["positions"])[:, axis] % 1.0 * bins)
    slab = np.minimum(slab.astype(int), bins - 1)
    counts = np.bincount(slab, minlength=bins)
    if (counts == 0).any():
        raise InvalidParameter("every slab needs atoms: fewer bins, or a longer box")
    cold, hot = slab == 0, slab == bins // 2
    masses = _masses(system)
    total = int(equilibration) + int(relaxation) + int(production)
    done = 0

    def counted(steps_done: int, _: int) -> None:
        if progress is not None:
            progress(done + steps_done, total)

    def baths(offset: int) -> list:
        return [
            Langevin(
                hot,
                temperature + 0.5 * difference,
                damping=damping,
                seed=int(seed) + offset,
                name="hot",
            ),
            Langevin(
                cold,
                temperature - 0.5 * difference,
                damping=damping,
                seed=int(seed) + offset + 1,
                name="cold",
            ),
        ]

    samples = []  # the temperatures of the slabs at each sample
    with MolecularDynamics(system, potential, timestep=timestep) as md:
        md.velocities(2 * temperature, seed=int(seed))
        md.run(
            int(equilibration),
            ensemble="nvt",
            temperature=temperature,
            thermo=int(equilibration),
            progress=counted,
            cancel=cancel,
        )
        done = int(equilibration)
        md.run(
            int(relaxation),
            controls=baths(1),
            thermo=int(relaxation),
            progress=counted,
            cancel=cancel,
        )
        done += int(relaxation)
        start = len(md.thermo)
        frames = md.iterate(
            int(production),
            save_every=int(every),
            controls=baths(3),
            thermo=int(every),
            progress=counted,
            cancel=cancel,
        )
        next(frames)  # the atoms at the start, where the relaxation ended
        for frame in frames:
            where = frame.box.to_fractional(frame["positions"])[:, axis] % 1.0
            index = np.minimum((where * bins).astype(int), bins - 1)
            v = frame["velocities"]
            energy = masses * np.einsum("ij,ij->i", v, v) * units.AMU_A2_PER_PS2_TO_EV
            kinetic = np.bincount(index, energy, minlength=bins)
            number = np.bincount(index, minlength=bins)
            samples.append(kinetic / (3 * units.BOLTZMANN * np.maximum(number, 1)))
        # the records of the measurement, at the samples
        rows = md.thermo[start : start + len(samples)]
    times = rows["time"].to_numpy() - rows["time"][0]
    heat_hot = rows["hot_energy"].to_numpy()
    heat_cold = rows["cold_energy"].to_numpy()
    profiles = np.array(samples)
    positions = (np.arange(bins) + 0.5) / bins * length

    def conductivity(part: slice) -> tuple[float, float, float]:
        power = 0.5 * (
            _slopes(times[part], heat_hot[part]) - _slopes(times[part], heat_cold[part])
        )
        flux = power / (2 * area)  # through each half
        gradient = _gradient(profiles[part].mean(axis=0), positions)
        return flux / gradient * _W_PER_M_K, flux, gradient

    kappa, flux, gradient = conductivity(slice(None))
    parts = np.array_split(np.arange(len(times)), int(blocks))
    values = [conductivity(slice(p[0], p[-1] + 1))[0] for p in parts]
    error = float(np.std(values, ddof=1) / np.sqrt(int(blocks)))
    return NEMD(
        temperature=float(rows["temperature"].mean()),
        kappa=float(kappa),
        error=error,
        flux=float(flux),
        gradient=float(gradient),
        positions=positions,
        temperatures=profiles.mean(axis=0),
        times=times,
        hot=heat_hot,
        cold=heat_cold,
        length=length,
        area=area,
    )
