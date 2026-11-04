# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Binary phase diagrams from the free energies of phases: at each temperature
the stable phases lie on the lower convex hull of the free energies."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from mdapy import units
from mdapy.core.result import Result
from mdapy.errors import InvalidParameter

# points of each end of a free energy that extend it past its temperatures
_EXTENDING = 5


@dataclass(frozen=True)
class TieLine:
    """Two phases in equilibrium at a temperature.

    Attributes
    ----------
    temperature : float64
        The temperature in K.
    first, second : str
        The phases, the first at the lower fraction.
    first_fraction, second_fraction : float64
        The fractions of the second element in each.
    """

    temperature: float
    first: str
    first_fraction: float
    second: str
    second_fraction: float


@dataclass(frozen=True)
class BinaryPhaseDiagram(Result):
    """The stable phases of two elements against composition and temperature.

    Attributes
    ----------
    elements : tuple of str
        The two elements; the fraction is that of the second.
    phases : tuple of str
        The names of the phases.
    temperatures : numpy.ndarray
        (T,) float64, the temperatures in K.
    fractions : numpy.ndarray
        (X,) float64, the fractions of the second element.
    stable : numpy.ndarray
        (T, X) int64, the index in ``phases`` of the one stable phase, or -1 where two
        phases coexist.
    tie_lines : tuple of TieLine
        Each pair of phases in equilibrium, by temperature and fraction.
    """

    elements: tuple[str, str]
    phases: tuple[str, ...]
    temperatures: np.ndarray
    fractions: np.ndarray
    stable: np.ndarray
    tie_lines: tuple[TieLine, ...]

    def phases_at(self, temperature: float, fraction: float) -> tuple[str, ...]:
        """The one or two stable phases at the nearest temperature of the diagram.

        Parameters
        ----------
        temperature : float64
            The temperature in K.
        fraction : float64
            The fraction of the second element, from 0 to 1.

        Returns
        -------
        tuple of str
            The phase, or the two phases in equilibrium.

        Raises
        ------
        InvalidParameter
            For a fraction out of range.
        """
        if not 0 <= fraction <= 1:
            raise InvalidParameter(f"fraction must be from 0 to 1, not {fraction!r}")
        t = float(self.temperatures[np.abs(self.temperatures - temperature).argmin()])
        for line in self.tie_lines:
            if (
                line.temperature == t
                and line.first_fraction <= fraction <= line.second_fraction
            ):
                return (line.first, line.second)
        k = int(np.abs(self.temperatures - t).argmin())
        x = int(np.abs(self.fractions - fraction).argmin())
        return (self.phases[self.stable[k, x]],)


def _values(result: Any) -> tuple[np.ndarray, np.ndarray]:
    """The temperatures, ascending, and free energies of a result or a pair."""
    if hasattr(result, "free_energy"):
        temperatures, free = result.temperatures, result.free_energy
    else:
        temperatures, free = result
    temperatures = np.asarray(temperatures, dtype=float)
    free = np.asarray(free, dtype=float)
    if temperatures.ndim != 1 or temperatures.shape != free.shape:
        raise InvalidParameter("each free energy needs one value per temperature")
    order = np.argsort(temperatures)
    return temperatures[order], free[order]


def _at_temperatures(result: Any, wanted: np.ndarray) -> np.ndarray:
    """The free energy at these temperatures, past its own along straight lines:
    a phase there is one that has already lost to another, as a crystal beyond
    the temperature to which it stayed a crystal."""
    temperatures, free = _values(result)
    out = np.interp(wanted, temperatures, free)
    if len(temperatures) > 1:
        ends = (
            (wanted < temperatures[0], temperatures[:_EXTENDING], free[:_EXTENDING]),
            (wanted > temperatures[-1], temperatures[-_EXTENDING:], free[-_EXTENDING:]),
        )
        for outside, t, g in ends:
            if outside.any():
                slope, level = np.polyfit(t, g, 1)
                out[outside] = slope * wanted[outside] + level
    return out


def _mixing(x: np.ndarray) -> np.ndarray:
    """x ln x + (1 - x) ln(1 - x), zero at the ends."""
    inside = np.clip(x, 1e-300, 1.0)
    other = np.clip(1 - x, 1e-300, 1.0)
    return x * np.log(inside) + (1 - x) * np.log(other)


def _lower_hull(x: np.ndarray, y: np.ndarray) -> list[int]:
    """The indices of the lower convex hull of points in order of x."""
    hull: list[int] = []
    for k in range(len(x)):
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            if (y[b] - y[a]) * (x[k] - x[a]) >= (y[k] - y[a]) * (x[b] - x[a]):
                hull.pop()
            else:
                break
        hull.append(k)
    return hull


def binary_phase_diagram(
    phases: Mapping[str, Mapping[float, Any]],
    temperatures: ArrayLike,
    *,
    elements: Sequence[str] = ("A", "B"),
    points: int = 1001,
    degree: int = 4,
) -> BinaryPhaseDiagram:
    """The phase diagram of two elements from free energies per atom, at constant
    pressure: at each temperature the stable phases at each composition lie on
    the lower convex hull of the free energies of all phases.

    Parameters
    ----------
    phases : mapping of str to mapping of float64
        For each phase, its free energies by the fraction of the second element:
        each a ``FreeEnergyResult``, or a pair of (T,) temperatures in K and
        (T,) free energies in eV per atom. A phase of one fraction is a compound
        of that composition. A phase of more is a solution between its lowest
        and its highest fractions: the free energy minus that of ideal mixing,
        at each temperature, is a polynomial of the fraction.
    temperatures : array_like
        (T,) the temperatures of the diagram in K, positive; beyond those of a
        free energy, it goes on along a straight line.
    elements : sequence of 2 str, optional
        The elements, for the labels. Default ("A", "B").
    points : int32, optional
        Number of fractions from 0 to 1. Default 1001.
    degree : int32, optional
        Largest degree of the polynomials. Default 4.

    Returns
    -------
    BinaryPhaseDiagram
        The stable phase at each temperature and fraction, and the tie lines.

    Raises
    ------
    InvalidParameter
        For no phases, a fraction out of range, no phase at some fraction, invalid
        temperatures, points or degree.
    """
    if not phases:
        raise InvalidParameter("a phase diagram needs phases")
    wanted = np.atleast_1d(np.asarray(temperatures, dtype=float))
    if wanted.ndim != 1 or not (wanted > 0).all():
        raise InvalidParameter("temperatures must be positive")
    if int(points) < 3:
        raise InvalidParameter(f"points must be at least 3, not {points!r}")
    if int(degree) < 0:
        raise InvalidParameter(f"degree must not be negative, not {degree!r}")
    if len(tuple(elements)) != 2:
        raise InvalidParameter("a binary phase diagram has two elements")
    grid = np.linspace(0.0, 1.0, int(points))
    names = tuple(phases)
    spacing = grid[1] - grid[0]
    kt = units.BOLTZMANN * wanted
    # (P, T, X) each solution at the fractions it reaches, and the compounds
    solutions = np.full((len(names), len(wanted), len(grid)), np.inf)
    compound_x, compound_free, compound_phase = [], [], []
    for p, name in enumerate(names):
        curves = phases[name]
        if not curves:
            raise InvalidParameter(f"phase {name!r} has no free energies")
        fractions = np.array(sorted(curves), dtype=float)
        if fractions.min() < 0 or fractions.max() > 1:
            raise InvalidParameter(
                f"the fractions of phase {name!r} must be from 0 to 1"
            )
        values = np.array(
            [_at_temperatures(curves[x], wanted) for x in sorted(curves)]
        )  # (F, T)
        if len(fractions) == 1:
            compound_x.append(fractions[0])
            compound_free.append(values[0])
            compound_phase.append(p)
            continue
        excess = values - _mixing(fractions)[:, None] * kt
        order = min(len(fractions) - 1, int(degree))
        coefficients = np.polynomial.polynomial.polyfit(fractions, excess, order)
        table = np.polynomial.polynomial.polyval(grid, coefficients)  # (T, X)
        table = table + kt[:, None] * _mixing(grid)
        low, high = fractions.min(), fractions.max()
        table[:, (grid < low - 1e-12) | (grid > high + 1e-12)] = np.inf
        solutions[p] = table
    # the points of every temperature in order of the fraction: grid, then compounds
    x_all = np.concatenate([grid, np.array(compound_x, dtype=float)])
    order = np.argsort(x_all, kind="stable")
    x_all = x_all[order]
    compound_table = np.array(compound_free, dtype=float).reshape(
        len(compound_x), len(wanted)
    )
    stable = np.full((len(wanted), len(grid)), -1, dtype=np.int64)
    tie_lines = []
    for k, temperature in enumerate(wanted):
        lowest = solutions[:, k, :]
        y_all = np.concatenate([lowest.min(axis=0), compound_table[:, k]])[order]
        labels = np.concatenate([lowest.argmin(axis=0), compound_phase]).astype(int)[
            order
        ]
        finite = np.flatnonzero(np.isfinite(y_all))
        if x_all[finite[0]] > 1e-12 or x_all[finite[-1]] < 1 - 1e-12:
            raise InvalidParameter(
                "no phase reaches the fractions 0 and 1: add one that does"
            )
        hull = finite[_lower_hull(x_all[finite], y_all[finite])]
        nearest = np.clip(np.searchsorted(x_all[hull], grid), 0, len(hull) - 1)
        on_hull = np.abs(x_all[hull][nearest] - grid) < 1e-12
        stable[k, on_hull] = labels[hull][nearest[on_hull]]
        for a, b in zip(hull[:-1], hull[1:]):
            first, second = names[labels[a]], names[labels[b]]
            if x_all[b] - x_all[a] > 1.5 * spacing or first != second:
                tie_lines.append(
                    TieLine(
                        float(temperature),
                        first,
                        float(x_all[a]),
                        second,
                        float(x_all[b]),
                    )
                )
                stable[k, (grid > x_all[a] + 1e-12) & (grid < x_all[b] - 1e-12)] = -1
    return BinaryPhaseDiagram(
        elements=tuple(str(e) for e in elements),
        phases=names,
        temperatures=wanted,
        fractions=grid,
        stable=stable,
        tie_lines=tuple(tie_lines),
    )


@dataclass(frozen=True)
class PhaseBoundary(Result):
    """Two phases of one element in equilibrium, along the pressure.

    Attributes
    ----------
    first, second : str
        The phases, the first stable below the temperature of the boundary and
        the second above it.
    pressures : numpy.ndarray
        (B,) float64, the pressures in GPa.
    temperatures : numpy.ndarray
        (B,) float64, the temperature of the boundary at each pressure in K.
    latent_heat : numpy.ndarray
        (B,) float64, the change of enthalpy from the first to the second in eV per atom.
    volume : numpy.ndarray
        (B,) float64, the change of volume from the first to the second in Å³ per atom.
    slope : numpy.ndarray
        (B,) float64 dT/dP of the boundary in K/GPa by the Clausius-Clapeyron relation,
        T times the change of volume over the latent heat.
    """

    first: str
    second: str
    pressures: np.ndarray
    temperatures: np.ndarray
    latent_heat: np.ndarray
    volume: np.ndarray
    slope: np.ndarray


@dataclass(frozen=True)
class PressureTemperatureDiagram(Result):
    """The stable phases of one element against pressure and temperature.

    Attributes
    ----------
    phases : tuple of str
        The names of the phases.
    pressures : numpy.ndarray
        (P,) float64, the pressures in GPa.
    temperatures : numpy.ndarray
        (T,) float64, the temperatures in K.
    stable : numpy.ndarray
        (P, T) int64, the index in ``phases`` of the phase with the lowest free energy
        of those known there, or -1 where none is.
    boundaries : tuple of PhaseBoundary
        Each pair of phases in equilibrium.
    """

    phases: tuple[str, ...]
    pressures: np.ndarray
    temperatures: np.ndarray
    stable: np.ndarray
    boundaries: tuple[PhaseBoundary, ...]

    def phase_at(self, pressure: float, temperature: float) -> str | None:
        """The stable phase at the nearest pressure and temperature of the diagram.

        Parameters
        ----------
        pressure : float64
            The pressure in GPa.
        temperature : float64
            The temperature in K.

        Returns
        -------
        str or None
            The phase, or None where no free energy is known.
        """
        i = np.abs(self.pressures - pressure).argmin()
        j = np.abs(self.temperatures - temperature).argmin()
        k = int(self.stable[i, j])
        return None if k < 0 else self.phases[k]


def _sorted_values(result: Any, name: str, temperature: float) -> float:
    """A property of a free energy at this temperature."""
    order = np.argsort(result.temperatures)
    temperatures = np.asarray(result.temperatures, dtype=float)[order]
    return float(
        np.interp(temperature, temperatures, np.asarray(getattr(result, name))[order])
    )


def pressure_temperature_diagram(
    phases: Mapping[str, Sequence[Any]], *, points: int = 1001
) -> PressureTemperatureDiagram:
    """The phase diagram of one element against pressure and temperature, from
    the free energies of its phases: at each pressure, the phase with the lowest
    free energy at each temperature.

    Parameters
    ----------
    phases : mapping
        For each name of a phase, its free energies, one per pressure, each a
        ``mdapy.simulation.FreeEnergyResult`` of a run at that pressure.
    points : int32, optional
        The temperatures of the diagram, from the lowest of the free energies to
        the highest. Default 1001.

    Returns
    -------
    PressureTemperatureDiagram
        The stable phases and the boundaries between them. A boundary is found
        along the temperature at each pressure.

    Raises
    ------
    InvalidParameter
        For fewer than two phases, too few points, a free energy without a
        pressure, or two free energies of one phase at one pressure.
    """
    names = tuple(phases)
    if len(names) < 2:
        raise InvalidParameter("a phase diagram needs at least two phases")
    if int(points) < 2:
        raise InvalidParameter(f"points must be at least 2, not {points!r}")
    known: dict[float, dict[int, Any]] = {}
    for k, name in enumerate(names):
        for result in phases[name]:
            if getattr(result, "pressure", None) is None:
                raise InvalidParameter(f"a free energy of {name} has no pressure")
            here = known.setdefault(float(result.pressure), {})
            if k in here:
                raise InvalidParameter(
                    f"two free energies of {name} at {result.pressure:g} GPa"
                )
            here[k] = result
    if not known:
        raise InvalidParameter("the phases have no free energies")
    pressures = np.array(sorted(known))
    every = [r.temperatures for here in known.values() for r in here.values()]
    grid = np.linspace(
        min(np.min(t) for t in every), max(np.max(t) for t in every), int(points)
    )
    stable = np.full((len(pressures), len(grid)), -1, dtype=np.int64)
    found: dict[tuple[int, int], list[tuple[float, float, float, float]]] = {}
    for i, pressure in enumerate(pressures):
        here = known[float(pressure)]
        values = np.full((len(names), len(grid)), np.inf)
        for k, result in here.items():
            temperatures, free = _values(result)
            inside = (grid >= temperatures[0]) & (grid <= temperatures[-1])
            values[k, inside] = np.interp(grid[inside], temperatures, free)
        lowest = values.argmin(axis=0)
        lowest[~np.isfinite(values.min(axis=0))] = -1
        stable[i] = lowest
        for j in np.flatnonzero(np.diff(lowest)):
            a, b = int(lowest[j]), int(lowest[j + 1])
            if a < 0 or b < 0:
                continue
            before = values[b, j] - values[a, j]
            after = values[b, j + 1] - values[a, j + 1]
            if not np.isfinite(before + after):
                continue  # one of them ends here: it lost nothing to the other
            where = grid[j] + (grid[j + 1] - grid[j]) * before / (before - after)
            heat, volume = (
                _sorted_values(here[b], name, where)
                - _sorted_values(here[a], name, where)
                for name in ("enthalpy", "volume")
            )
            found.setdefault((a, b), []).append(
                (float(pressure), float(where), heat, volume)
            )
    boundaries = []
    for (a, b), rows in found.items():
        at, temperatures, heat, volume = np.array(rows).T
        boundaries.append(
            PhaseBoundary(
                first=names[a],
                second=names[b],
                pressures=at,
                temperatures=temperatures,
                latent_heat=heat,
                volume=volume,
                slope=temperatures * volume / heat * units.GPA_TO_EV_PER_A3,
            )
        )
    return PressureTemperatureDiagram(
        phases=names,
        pressures=pressures,
        temperatures=grid,
        stable=stable,
        boundaries=tuple(boundaries),
    )
