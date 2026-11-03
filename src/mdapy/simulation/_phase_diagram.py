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
    temperature : float
        The temperature in K.
    first, second : str
        The phases, the first at the lower fraction.
    first_fraction, second_fraction : float
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
        (T,) the temperatures in K.
    fractions : numpy.ndarray
        (X,) the fractions of the second element.
    stable : numpy.ndarray
        (T, X) the index in ``phases`` of the one stable phase, or -1 where two
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
        temperature : float
            The temperature in K.
        fraction : float
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
    phases : mapping of str to mapping of float
        For each phase, its free energies by the fraction of the second element:
        each a :class:`FreeEnergyResult`, or a pair of (T,) temperatures in K and
        (T,) free energies in eV per atom. A phase of one fraction is a compound
        of that composition. A phase of more is a solution between its lowest
        and its highest fractions: the free energy minus that of ideal mixing,
        at each temperature, is a polynomial of the fraction.
    temperatures : array_like
        (T,) the temperatures of the diagram in K, positive; beyond those of a
        free energy, it goes on along a straight line.
    elements : sequence of 2 str, optional
        The elements, for the labels. Default ("A", "B").
    points : int, optional
        Number of fractions from 0 to 1. Default 1001.
    degree : int, optional
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
