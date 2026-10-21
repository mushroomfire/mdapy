# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Local temperature of atoms."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from mdapy import units
from mdapy.core.result import Result
from mdapy.core.system import System
from mdapy.elements import atomic_masses
from mdapy.errors import InvalidParameter, MissingProperty
from mdapy.neighbor import NeighborList
from mdapy.neighbor._neighbor_list import _given_or_built


@dataclass(frozen=True)
class TemperatureResult(Result):
    """The local temperature of each atom.

    Attributes
    ----------
    temperature : numpy.ndarray
        (N,) temperature in K; NaN for atoms without neighbors.
    """

    temperature: np.ndarray

    def columns(self) -> dict[str, np.ndarray]:
        return {"temperature": self.temperature}


class AtomicTemperature:
    """The temperature of each atom and its neighbors, from their velocities
    relative to their center of mass.

    Parameters
    ----------
    cutoff : float
        The neighbors closer than cutoff, in Å, form the group of an atom.
    """

    def __init__(self, cutoff: float):
        if not cutoff > 0:
            raise InvalidParameter(f"cutoff must be positive, not {cutoff!r}")
        self.cutoff = float(cutoff)

    def compute(
        self,
        system: System,
        *,
        neighbors: NeighborList | None = None,
        progress: Any = None,
        cancel: Any = None,
    ) -> TemperatureResult:
        """The temperature of each atom.

        Parameters
        ----------
        system : System
            The atoms, with ``velocities`` and either ``mass`` or ``element``.
        neighbors : NeighborList, optional
            A neighbor list of this system reaching at least ``cutoff``.

        Returns
        -------
        TemperatureResult
            The temperature of each atom.
        """
        table = system.table
        if "velocities" not in table.columns:
            raise MissingProperty("the temperature needs the velocities column")
        if "mass" in table.columns:
            mass = table["mass"].to_numpy()
        elif "element" in table.columns:
            mass = atomic_masses(table["element"].to_physical().to_numpy())
        else:
            raise MissingProperty("the temperature needs a mass or an element column")
        v = table["velocities"].to_numpy()
        nl = _given_or_built(system, self.cutoff, neighbors)
        nl = nl.within(self.cutoff) if nl.cutoff > self.cutoff else nl
        i, j = nl.i, nl.j
        other = j != i  # images of the atom itself are the same atom
        i, j = i[other], j[other]
        n = len(system)
        members = 1 + np.bincount(i, minlength=n)
        total = mass + np.bincount(i, weights=mass[j], minlength=n)
        momentum = mass[:, None] * v + np.stack(
            [np.bincount(i, weights=mass[j] * v[j, a], minlength=n) for a in range(3)],
            axis=1,
        )
        squares = mass * (v * v).sum(1) + np.bincount(
            i, weights=mass[j] * (v[j] * v[j]).sum(1), minlength=n
        )
        # twice the kinetic energy relative to the center of mass of the group
        twice = squares - (momentum * momentum).sum(1) / total
        with np.errstate(invalid="ignore", divide="ignore"):
            temperature = (
                twice
                * units.AMU_A2_PER_PS2_TO_EV
                / (3.0 * (members - 1) * units.BOLTZMANN)
            )
        temperature[members == 1] = np.nan
        return TemperatureResult(temperature)

    def __repr__(self) -> str:
        return f"AtomicTemperature(cutoff={self.cutoff!r})"
