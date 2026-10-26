# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Local temperature of atoms."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from mdapy import units
from mdapy._core import analysis as _native
from mdapy.core.result import Result
from mdapy.core.system import System
from mdapy.core.threads import _native_token
from mdapy.elements import atomic_masses
from mdapy.errors import InvalidParameter, MissingProperty
from mdapy.neighbor import NeighborList
from mdapy.neighbor._neighbor_list import _given_or_built, _view


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
        mass = np.ascontiguousarray(mass, dtype=np.float64)
        v = np.ascontiguousarray(v, dtype=np.float64)
        if neighbors is None:
            # the neighbors found on the grid, without a list
            box = system.box
            temperature = _native.atomic_temperature_search(
                np.ascontiguousarray(system["positions"]),
                np.ascontiguousarray(box.matrix),
                np.ascontiguousarray(box.origin),
                np.ascontiguousarray(box.pbc),
                self.cutoff,
                mass,
                v,
                units.AMU_A2_PER_PS2_TO_EV,
                units.BOLTZMANN,
                progress,
                _native_token(cancel),
            )
        else:
            nl = _given_or_built(system, self.cutoff, neighbors)
            with nl._reading():
                temperature = _native.atomic_temperature(
                    *_view(system, nl),
                    self.cutoff,
                    mass,
                    v,
                    units.AMU_A2_PER_PS2_TO_EV,
                    units.BOLTZMANN,
                    progress,
                    _native_token(cancel),
                )
        return TemperatureResult(temperature)

    def __repr__(self) -> str:
        return f"AtomicTemperature(cutoff={self.cutoff!r})"
