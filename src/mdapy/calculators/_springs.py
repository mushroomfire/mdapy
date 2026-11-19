# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Harmonic springs that tie each atom to a site: the Einstein crystal."""

from collections.abc import Iterable
from typing import Any

import numpy as np

from mdapy.calculators._base import Calculator, CalculatorResult
from mdapy.core.system import System
from mdapy.errors import InvalidParameter


class Springs(Calculator):
    """Each atom tied to its own site by a harmonic spring, E = k |x − x0|²
    / 2, the displacement taken to the nearest image of the site across the
    periodic directions: an Einstein crystal, whose free energy is known, as
    LAMMPS's fix spring/self and fix ti/spring tie atoms.

    Parameters
    ----------
    sites : numpy.ndarray
        (N, 3) float64, the site of each atom in Å, in the order of the
        atoms of the systems computed.
    constants : float64 or numpy.ndarray
        The spring constant in eV/Å², of every atom or (N,) float64 of each,
        positive.

    Attributes
    ----------
    sites : numpy.ndarray
        (N, 3) float64, the sites in Å.
    constants : numpy.ndarray
        (N,) float64, the spring constants in eV/Å².
    """

    implemented = frozenset({"energy", "atomic_energies", "forces"})
    cutoff = 0.0
    elements = None

    def __init__(self, sites: Any, constants: Any):
        sites = np.array(sites, dtype=float)
        if sites.ndim != 2 or sites.shape[1] != 3 or not np.isfinite(sites).all():
            raise InvalidParameter("sites must be (N, 3) finite numbers")
        constants = np.broadcast_to(
            np.asarray(constants, dtype=float), (len(sites),)
        ).copy()
        if not (np.isfinite(constants).all() and (constants > 0).all()):
            raise InvalidParameter("the spring constants must be positive")
        sites.flags.writeable = False
        constants.flags.writeable = False
        self.sites = sites
        self.constants = constants

    def compute(
        self,
        system: System,
        properties: Iterable[str] = ("energy", "forces"),
        *,
        neighbors: Any = None,
    ) -> CalculatorResult:
        wanted = self._check_properties(properties)
        positions = system["positions"]
        if positions.shape != self.sites.shape:
            raise InvalidParameter(
                f"the springs have {len(self.sites)} sites, the system {len(system)} atoms"
            )
        box = system.box
        shift = (positions - self.sites) @ np.linalg.inv(box.matrix)
        shift[:, box.pbc] -= np.round(shift[:, box.pbc])
        displacement = shift @ box.matrix
        energies = (
            0.5 * self.constants * np.einsum("ij,ij->i", displacement, displacement)
        )
        return CalculatorResult(
            energy=float(energies.sum()) if "energy" in wanted else None,
            atomic_energies=energies if "atomic_energies" in wanted else None,
            forces=-self.constants[:, None] * displacement
            if "forces" in wanted
            else None,
            volume=box.volume,
        )

    def __repr__(self) -> str:
        return f"Springs({len(self.sites)} sites)"
