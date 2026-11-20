# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Simulations: energy minimization and molecular dynamics, in MDAPY or in
LAMMPS, with the controls of a run in ``mdapy.simulation.controls``. The
properties that these give, such as elastic constants or free energies, are in
``mdapy.properties``."""

from typing import Any

from mdapy._moved import forward
from mdapy.simulation._dynamics import MolecularDynamics
from mdapy.simulation._minimize import MinimizeResult, minimize
from mdapy.simulation._velocities import thermal_velocities

__all__ = ["MinimizeResult", "MolecularDynamics", "minimize", "thermal_velocities"]


def __getattr__(name: str) -> Any:
    # the names that moved to mdapy.properties in 2.2, until 2.4
    return forward(__name__, name)
