# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Velocities of a temperature for the atoms of a system, the start of a
molecular dynamics run."""

from typing import Any

import numpy as np

from mdapy import units
from mdapy.core.system import System
from mdapy.elements import atomic_masses
from mdapy.errors import InvalidParameter, MissingProperty


def thermal_velocities(
    system: System, temperature: float, *, seed: int, mask: Any = None
) -> np.ndarray:
    """Random velocities of a temperature: Gaussian, with no total momentum,
    and scaled so that their kinetic energy is that of the temperature
    exactly; those that ``MolecularDynamics.velocities`` gives in MDAPY with
    the same seed.

    Parameters
    ----------
    system : System
        The atoms, with a ``mass`` or an ``element`` column; a ``movable``
        column, (N, 3) bool, keeps them still along the directions where it
        is False, and then the total momentum is not removed.
    temperature : float64
        Temperature in K, not negative.
    seed : int32
        Seed of the random numbers, positive.
    mask : numpy.ndarray, optional
        (N,) bool, the atoms that get velocities, whose total momentum is
        removed; the others get none. Default every atom.

    Returns
    -------
    numpy.ndarray
        (N, 3) float64 velocities in Å/ps, for
        ``system.with_columns(velocities=...)``.

    Raises
    ------
    InvalidParameter
        For a negative temperature, a seed that is not positive, or a mask
        that is not one bool for each atom.
    MissingProperty
        If the system has neither a ``mass`` nor an ``element`` column.
    """
    if not temperature >= 0:
        raise InvalidParameter(f"temperature must not be negative, not {temperature!r}")
    if isinstance(seed, bool) or int(seed) != seed or int(seed) < 1:
        raise InvalidParameter(f"seed must be a positive integer, not {seed!r}")
    table = system.table
    n = len(system)
    if "mass" in table.columns:
        masses = table["mass"].to_numpy().astype(float)
    elif "element" in table.columns:
        masses = atomic_masses(table["element"].to_physical().to_numpy())
    else:
        raise MissingProperty("velocities need the masses: a mass or an element column")
    if mask is None:
        chosen = np.ones(n, dtype=bool)
    else:
        chosen = np.asarray(mask)
        if chosen.dtype != bool or chosen.shape != (n,):
            raise InvalidParameter(f"mask must be (N,) bool for {n} atoms")
    movable = (
        table["movable"].to_numpy().astype(bool) if "movable" in table.columns else None
    )
    if movable is not None and movable.all():
        movable = None
    rng = np.random.default_rng(int(seed))
    v = rng.normal(size=(n, 3)) / np.sqrt(masses)[:, None]
    v[~chosen] = 0.0
    if movable is None:
        m = masses[chosen]
        if m.size:
            v[chosen] -= (m @ v[chosen]) / m.sum()
        dof = 3 * int(chosen.sum()) - 3
    else:
        v *= movable
        dof = int(movable[chosen].sum())
    twice_kinetic = (
        float(np.einsum("i,ij,ij->", masses, v, v)) * units.AMU_A2_PER_PS2_TO_EV
    )
    now = twice_kinetic / (dof * units.BOLTZMANN) if dof > 0 else 0.0
    if temperature == 0.0 or not now > 0.0:
        return np.zeros((n, 3))
    return v * np.sqrt(temperature / now)
