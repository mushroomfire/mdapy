# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Conversion between System and ase.Atoms."""

import math
from typing import Any

import numpy as np

from mdapy import units
from mdapy.core.box import Box
from mdapy.core.system import System
from mdapy.errors import InvalidParameter

# ASE's unit of time, Å·sqrt(amu/eV), in ps.
_ASE_TIME_IN_PS = math.sqrt(units.AMU_A2_PER_PS2_TO_EV)
# Arrays of ase.Atoms that are not copied as columns of the same name.
_SPECIAL = {"numbers", "positions", "momenta", "masses", "initial_charges"}
# Columns of a System that ase.Atoms holds elsewhere.
_HELD = {"positions", "element", "mass", "velocities", "charge", "movable", "forces"}


def _ase():
    try:
        import ase
        import ase.constraints
        from ase.calculators.singlepoint import SinglePointCalculator
    except ImportError:  # pragma: no cover - ASE is installed for the tests
        raise ImportError("converting to and from ase.Atoms needs ASE") from None
    return ase, ase.constraints, SinglePointCalculator


def to_ase(system: System) -> Any:
    """The system as an ``ase.Atoms``.

    Parameters
    ----------
    system : System
        The atoms. Elements, masses, velocities, charges and ``movable``
        become the corresponding properties of ``ase.Atoms``; other numeric
        columns become arrays of the same name; the box origin becomes the
        cell displacement; ``info`` is copied, and its ``energy`` and
        ``stress`` or ``virial``, with a ``forces`` column, become a
        single-point calculator.

    Returns
    -------
    ase.Atoms
        A new object; nothing is shared with the system.
    """
    ase, constraints, SinglePointCalculator = _ase()
    table = system.table
    symbols = (
        table["element"].cast(str).to_list()
        if "element" in table.columns
        else ["X"] * len(system)
    )
    atoms = ase.Atoms(
        symbols=symbols,
        positions=system["positions"],
        cell=system.box.matrix,
        pbc=system.box.pbc,
        celldisp=system.box.origin,
    )
    if "mass" in table.columns:
        atoms.set_masses(table["mass"].to_numpy())
    if "velocities" in table.columns:
        atoms.set_velocities(table["velocities"].to_numpy() * _ASE_TIME_IN_PS)
    if "charge" in table.columns:
        atoms.set_initial_charges(table["charge"].to_numpy())
    if "movable" in table.columns:
        movable = table["movable"].to_numpy()
        fixed = ~movable.any(axis=1)
        partly = ~fixed & ~movable.all(axis=1)
        found = []
        if fixed.any():
            found.append(constraints.FixAtoms(indices=np.flatnonzero(fixed)))
        for i in np.flatnonzero(partly):
            found.append(constraints.FixCartesian(int(i), mask=~movable[i]))
        atoms.set_constraint(found)
    for name in table.columns:
        if name not in _HELD:
            values = table[name].to_numpy()
            if values.dtype.kind in "biuf":  # numbers and fixed-size arrays of them
                atoms.set_array(name, values.copy())
    info = dict(system.info)
    results = {}
    if "energy" in info:
        results["energy"] = float(info.pop("energy"))
    if "forces" in table.columns:
        results["forces"] = table["forces"].to_numpy().copy()
    if "stress" in info:
        results["stress"] = np.asarray(info.pop("stress")) * units.GPA_TO_EV_PER_A3
        info.pop("virial", None)
    elif "virial" in info:
        results["stress"] = -np.asarray(info.pop("virial")) / system.box.volume
    if "stress" in results:
        s = results["stress"]
        results["stress"] = np.array(
            [s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]]
        )
    atoms.info.update(info)
    if results:
        atoms.calc = SinglePointCalculator(atoms, **results)
    return atoms


def from_ase(atoms: Any) -> System:
    """An ``ase.Atoms`` as a System.

    Parameters
    ----------
    atoms : ase.Atoms
        The atoms. Masses set on them, momenta, initial charges and the
        constraints ``FixAtoms`` and ``FixCartesian`` become the columns
        ``mass``, ``velocities``, ``charge`` and ``movable``; other arrays
        become columns of the same name; the cell displacement becomes the
        box origin; ``info`` is copied, and the energy, forces and stress
        of an attached calculator become ``energy``, a ``forces`` column and
        ``stress`` in GPa.

    Returns
    -------
    System
        A new system; nothing is shared with the atoms.

    Raises
    ------
    InvalidParameter
        If the atoms have constraints other than ``FixAtoms`` and
        ``FixCartesian``.
    """
    _, constraints, _ = _ase()
    n = len(atoms)
    columns: dict[str, Any] = {}
    if atoms.has("masses"):
        columns["mass"] = atoms.get_masses()
    if atoms.has("momenta"):
        columns["velocities"] = atoms.get_velocities() / _ASE_TIME_IN_PS
    if atoms.has("initial_charges"):
        columns["charge"] = atoms.get_initial_charges()
    if atoms.constraints:
        movable = np.ones((n, 3), dtype=bool)
        for constraint in atoms.constraints:
            if isinstance(constraint, constraints.FixAtoms):
                movable[constraint.index] = False
            elif isinstance(constraint, constraints.FixCartesian):
                movable[constraint.index] &= ~np.asarray(constraint.mask, dtype=bool)
            else:
                raise InvalidParameter(
                    f"the constraint {type(constraint).__name__} has no equivalent in mdapy"
                )
        columns["movable"] = movable
    for name, values in atoms.arrays.items():
        if name not in _SPECIAL:
            columns[name] = np.array(values, copy=True)
    info = dict(atoms.info)
    calculator = atoms.calc
    if calculator is not None:
        results = calculator.results
        if "energy" in results:
            info["energy"] = float(results["energy"])
        if "forces" in results:
            columns["forces"] = np.array(results["forces"], copy=True)
        if "stress" in results:
            s = np.asarray(results["stress"], dtype=np.float64)
            if s.shape == (6,):
                xx, yy, zz, yz, xz, xy = s
                s = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
            info["stress"] = s * units.EV_PER_A3_TO_GPA
    box = Box(np.array(atoms.cell), origin=atoms.get_celldisp().ravel(), pbc=atoms.pbc)
    system = System(atoms.positions.copy(), box, elements=atoms.get_chemical_symbols())
    if columns:
        system = system.with_columns(**columns)
    return system.with_info(**info) if info else system
