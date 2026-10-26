# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Conversion between System and the structures of pymatgen."""

from typing import Any

import numpy as np

from mdapy.core.box import Box
from mdapy.core.system import System
from mdapy.errors import InvalidParameter, MissingProperty

# pymatgen keeps velocities in Å/fs, as POSCAR files do.
_PS_PER_FS = 1e-3
# Columns of a System that a pymatgen structure holds elsewhere.
_HELD = {"positions", "element", "velocities", "movable"}


def _pymatgen():
    try:
        from pymatgen.core import Lattice, Molecule, Structure
    except ImportError:  # pragma: no cover - pymatgen is installed for the tests
        raise ImportError("converting to and from pymatgen needs pymatgen") from None
    return Lattice, Molecule, Structure


def to_pymatgen(system: System) -> Any:
    """The system as a pymatgen ``Structure``.

    Parameters
    ----------
    system : System
        The atoms, with an ``element`` column. The box becomes the lattice
        with its periodicity, and the positions are taken from the origin
        of the box; ``velocities`` (in Å/fs, as pymatgen keeps them) and
        ``movable`` (as ``selective_dynamics``) and other numeric columns
        become site properties; ``info`` becomes the properties.

    Returns
    -------
    pymatgen.core.Structure
        A new structure; nothing is shared with the system.

    Raises
    ------
    MissingProperty
        If the system has no ``element`` column.
    """
    Lattice, _, Structure = _pymatgen()
    table = system.table
    if "element" not in table.columns:
        raise MissingProperty("a pymatgen structure needs the element column")
    box = system.box
    sites: dict[str, Any] = {}
    if "velocities" in table.columns:
        sites["velocities"] = table["velocities"].to_numpy() * _PS_PER_FS
    if "movable" in table.columns:
        sites["selective_dynamics"] = table["movable"].to_numpy().copy()
    for name in table.columns:
        if name not in _HELD:
            values = table[name].to_numpy()
            if values.dtype.kind in "biuf":
                sites[name] = values.copy()
    return Structure(
        Lattice(box.matrix, pbc=tuple(bool(p) for p in box.pbc)),
        table["element"].cast(str).to_list(),
        system["positions"] - box.origin,
        coords_are_cartesian=True,
        site_properties={k: list(v) for k, v in sites.items()},
        properties=dict(system.info),
    )


def from_pymatgen(structure: Any) -> System:
    """A pymatgen ``Structure`` or ``Molecule`` as a System.

    Parameters
    ----------
    structure : pymatgen.core.Structure or pymatgen.core.Molecule
        The atoms, each site of one element. The lattice of a structure
        becomes the box, with its periodicity; a molecule gets a box that
        encloses it, not periodic. The site properties ``velocities``
        (Å/fs) and ``selective_dynamics`` become ``velocities`` (Å/ps) and
        ``movable``, other numeric site properties columns of the same name;
        the properties become ``info``.

    Returns
    -------
    System
        A new system; nothing is shared with the structure.

    Raises
    ------
    InvalidParameter
        If a site is disordered, with more than one element.
    """
    _, _, Structure = _pymatgen()
    if not structure.is_ordered:
        raise InvalidParameter("a site of the structure holds more than one element")
    box = None
    if isinstance(structure, Structure):
        lattice = structure.lattice
        box = Box(lattice.matrix, pbc=lattice.pbc)
    elements = [site.specie.symbol for site in structure]
    columns: dict[str, Any] = {}
    for name, values in structure.site_properties.items():
        array = np.array(values)
        if name == "velocities":
            columns["velocities"] = array.astype(np.float64) / _PS_PER_FS
        elif name == "selective_dynamics":
            columns["movable"] = array.astype(bool)
        elif array.dtype.kind in "biuf":
            columns[name] = array
    system = System(
        np.array(structure.cart_coords, dtype=np.float64),
        box,
        elements=elements,
        info=dict(structure.properties),
    )
    return system.with_columns(**columns) if columns else system
