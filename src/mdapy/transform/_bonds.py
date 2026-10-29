# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Bonds between near atoms, kept in the topology of a system."""

import numpy as np

from mdapy.core.system import System
from mdapy.core.topology import Topology
from mdapy.errors import InvalidParameter
from mdapy.neighbor import NeighborList


def create_bonds(system: System, cutoff: float | str) -> System:
    """Bonds between the atoms closer than a cutoff, as OVITO's create
    bonds.

    Parameters
    ----------
    system : System
        The atoms.
    cutoff : float or "covalent"
        Bonds between atoms closer than this in Å, also across periodic
        boundaries; or "covalent", between atoms closer than 1.2 times the
        sum of their covalent radii, as :func:`mdapy.visualization.covalent_bonds`
        finds them (needs an ``element`` column).

    Returns
    -------
    System
        The system with these bonds in its topology, each pair once, the
        first atom before the second; bonds it had are replaced, its angles,
        dihedrals and impropers kept.

    Raises
    ------
    InvalidParameter
        For a cutoff that is neither positive nor "covalent".
    """
    if isinstance(cutoff, str):
        if cutoff != "covalent":
            raise InvalidParameter(
                f'cutoff must be a distance or "covalent", not {cutoff!r}'
            )
        from mdapy.visualization import covalent_bonds

        neighbors = covalent_bonds(system)
    else:
        if not cutoff > 0:
            raise InvalidParameter(f"cutoff must be positive, not {cutoff!r}")
        neighbors = NeighborList.build(system, float(cutoff))
    first = neighbors.i < neighbors.j
    pairs = np.column_stack([neighbors.i[first], neighbors.j[first]]).astype(np.int64)
    old = system.topology
    topology = Topology(
        bonds=pairs,
        angles=None if old is None else old.angles,
        dihedrals=None if old is None else old.dihedrals,
        impropers=None if old is None else old.impropers,
    )
    return system.with_topology(topology)
