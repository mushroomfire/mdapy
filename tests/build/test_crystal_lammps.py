# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Oriented crystals against LAMMPS lattice orient. Skipped where LAMMPS is
not installed."""

import numpy as np
import pytest

import mdapy
from mdapy.build import build_crystal

from lattices import same_crystal

lammps = pytest.importorskip("lammps")


@pytest.mark.parametrize(
    "structure, orientation",
    [
        ("fcc", [(1, -1, 0), (1, 1, -2), (1, 1, 1)]),
        ("bcc", [(1, 1, 1), (-1, 1, 0), (-1, -1, 2)]),
        ("diamond", [(1, 1, 0), (-1, 1, 0), (0, 0, 1)]),
        ("sc", [(2, 0, 0), (0, 3, 3), (0, -1, 1)]),
        ("fcc", [(1, 2, 3), (-2, 1, 0), (-3, -6, 5)]),
    ],
)
def test_oriented_crystals_as_lammps(structure, orientation):
    """The same atoms, up to a translation, as LAMMPS fills the same box."""
    a = 3.0
    system = build_crystal(structure, a, orientation=orientation, repeat=(2, 2, 2))
    lengths = system.box.matrix.diagonal()
    orient = " ".join(
        f"orient {axis} {' '.join(map(str, v))}" for axis, v in zip("xyz", orientation)
    )
    lmp = lammps.lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
    try:
        lmp.commands_string(
            f"""
units metal
atom_style atomic
boundary p p p
lattice {structure} {a} {orient}
region box block 0 {lengths[0]} 0 {lengths[1]} 0 {lengths[2]} units box
create_box 1 box
create_atoms 1 box
"""
        )
        n = lmp.extract_global("nlocal")
        positions = np.array(lmp.numpy.extract_atom("x")[:n], copy=True)
    finally:
        lmp.close()
    theirs = mdapy.System(positions, system.box, elements="X")
    assert same_crystal(system, theirs, tolerance=1e-8)
