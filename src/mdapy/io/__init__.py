# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Reading and writing files.

The functions are also available at the top level, as ``mdapy.read`` and so
on.
"""

# Importing the format modules registers the formats.
from mdapy.io import (  # noqa: F401
    _cif,
    _extxyz,
    _lammps_data,
    _lammps_dump,
    _native,
    _outcar,
    _pdb,
    _poscar,
)
from mdapy.io._registry import detect_format, formats, read, write
from mdapy.io._trajectory import Mapped, Trajectory, read_trajectory

__all__ = [
    "read",
    "write",
    "read_trajectory",
    "Trajectory",
    "Mapped",
    "formats",
    "detect_format",
]
