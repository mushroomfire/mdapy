# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The data model and its parts: the box of LAMMPS and the base of results.

Box, System, Topology, the threads and the index type are reached from the
top level, as ``mdapy.Box``.
"""

from mdapy.core.box import LammpsBox
from mdapy.core.result import Result

__all__ = ["LammpsBox", "Result"]
