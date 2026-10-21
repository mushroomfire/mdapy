# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""mdapy: analysis and modelling of atomistic simulations."""

__version__ = "2.0.0.dev0"

from mdapy import elements, errors, units
from mdapy.core.box import Box
from mdapy.core.system import System
from mdapy.core.threads import (
    CancelToken,
    get_num_threads,
    num_threads,
    set_num_threads,
)
from mdapy.core.topology import Topology
from mdapy.io import Trajectory, detect_format, formats, read, read_trajectory, write
from mdapy.neighbor import NeighborList

__all__ = [
    "Box",
    "System",
    "Topology",
    "__version__",
    "elements",
    "errors",
    "units",
    "CancelToken",
    "get_num_threads",
    "set_num_threads",
    "num_threads",
    "read",
    "write",
    "read_trajectory",
    "Trajectory",
    "NeighborList",
    "formats",
    "detect_format",
]
