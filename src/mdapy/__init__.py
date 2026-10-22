# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""mdapy: analysis and modelling of atomistic simulations."""

__version__ = "2.0.0.dev0"

from mdapy import analysis, calculators, elements, errors, identify, simulation, units
from mdapy.core.box import Box
from mdapy.core.indices import index_dtype
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
    "analysis",
    "calculators",
    "identify",
    "elements",
    "errors",
    "simulation",
    "units",
    "CancelToken",
    "get_num_threads",
    "set_num_threads",
    "num_threads",
    "index_dtype",
    "read",
    "write",
    "read_trajectory",
    "Trajectory",
    "NeighborList",
    "formats",
    "detect_format",
]
