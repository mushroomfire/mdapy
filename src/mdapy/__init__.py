# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""mdapy: analysis and modelling of atomistic simulations."""

__version__ = "2.0.0.dev0"

from mdapy import (
    analysis,
    build,
    calculators,
    elements,
    errors,
    identify,
    interop,
    lattice_dynamics,
    mlip,
    simulation,
    transform,
    units,
)
from mdapy.core.box import Box
from mdapy.core.indices import index_dtype
from mdapy.core.mesh import TriangleMesh
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
    "TriangleMesh",
    "__version__",
    "analysis",
    "build",
    "calculators",
    "identify",
    "interop",
    "elements",
    "errors",
    "lattice_dynamics",
    "mlip",
    "simulation",
    "transform",
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
