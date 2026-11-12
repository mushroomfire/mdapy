# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""mdapy: analysis and modelling of atomistic simulations."""

__version__ = "2.1.0rc1"

from mdapy import (
    analysis,
    build,
    calculators,
    elements,
    errors,
    fitting,
    identify,
    interop,
    io,
    lattice_dynamics,
    mlip,
    neighbor,
    simulation,
    transform,
    units,
    visualization,
)
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
from mdapy.core.versions import show_versions
from mdapy.neighbor._neighbor_list import NeighborList

__all__ = [
    "Box",
    "System",
    "Topology",
    "__version__",
    "analysis",
    "build",
    "calculators",
    "identify",
    "interop",
    "io",
    "elements",
    "errors",
    "fitting",
    "lattice_dynamics",
    "mlip",
    "neighbor",
    "simulation",
    "transform",
    "units",
    "visualization",
    "CancelToken",
    "get_num_threads",
    "set_num_threads",
    "show_versions",
    "num_threads",
    "index_dtype",
    "NeighborList",
]
