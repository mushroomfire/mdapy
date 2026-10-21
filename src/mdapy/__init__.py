# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""mdapy: analysis and modelling of atomistic simulations."""

__version__ = "2.0.0.dev0"

from mdapy import errors, units
from mdapy.core.threads import (
    CancelToken,
    get_num_threads,
    num_threads,
    set_num_threads,
)

__all__ = [
    "__version__",
    "errors",
    "units",
    "CancelToken",
    "get_num_threads",
    "set_num_threads",
    "num_threads",
]
