# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Core building blocks: the data model and thread control."""

from mdapy.core.threads import (
    CancelToken,
    get_num_threads,
    num_threads,
    set_num_threads,
)

__all__ = [
    "CancelToken",
    "get_num_threads",
    "set_num_threads",
    "num_threads",
]
