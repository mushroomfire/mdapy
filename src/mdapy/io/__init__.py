# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Reading and writing files.

The functions are also available at the top level, as ``mdapy.read`` and so
on.
"""

from mdapy.io import _extxyz, _lammps_data  # noqa: F401  (register the formats)
from mdapy.io._registry import detect_format, formats, read, write

__all__ = ["read", "write", "formats", "detect_format"]
