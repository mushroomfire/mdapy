# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""The integer type of atom indices in compiled results."""

import numpy as np

from mdapy import _core

__all__ = ["index_dtype"]

index_dtype: np.dtype = np.dtype(f"int{_core.index_bits}")
"""numpy.dtype of atom indices in results such as neighbor lists: int32, or
int64 when mdapy was built with ``-Ccmake.define.MDAPY_INDEX64=ON``."""
