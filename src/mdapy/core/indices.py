# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The integer type of atom indices in compiled results."""

import numpy as np

from mdapy import _core

__all__ = ["index_dtype"]

index_dtype: np.dtype = np.dtype(f"int{_core.index_bits}")
"""numpy.dtype of atom indices in results such as neighbor lists: int32, or
int64 when mdapy was built with ``-Ccmake.define.MDAPY_INDEX64=ON``."""
