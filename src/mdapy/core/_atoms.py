# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Some atoms of a system, given as a boolean mask or as indices."""

import numpy as np
from numpy.typing import ArrayLike

from mdapy.errors import InvalidParameter


def _mask(atoms: ArrayLike, n: int) -> np.ndarray:
    """(n,) the boolean mask of atoms given as a mask or as indices of a
    system of n atoms."""
    given = np.asarray(atoms)
    if given.dtype == bool:
        if given.shape != (n,):
            raise InvalidParameter(f"a mask of atoms must have {n} entries")
        return given
    index = given.reshape(-1)
    if index.size and (
        index.dtype.kind not in "iu" or index.min() < 0 or index.max() >= n
    ):
        raise InvalidParameter("atoms must be a mask or indices of the system")
    mask = np.zeros(n, dtype=bool)
    mask[index.astype(np.int64)] = True
    return mask
