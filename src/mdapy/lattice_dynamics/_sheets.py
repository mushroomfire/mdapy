# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Sheets: crystals periodic along the x-y plane only."""

import numpy as np

from mdapy.core.box import Box
from mdapy.core.system import System
from mdapy.errors import InvalidParameter

# empty space in Å between the images of a sheet in a periodic box, more than
# enough that symmetry searches see one sheet
_VACUUM = 20.0


def _is_sheet(box: Box, what: str) -> bool:
    """Whether the box is that of a sheet, periodic along its first two
    vectors, which lie in the x-y plane, and not along the third, along z. A
    box periodic in every direction is not; any other box is refused."""
    if box.pbc.all():
        return False
    matrix = box.matrix
    size = np.abs(matrix).max()
    if (
        not box.pbc[0]
        or not box.pbc[1]
        or np.abs(matrix[:2, 2]).max() > 1e-9 * size
        or np.abs(matrix[2, :2]).max() > 1e-9 * size
    ):
        raise InvalidParameter(
            f"{what} need a box periodic in every direction, or a sheet: a box "
            "periodic along two vectors in the x-y plane, not along a third along z"
        )
    return True


def _in_periodic_box(system: System) -> System:
    """A sheet in a box periodic in every direction, tall enough that spglib
    and seekpath see one sheet whatever the height of its own box."""
    box = system.box
    z = system["positions"][:, 2]
    height = max(box.matrix[2, 2], float(np.ptp(z)) + _VACUUM)
    matrix = box.matrix.copy()
    matrix[2, 2] = height
    origin = box.origin.copy()
    origin[2] = 0.5 * (z.min() + z.max()) - 0.5 * height
    return system.with_box(Box(matrix, origin=origin))
