# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The atoms on one side of a plane, or in a slab about it."""

import numpy as np
from numpy.typing import ArrayLike

from mdapy.core.system import System
from mdapy.errors import InvalidParameter

_AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}


# named after the Slice modifier of OVITO
def slice(
    system: System,
    normal: ArrayLike | str,
    distance: float = 0.0,
    *,
    width: float = 0.0,
    invert: bool = False,
) -> System:
    """The atoms on one side of a plane, or in a slab about it.

    Parameters
    ----------
    system : System
        The atoms.
    normal : array_like or str
        (3,) the normal of the plane in Cartesian coordinates, of any
        length, or an axis "x", "y" or "z".
    distance : float, optional
        The plane holds the points x with n . x = distance in Å, n the unit
        normal. Default 0.
    width : float, optional
        Keep the atoms closer to the plane than width / 2 in Å instead.
        Default 0: keep those on the side the normal points away from.
    invert : bool, optional
        Keep the other atoms. Default False.

    Returns
    -------
    System
        The atoms kept, in their order, in the same box.

    Raises
    ------
    InvalidParameter
        For a normal that is zero or not three numbers or an axis, or a
        negative width.
    """
    if isinstance(normal, str):
        if normal not in _AXES:
            raise InvalidParameter(
                f"normal must be x, y, z or three numbers, not {normal!r}"
            )
        normal = _AXES[normal]
    n = np.asarray(normal, dtype=float)
    if n.shape != (3,) or not np.isfinite(n).all() or not np.linalg.norm(n) > 0:
        raise InvalidParameter("normal must be three numbers, not all zero")
    width = float(width)
    if not width >= 0:
        raise InvalidParameter(f"width must be at least 0, not {width!r}")
    side = system["positions"] @ (n / np.linalg.norm(n)) - float(distance)
    keep = np.abs(side) <= width / 2 if width > 0 else side <= 0
    return system.gather(np.flatnonzero(keep != bool(invert)))
