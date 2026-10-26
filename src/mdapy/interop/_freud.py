# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Conversion between System and the boxes and points of freud."""

from typing import Any

import numpy as np

from mdapy.core.box import Box
from mdapy.core.system import System
from mdapy.errors import InvalidParameter


def _freud():
    try:
        import freud
    except ImportError:  # pragma: no cover - freud is installed for the tests
        raise ImportError("converting to and from freud needs freud") from None
    return freud


def to_freud(system: System) -> tuple[Any, np.ndarray]:
    """The system as a freud box and points, the input of freud's analyses.

    Parameters
    ----------
    system : System
        The atoms, in a right-handed box periodic in every direction.

    Returns
    -------
    box : freud.box.Box
        The box, with its first vector along x and its second in the xy
        plane, as freud keeps boxes.
    points : numpy.ndarray
        (N, 3) float32 positions in that box, centered on the origin as
        freud expects, moved into the box.

    Raises
    ------
    InvalidParameter
        If the box is not periodic in every direction, or is left-handed.
    """
    freud = _freud()
    box = system.box
    if not box.pbc.all():
        raise InvalidParameter(
            "freud finds neighbors only in a box periodic in every direction"
        )
    if np.linalg.det(box.matrix) < 0:
        raise InvalidParameter("freud needs a right-handed box")
    converted = freud.box.Box.from_matrix(box.matrix.T)  # vectors as columns
    fractional = box.to_fractional(system["positions"])
    fractional = fractional - np.floor(fractional)
    points = converted.make_absolute(fractional.astype(np.float32))
    return converted, np.asarray(points, dtype=np.float32)


def from_freud(box: Any, points: Any) -> System:
    """A freud box and points as a System.

    Parameters
    ----------
    box : freud.box.Box
        A three-dimensional box; its periodicity becomes that of the system.
    points : array_like
        (N, 3) positions in the box.

    Returns
    -------
    System
        The points in a box with freud's vectors, its origin at the corner
        of the box centered on the origin.

    Raises
    ------
    InvalidParameter
        If the box is two-dimensional.
    """
    if box.is2D:
        raise InvalidParameter("a System needs a three-dimensional box")
    matrix = np.asarray(box.to_matrix(), dtype=np.float64).T  # vectors as rows
    origin = -0.5 * matrix.sum(axis=0)
    converted = Box(matrix, origin=origin, pbc=np.asarray(box.periodic, dtype=bool))
    return System(np.asarray(points, dtype=np.float64), converted)
