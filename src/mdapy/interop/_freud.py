# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

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
        The atoms.

    Returns
    -------
    box : freud.box.Box
        The box, with its first vector along x and its second in the xy
        plane, as freud keeps boxes, and the periodicity of the system.
    points : numpy.ndarray
        (N, 3) float32 positions in that box, centered on the origin as
        freud expects, moved into the box along periodic directions.
    """
    freud = _freud()
    box = system.box
    if np.linalg.det(box.matrix) < 0:
        raise InvalidParameter("freud needs a right-handed box")
    converted = freud.box.Box.from_matrix(box.matrix.T)  # vectors as columns
    converted.periodic = [bool(p) for p in box.pbc]
    fractional = box.to_fractional(system.positions)
    fractional = np.where(box.pbc, fractional - np.floor(fractional), fractional)
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
