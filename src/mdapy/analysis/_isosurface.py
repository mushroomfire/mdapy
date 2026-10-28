# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The surface where a field on a grid over the box crosses a level."""

from typing import Any

import numpy as np

from mdapy._core import analysis as _native
from mdapy.core.box import Box, _det
from mdapy.core.mesh import TriangleMesh
from mdapy.errors import InvalidParameter


def _grid(values: Any) -> np.ndarray:
    """The values of a grid over the box as a (na, nb, nc) array of finite
    numbers."""
    grid = np.asarray(values, dtype=np.float64)
    if grid.ndim != 3 or 0 in grid.shape:
        raise InvalidParameter(
            f"values must be a grid of (na, nb, nc) cells, not of shape {grid.shape}"
        )
    if not np.isfinite(grid).all():
        raise InvalidParameter("values must be finite; empty bins give NaN")
    return np.ascontiguousarray(grid)


def isosurface(values: Any, box: Box, level: float) -> TriangleMesh:
    """The surface where a field on a grid over the box crosses a level.

    Parameters
    ----------
    values : array_like
        (na, nb, nc) the field at the centers of the cells of a grid that
        divides the box into na, nb and nc cells along its vectors a, b and
        c, such as the values of a SpatialBinning along all three.
    box : Box
        The box the grid divides.
    level : float
        The value of the field on the surface.

    Returns
    -------
    TriangleMesh
        The triangles of the surface in the box, across its periodic
        boundaries, their normals pointing to lower values; it ends at the
        outermost centers along directions that are not periodic. Empty
        where the field does not cross the level.

    Raises
    ------
    InvalidParameter
        For values that are not a three-dimensional grid of finite numbers,
        or a level that is not finite.
    """
    grid = _grid(values)
    if not np.isfinite(level):
        raise InvalidParameter(f"level must be finite, not {level!r}")
    vertices, faces, shifts = _native.isosurface(
        grid,
        np.ascontiguousarray(box.pbc),
        float(level),
        _det(box.matrix) < 0,
    )
    return TriangleMesh(
        vertices.reshape(-1, 3) @ box.matrix + box.origin,
        faces.reshape(-1, 3),
        shifts.reshape(-1, 3, 3),
        box,
    )
