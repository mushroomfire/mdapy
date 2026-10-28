# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The faces of a grid over the box, to show its values on the box."""

from typing import Any

import numpy as np

from mdapy.analysis._isosurface import _grid
from mdapy.core.box import Box
from mdapy.core.mesh import TriangleMesh


def voxel_faces(values: Any, box: Box) -> tuple[TriangleMesh, np.ndarray]:
    """The faces of the box cut into the cells of a grid over it, each with
    the value of the cell inside it, to show the grid as colors on the box.

    Parameters
    ----------
    values : array_like
        (na, nb, nc) the values of the cells of a grid that divides the box
        into na, nb and nc cells along its vectors a, b and c, such as the
        values of a SpatialBinning along all three.
    box : Box
        The box the grid divides.

    Returns
    -------
    mesh : TriangleMesh
        Two triangles for each cell at each of the six faces of the box,
        their normals pointing out of a right-handed box.
    face_values : numpy.ndarray
        (F,) the value of the cell of each triangle, for colors such as
        those of ``colormap``.

    Raises
    ------
    InvalidParameter
        For values that are not a three-dimensional grid of finite numbers.
    """
    grid = _grid(values)
    n = grid.shape
    corners, found = [], []
    for e in range(3):
        p, q = (e + 1) % 3, (e + 2) % 3
        i, j = np.meshgrid(np.arange(n[p]), np.arange(n[q]), indexing="ij")
        i, j = i.ravel(), j.ravel()
        for side in (0, 1):
            quad = np.zeros((len(i), 4, 3))
            quad[:, :, e] = side
            quad[:, :, p] = (i[:, None] + [0, 1, 1, 0]) / n[p]
            quad[:, :, q] = (j[:, None] + [0, 0, 1, 1]) / n[q]
            # counterclockwise about the outward normal: +e on side 1
            order = [0, 1, 2, 0, 2, 3] if side else [0, 2, 1, 0, 3, 2]
            corners.append(quad[:, order].reshape(-1, 3, 3))
            cell = [None, None, None]
            cell[e], cell[p], cell[q] = side * (n[e] - 1), i, j
            found.append(np.repeat(grid[tuple(cell)], 2))
    triangles = np.concatenate(corners) @ box.matrix + box.origin
    count = len(triangles)
    mesh = TriangleMesh(triangles.reshape(-1, 3), np.arange(3 * count).reshape(-1, 3))
    return mesh, np.concatenate(found)
