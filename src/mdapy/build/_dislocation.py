# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Straight dislocations from isotropic elasticity, and edge dislocations
of two crystals."""

import numpy as np
from numpy.typing import ArrayLike

from mdapy.build._defects import _axis
from mdapy.core.box import Box
from mdapy.core.system import System
from mdapy.errors import InvalidParameter


def insert_dislocation(
    system: System,
    burgers: ArrayLike,
    *,
    line: int | str = 2,
    normal: int | str = 1,
    center: ArrayLike | None = None,
    poisson: float = 1 / 3,
) -> System:
    """The system with a straight dislocation along a box axis, its atoms
    displaced as isotropic elasticity predicts.

    Parameters
    ----------
    system : System
        The atoms.
    burgers : array_like
        (3,) Burgers vector in Å, in the plane of cut: its part along line is
        the screw part, the rest the edge part.
    line : int or str, optional
        The Cartesian axis along the dislocation line, 0, 1, 2 or x, y, z.
        Default z.
    normal : int or str, optional
        The Cartesian axis normal to the plane of cut, other than line.
        Default y.
    center : array_like, optional
        (3,) a point of the dislocation line in Å. Default the center of the
        box.
    poisson : float, optional
        Poisson's ratio, from -1 to below 0.5. Default 1/3.

    Returns
    -------
    System
        The same atoms, displaced; the box is unchanged.
    """
    a3, a2 = _axis(line), _axis(normal)
    if a2 == a3:
        raise InvalidParameter("normal must differ from line")
    a1 = 3 - a2 - a3
    b = np.asarray(burgers, dtype=float)
    if b.shape != (3,):
        raise InvalidParameter("burgers must have three components")
    if abs(b[a2]) > 1e-12 * max(np.linalg.norm(b), 1.0):
        raise InvalidParameter("burgers must lie in the plane of cut")
    if not -1 < poisson < 0.5:
        raise InvalidParameter(f"poisson must be from -1 to below 0.5, not {poisson!r}")
    box = system.box
    point = (
        box.origin + 0.5 * box.matrix.sum(axis=0)
        if center is None
        else np.asarray(center, dtype=float)
    )
    x = system.positions - point
    x1, x2 = x[:, a1], x[:, a2]
    if ((x1 == 0) & (x2 == 0)).any():
        raise InvalidParameter("an atom lies on the dislocation line")
    u = np.zeros_like(x)
    # screw part: the cut is the half plane x1 < 0
    u[:, a3] = b[a3] / (2 * np.pi) * np.arctan2(x2, x1)
    if b[a1] != 0:
        # edge part: the cut is the half plane x1 > 0, as atomsk makes it;
        # the atoms there below the plane first move by half the Burgers
        # vector, and those above it after the elastic displacement move back
        nu = float(poisson)
        half = 0.5 * abs(b[a1])
        side = np.sign(b[a1])
        start = x1 + np.where((x1 > 0) & (x2 * side <= 0), half, 0.0)
        r2 = start * start + x2 * x2
        if (r2 == 0).any():
            raise InvalidParameter("an atom lies on the dislocation line")
        with np.errstate(divide="ignore"):
            angle = np.arctan(x2 / start)
        end1 = start + b[a1] / (2 * np.pi) * (angle + start * x2 / (2 * (1 - nu) * r2))
        end2 = x2 - b[a1] / (2 * np.pi) * (
            (1 - 2 * nu) / (4 * (1 - nu)) * np.log(r2)
            + (start * start - x2 * x2) / (4 * (1 - nu) * r2)
        )
        end1 -= np.where((end1 > 0) & (end2 * side >= 0), half, 0.0)
        u[:, a1] = end1 - x1
        u[:, a2] = end2 - x2
    return system.with_positions(system.positions + u)


def edge_dislocation(cell: System, repeat: tuple[int, int, int]) -> System:
    """An edge dislocation between two crystals, with one more period along
    x below its line than above.

    Parameters
    ----------
    cell : System
        An orthogonal periodic cell whose period along x is the Burgers
        vector, such as fcc along [1-10], [11-2], [111].
    repeat : tuple of int
        (nx, ny, nz): above the line nx periods along x, below it nx + 1;
        ny along y, the line; nz along z in all, at least 2.

    Returns
    -------
    System
        The lower crystal compressed and the upper one stretched along x to
        nx + 1/2 periods, in a box periodic along x and y but not z; the
        line lies between them, at the middle of x.
    """
    matrix = cell.box.matrix
    if not np.allclose(matrix, np.diag(np.diag(matrix))) or not cell.box.pbc.all():
        raise InvalidParameter("cell must be an orthogonal periodic box")
    nx, ny, nz = (int(n) for n in repeat)
    if nx < 1 or ny < 1 or nz < 2:
        raise InvalidParameter("repeat needs nx >= 1, ny >= 1 and nz >= 2")
    length = np.diag(matrix)
    fractional = cell.box.to_fractional(cell.positions)
    fractional -= np.floor(fractional)
    names = (
        cell.table["element"].to_list()
        if "element" in cell.table.columns
        else ["X"] * len(cell)
    )
    width = (nx + 0.5) * length[0]
    lower_z = nz // 2
    blocks, elements = [], []
    for periods, z0, z1 in ((nx + 1, 0, lower_z), (nx, lower_z, nz)):
        grid = np.stack(
            np.meshgrid(
                np.arange(periods), np.arange(ny), np.arange(z0, z1), indexing="ij"
            ),
            -1,
        ).reshape(-1, 1, 3)
        points = ((grid + fractional) * length).reshape(-1, 3)
        points[:, 0] *= width / (periods * length[0])
        blocks.append(points)
        elements += names * len(grid)
    box = Box(np.diag([width, ny * length[1], nz * length[2]]), pbc=(True, True, False))
    return System(np.concatenate(blocks), box, elements=elements)
