# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Meshes of triangles."""

import numpy as np
from numpy.typing import ArrayLike

from mdapy.core.box import Box
from mdapy.errors import InvalidParameter

__all__ = ["TriangleMesh"]


def _read_only(a: np.ndarray) -> np.ndarray:
    a.flags.writeable = False
    return a


class TriangleMesh:
    """A mesh of triangles, which may cross the periodic boundaries of a
    box.

    Parameters
    ----------
    vertices : array_like
        (V, 3) positions of the vertices in Å.
    faces : array_like
        (F, 3) vertices of each triangle, counterclockwise seen from the side
        its normal points to.
    shifts : array_like, optional
        (F, 3, 3) periodic image of each corner: corner k of triangle f is at
        ``vertices[faces[f, k]] + shifts[f, k] @ box.matrix``. Default all in
        the box.
    box : Box, optional
        The box of the periodic images; needed with shifts.

    Raises
    ------
    InvalidParameter
        For arrays of the wrong shape, vertices that are not finite, faces
        of vertices that do not exist, or shifts without a box.
    """

    __slots__ = ("__vertices", "__faces", "__shifts", "__box")

    def __init__(
        self,
        vertices: ArrayLike,
        faces: ArrayLike,
        shifts: ArrayLike | None = None,
        box: Box | None = None,
    ) -> None:
        v = np.array(vertices, dtype=np.float64).reshape(-1, 3)
        if not np.isfinite(v).all():
            raise InvalidParameter("the vertices must be finite")
        f = np.array(faces, dtype=np.int64)
        if f.size == 0:
            f = f.reshape(0, 3)
        if f.ndim != 2 or f.shape[1] != 3:
            raise InvalidParameter(f"faces must have shape (F, 3), not {f.shape}")
        if len(f) and (f.min() < 0 or f.max() >= len(v)):
            raise InvalidParameter("a face has a vertex that does not exist")
        if box is not None and not isinstance(box, Box):
            raise InvalidParameter(f"box must be a Box, not {type(box).__name__}")
        s = None
        if shifts is not None:
            if box is None:
                raise InvalidParameter("shifts need the box of the periodic images")
            s = np.array(shifts, dtype=np.int64)
            if s.shape != (len(f), 3, 3):
                raise InvalidParameter(
                    f"shifts must have shape {(len(f), 3, 3)}, not {s.shape}"
                )
            s = s.astype(np.int8)
        for name, value in (("vertices", v), ("faces", f), ("shifts", s)):
            if value is not None:
                _read_only(value)
            object.__setattr__(self, f"_TriangleMesh__{name}", value)
        object.__setattr__(self, "_TriangleMesh__box", box)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("a TriangleMesh cannot be changed; make a new one")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("a TriangleMesh cannot be changed")

    def __copy__(self) -> "TriangleMesh":
        return self

    def __deepcopy__(self, memo: dict) -> "TriangleMesh":
        return self

    def __reduce__(self) -> tuple:
        # pickled as the arguments that make it again
        return (
            TriangleMesh,
            (self.__vertices, self.__faces, self.__shifts, self.__box),
        )

    @property
    def vertices(self) -> np.ndarray:
        """(V, 3) positions of the vertices in Å."""
        return self.__vertices

    @property
    def faces(self) -> np.ndarray:
        """(F, 3) int64 vertices of each triangle."""
        return self.__faces

    @property
    def shifts(self) -> np.ndarray | None:
        """(F, 3, 3) int8 periodic image of each corner, or None."""
        return self.__shifts

    @property
    def box(self) -> Box | None:
        """The box of the periodic images, or None."""
        return self.__box

    def __len__(self) -> int:
        return len(self.__faces)

    def corners(self) -> np.ndarray:
        """The corners of each triangle, each triangle whole.

        Returns
        -------
        numpy.ndarray
            (F, 3, 3) positions in Å.
        """
        corners = self.__vertices[self.__faces]
        if self.__shifts is not None:
            corners = corners + self.__shifts @ self.__box.matrix
        return corners

    @property
    def areas(self) -> np.ndarray:
        """(F,) area of each triangle in Å²."""
        return 0.5 * np.linalg.norm(self.__cross(), axis=1)

    @property
    def area(self) -> float:
        """Total area of the triangles in Å²."""
        return float(self.areas.sum())

    @property
    def normals(self) -> np.ndarray:
        """(F, 3) unit normal of each triangle; zero for a triangle without
        area."""
        cross = self.__cross()
        length = np.linalg.norm(cross, axis=1, keepdims=True)
        return np.divide(cross, length, out=np.zeros_like(cross), where=length > 0)

    def __cross(self) -> np.ndarray:
        c = self.corners()
        return np.cross(c[:, 1] - c[:, 0], c[:, 2] - c[:, 0])

    def pieces(self) -> tuple[np.ndarray, np.ndarray]:
        """The triangles inside the box: each moved by whole periods so that
        its center lies in the box, then cut where it crosses a periodic
        face and the part outside moved in through the opposite face.
        Without a box, the triangles as they are.

        Returns
        -------
        corners : numpy.ndarray
            (T, 3, 3) the corners of each piece in Å, counterclockwise as
            its triangle.
        triangle : numpy.ndarray
            (T,) int64 the triangle of each piece.
        """
        corners = self.corners()
        triangle = np.arange(len(corners))
        box = self.__box
        if box is None or not box.pbc.any() or not len(corners):
            return corners, triangle
        s = (corners - box.origin) @ box.inverse
        s -= np.where(box.pbc, np.floor(s.mean(axis=1)), 0.0)[:, None, :]
        for axis in np.flatnonzero(box.pbc):
            for plane, move in ((0.0, 1.0), (1.0, -1.0)):
                s, triangle = _cut(s, triangle, axis, plane, move)
        return s @ box.matrix + box.origin, triangle

    def __repr__(self) -> str:
        periodic = "" if self.__box is None else ", periodic"
        return (
            f"TriangleMesh({len(self.__vertices)} vertices, "
            f"{len(self.__faces)} triangles{periodic})"
        )


def _cut(
    s: np.ndarray, triangle: np.ndarray, axis: int, plane: float, move: float
) -> tuple[np.ndarray, np.ndarray]:
    """Cuts triangles in fractional coordinates s at s[axis] = plane and
    moves the parts beyond it, outside the box, by move along axis."""
    beyond = (s[:, :, axis] - plane) * move  # negative outside
    out, inside = beyond < 0, beyond > 0
    crossing = out.any(axis=1) & inside.any(axis=1)
    whole = s[~crossing]
    whole[out[~crossing].any(axis=1), :, axis] += move
    if not crossing.any():
        return whole, triangle[~crossing]
    cut, h, side = s[crossing], beyond[crossing], out[crossing]
    # the corner alone on its side first, which keeps the turn
    lonely = (side != np.roll(side, -1, axis=1)) & (side != np.roll(side, 1, axis=1))
    order = (np.argmax(lonely, axis=1)[:, None] + np.arange(3)) % 3
    rows = np.arange(len(cut))[:, None]
    cut, h, side = cut[rows, order], h[rows, order], side[rows, order]
    a, b, c = cut[:, 0], cut[:, 1], cut[:, 2]
    p = a + (b - a) * (h[:, 0] / (h[:, 0] - h[:, 1]))[:, None]
    q = a + (c - a) * (h[:, 0] / (h[:, 0] - h[:, 2]))[:, None]
    lone = np.stack([a, p, q], axis=1)
    rest = np.concatenate([np.stack([p, b, c], axis=1), np.stack([p, c, q], axis=1)])
    lone[side[:, 0], :, axis] += move
    rest[np.concatenate([~side[:, 0], ~side[:, 0]]), :, axis] += move
    owner = triangle[crossing]
    return (
        np.concatenate([whole, lone, rest]),
        np.concatenate([triangle[~crossing], owner, owner, owner]),
    )
