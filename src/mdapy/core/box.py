# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The simulation box."""

import itertools
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike

from mdapy.errors import InvalidParameter

__all__ = ["Box", "LammpsBox"]

# Relative tolerance for treating a value as zero in shape checks.
_EPS = 1e-12


def _read_only(a: np.ndarray) -> np.ndarray:
    a.flags.writeable = False
    return a


class LammpsBox(NamedTuple):
    """A box in LAMMPS restricted triclinic form.

    Attributes
    ----------
    lo : numpy.ndarray
        (xlo, ylo, zlo) in Å.
    hi : numpy.ndarray
        (xhi, yhi, zhi) in Å.
    tilt : numpy.ndarray
        Tilt factors (xy, xz, yz) in Å.
    rotation : numpy.ndarray
        (3, 3) rotation matrix. A position or other vector ``v`` (row
        vector) of the original box becomes ``v @ rotation`` in the LAMMPS
        box. It is exactly the identity when the box already has this form.
    """

    lo: np.ndarray
    hi: np.ndarray
    tilt: np.ndarray
    rotation: np.ndarray


class Box:
    """Simulation box: three lattice vectors, an origin and periodicity.

    Parameters
    ----------
    matrix : float or array_like
        Lattice vectors in Å. A number gives a cube, three numbers an
        orthogonal box with these edge lengths, and a (3, 3) array a general
        box whose rows are the lattice vectors **a**, **b**, **c**.
    origin : array_like, optional
        Position of the corner of the box in Å. Default (0, 0, 0).
    pbc : bool or sequence of 3 bool, optional
        Whether the box is periodic along **a**, **b** and **c**. Default
        True.

    Raises
    ------
    InvalidParameter
        If the lattice vectors are not finite, have the wrong shape or are
        linearly dependent, or if origin or pbc are invalid.
    """

    __slots__ = ("__matrix", "__origin", "__pbc", "__inverse")

    def __init__(
        self,
        matrix: float | ArrayLike,
        origin: ArrayLike = (0.0, 0.0, 0.0),
        pbc: bool | ArrayLike = True,
    ) -> None:
        try:
            m = np.array(matrix, dtype=np.float64)
        except (TypeError, ValueError):
            raise InvalidParameter(f"box matrix must be numeric, got {matrix!r}")
        if m.ndim == 0:
            m = np.eye(3) * m
        elif m.shape == (3,):
            m = np.diag(m)
        if m.shape != (3, 3):
            raise InvalidParameter(
                f"box matrix must be a number, 3 lengths or a (3, 3) array, "
                f"got shape {m.shape}"
            )
        if not np.all(np.isfinite(m)):
            raise InvalidParameter("box matrix must be finite")
        scale = np.prod(np.linalg.norm(m, axis=1))
        det = np.linalg.det(m)
        if scale == 0.0 or abs(det) <= _EPS * scale:
            raise InvalidParameter("box lattice vectors are linearly dependent")

        try:
            o = np.array(origin, dtype=np.float64)
        except (TypeError, ValueError):
            raise InvalidParameter(f"box origin must be numeric, got {origin!r}")
        if o.shape != (3,) or not np.all(np.isfinite(o)):
            raise InvalidParameter(
                f"box origin must be 3 finite numbers, got {origin!r}"
            )

        p = np.array(pbc)
        if p.ndim == 0:
            p = np.repeat(p, 3)
        if (
            p.shape != (3,)
            or p.dtype.kind not in "bi"
            or not np.all((p == 0) | (p == 1))
        ):
            raise InvalidParameter(f"pbc must be a bool or 3 bools, got {pbc!r}")

        for name, value in (
            ("matrix", m),
            ("origin", o),
            ("pbc", p.astype(bool)),
            ("inverse", np.linalg.inv(m)),
        ):
            object.__setattr__(self, f"_Box__{name}", _read_only(value))

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("a Box cannot be changed; make a new one")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("a Box cannot be changed")

    def __copy__(self) -> "Box":
        return self

    def __deepcopy__(self, memo: dict) -> "Box":
        return self

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_lengths_angles(
        cls,
        a: float,
        b: float,
        c: float,
        alpha: float = 90.0,
        beta: float = 90.0,
        gamma: float = 90.0,
        origin: ArrayLike = (0.0, 0.0, 0.0),
        pbc: bool | ArrayLike = True,
    ) -> "Box":
        """Box from edge lengths and angles.

        **a** lies along x and **b** in the xy plane.

        Parameters
        ----------
        a, b, c : float
            Edge lengths in Å.
        alpha, beta, gamma : float, optional
            Angles in degrees between **b** and **c**, **a** and **c**, and
            **a** and **b**. Default 90.
        origin, pbc
            As for :class:`Box`.

        Raises
        ------
        InvalidParameter
            If the lengths are not positive or the angles do not form a box.
        """
        if min(a, b, c) <= 0:
            raise InvalidParameter(f"edge lengths must be positive, got {(a, b, c)}")
        ca, cb, cg = np.cos(np.radians([alpha, beta, gamma]))
        sg = np.sin(np.radians(gamma))
        if abs(sg) < _EPS:
            raise InvalidParameter("gamma must not be 0 or 180 degrees")
        cy = (ca - cb * cg) / sg
        cz2 = 1.0 - cb * cb - cy * cy
        if cz2 <= _EPS:
            raise InvalidParameter(
                f"angles {(alpha, beta, gamma)} do not form a three-dimensional box"
            )
        matrix = [
            [a, 0.0, 0.0],
            [b * cg, b * sg, 0.0],
            [c * cb, c * cy, c * np.sqrt(cz2)],
        ]
        return cls(matrix, origin=origin, pbc=pbc)

    @classmethod
    def enclosing(cls, positions: ArrayLike, padding: float = 1.0) -> "Box":
        """Non-periodic orthogonal box around a set of positions.

        Parameters
        ----------
        positions : array_like
            (N, 3) positions in Å. N may be 0.
        padding : float, optional
            Distance in Å between the outermost atoms and the box faces.
            Default 1.0; must be positive.

        Returns
        -------
        Box
            Box with ``pbc`` False in all directions.
        """
        if not padding > 0:
            raise InvalidParameter(f"padding must be positive, got {padding}")
        r = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        if len(r) == 0:
            lo = np.zeros(3)
            hi = np.zeros(3)
        else:
            lo, hi = r.min(axis=0), r.max(axis=0)
        return cls(hi - lo + 2.0 * padding, origin=lo - padding, pbc=False)

    @classmethod
    def from_lammps(
        cls,
        lo: ArrayLike,
        hi: ArrayLike,
        tilt: ArrayLike = (0.0, 0.0, 0.0),
        pbc: bool | ArrayLike = True,
    ) -> "Box":
        """Box from LAMMPS bounds and tilt factors.

        Parameters
        ----------
        lo, hi : array_like
            (xlo, ylo, zlo) and (xhi, yhi, zhi) in Å.
        tilt : array_like, optional
            Tilt factors (xy, xz, yz) in Å. Default zero.
        pbc
            As for :class:`Box`.
        """
        lo = np.asarray(lo, dtype=np.float64)
        length = np.asarray(hi, dtype=np.float64) - lo
        xy, xz, yz = np.asarray(tilt, dtype=np.float64)
        matrix = [[length[0], 0.0, 0.0], [xy, length[1], 0.0], [xz, yz, length[2]]]
        return cls(matrix, origin=lo, pbc=pbc)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def matrix(self) -> np.ndarray:
        """(3, 3) lattice vectors in Å, one per row. Read-only."""
        return self.__matrix

    @property
    def origin(self) -> np.ndarray:
        """(3,) corner of the box in Å. Read-only."""
        return self.__origin

    @property
    def pbc(self) -> np.ndarray:
        """(3,) bool, periodicity along **a**, **b**, **c**. Read-only."""
        return self.__pbc

    @property
    def inverse(self) -> np.ndarray:
        """(3, 3) inverse of :attr:`matrix`. Read-only."""
        return self.__inverse

    @property
    def volume(self) -> float:
        """Volume in Å³."""
        return float(abs(np.linalg.det(self.__matrix)))

    @property
    def lengths(self) -> np.ndarray:
        """(3,) lengths of **a**, **b**, **c** in Å."""
        return np.linalg.norm(self.__matrix, axis=1)

    @property
    def angles(self) -> np.ndarray:
        """(3,) angles alpha, beta, gamma in degrees.

        alpha is between **b** and **c**, beta between **a** and **c**, and
        gamma between **a** and **b**.
        """
        a, b, c = self.__matrix
        la, lb, lc = self.lengths

        def angle(u, v, lu, lv):
            return np.degrees(np.arccos(np.clip(np.dot(u, v) / (lu * lv), -1.0, 1.0)))

        return np.array([angle(b, c, lb, lc), angle(a, c, la, lc), angle(a, b, la, lb)])

    @property
    def thickness(self) -> np.ndarray:
        """(3,) distances in Å between opposite faces of the box.

        Entry i is the distance between the two faces that are not
        crossed by lattice vector i.
        """
        return 1.0 / np.linalg.norm(self.__inverse, axis=0)

    @property
    def is_orthogonal(self) -> bool:
        """Whether the lattice vectors lie along +x, +y and +z."""
        m = self.__matrix
        off = m - np.diag(np.diag(m))
        return bool(
            np.all(np.abs(off) <= _EPS * np.abs(m).max()) and np.all(np.diag(m) > 0)
        )

    # ------------------------------------------------------------------
    # Coordinates
    # ------------------------------------------------------------------

    def to_fractional(self, positions: ArrayLike) -> np.ndarray:
        """Fractional coordinates of Cartesian positions.

        Parameters
        ----------
        positions : array_like
            (..., 3) positions in Å.

        Returns
        -------
        numpy.ndarray
            (..., 3) coordinates in units of the lattice vectors, relative to
            :attr:`origin`.
        """
        return (
            np.asarray(positions, dtype=np.float64) - self.__origin
        ) @ self.__inverse

    def to_cartesian(self, fractional: ArrayLike) -> np.ndarray:
        """Cartesian positions of fractional coordinates.

        Parameters
        ----------
        fractional : array_like
            (..., 3) coordinates in units of the lattice vectors.

        Returns
        -------
        numpy.ndarray
            (..., 3) positions in Å.
        """
        return np.asarray(fractional, dtype=np.float64) @ self.__matrix + self.__origin

    def wrap(self, positions: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Map positions into the box along periodic directions.

        Parameters
        ----------
        positions : array_like
            (N, 3) positions in Å.

        Returns
        -------
        wrapped : numpy.ndarray
            (N, 3) positions in Å. Their fractional coordinates lie in
            [0, 1) along periodic directions; other directions are unchanged.
        image : numpy.ndarray
            (N, 3) int64 image flags, so that
            ``positions = wrapped + image @ matrix`` up to rounding. Zero along
            non-periodic directions.
        """
        r = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        image = np.floor(self.to_fractional(r)).astype(np.int64)
        image[:, ~self.__pbc] = 0
        # Subtracting whole lattice vectors keeps the input precision (11 - 10
        # is exactly 1), unlike converting back from fractional coordinates.
        wrapped = r - image @ self.__matrix
        # Rounding can leave a coordinate just outside [0, 1); fix it once.
        frac = self.to_fractional(wrapped)
        shift = np.zeros_like(image)
        shift[(frac < 0.0) & self.__pbc] = -1
        shift[(frac >= 1.0) & self.__pbc] = 1
        if shift.any():
            image += shift
            wrapped = wrapped - shift @ self.__matrix
        return wrapped, image

    def minimum_image(self, vectors: ArrayLike) -> np.ndarray:
        """Shortest periodic images of displacement vectors.

        Parameters
        ----------
        vectors : array_like
            (N, 3) displacement vectors in Å.

        Returns
        -------
        numpy.ndarray
            (N, 3) vectors, each the shortest among all vectors that differ
            from the input by a lattice vector along periodic directions.
            Exact for boxes whose tilt factors stay within half the
            corresponding edge length, as LAMMPS requires.
        """
        d = np.asarray(vectors, dtype=np.float64).reshape(-1, 3)
        n = np.round(d @ self.__inverse)
        n[:, ~self.__pbc] = 0.0
        base = d - n @ self.__matrix
        # Rounding is not enough in skewed boxes: also try the neighboring images.
        shifts = [
            s
            for s in itertools.product((-1, 0, 1), repeat=3)
            if all(si == 0 or p for si, p in zip(s, self.__pbc))
        ]
        candidates = (
            base[:, None, :] + np.array(shifts, dtype=np.float64) @ self.__matrix
        )
        best = np.argmin(np.einsum("nsk,nsk->ns", candidates, candidates), axis=1)
        return candidates[np.arange(len(d)), best]

    # ------------------------------------------------------------------
    # LAMMPS
    # ------------------------------------------------------------------

    def to_lammps(self) -> LammpsBox:
        """This box in LAMMPS restricted triclinic form.

        The box is rotated so that **a** lies along +x and **b** in the xy
        plane with a positive y component. A left-handed set of lattice
        vectors is first made right-handed by swapping **b** and **c**,
        which describes the same lattice.

        Returns
        -------
        LammpsBox
            Bounds, tilt factors and the rotation to apply to positions and
            other vectors.
        """
        m = self.__matrix
        if np.linalg.det(m) < 0:
            m = m[[0, 2, 1]]
        upper = np.array([m[0, 1], m[0, 2], m[1, 2]])
        if np.all(np.abs(upper) <= _EPS * np.abs(m).max()) and np.all(np.diag(m) > 0):
            rotation = np.eye(3)
            lower = np.tril(m)
        else:
            q, r = np.linalg.qr(m.T)
            signs = np.sign(np.diag(r))
            rotation = q * signs  # flip columns so the diagonal is positive
            lower = (signs[:, None] * r).T
            lower = np.tril(lower)
        lo = self.__origin @ rotation
        hi = lo + np.diag(lower)
        tilt = np.array([lower[1, 0], lower[2, 0], lower[2, 1]])
        return LammpsBox(lo=lo, hi=hi, tilt=tilt, rotation=rotation)

    # ------------------------------------------------------------------
    # Comparison and display
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Box):
            return NotImplemented
        return (
            np.array_equal(self.__matrix, other.__matrix)
            and np.array_equal(self.__origin, other.__origin)
            and np.array_equal(self.__pbc, other.__pbc)
        )

    def __hash__(self) -> int:
        # Adding 0.0 turns -0.0 into 0.0, so equal boxes hash equally.
        return hash(
            (
                (self.__matrix + 0.0).tobytes(),
                (self.__origin + 0.0).tobytes(),
                self.__pbc.tobytes(),
            )
        )

    def allclose(self, other: "Box", rtol: float = 1e-10, atol: float = 1e-10) -> bool:
        """Whether two boxes are equal within a tolerance.

        Parameters
        ----------
        other : Box
            The box to compare with.
        rtol, atol : float, optional
            Relative and absolute tolerances in Å, as in ``numpy.allclose``.
        """
        return (
            np.allclose(self.__matrix, other.__matrix, rtol=rtol, atol=atol)
            and np.allclose(self.__origin, other.__origin, rtol=rtol, atol=atol)
            and np.array_equal(self.__pbc, other.__pbc)
        )

    def __repr__(self) -> str:
        rows = ", ".join(
            np.array2string(r, precision=6, separator=", ") for r in self.__matrix
        )
        origin = np.array2string(self.__origin, precision=6, separator=", ")
        pbc = ", ".join(str(bool(p)) for p in self.__pbc)
        return f"Box(matrix=[{rows}], origin={origin}, pbc=({pbc}))"
