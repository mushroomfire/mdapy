# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Cubic splines through values on a line and on a grid."""

import numpy as np
from numpy.typing import ArrayLike

from mdapy.errors import InvalidParameter

BOUNDARIES = ("not-a-knot", "natural", "clamped", "periodic")


def _grid(x: ArrayLike, name: str) -> np.ndarray:
    try:
        x = np.asarray(x, dtype=np.float64)
    except (TypeError, ValueError):
        raise InvalidParameter(f"{name} must be numbers") from None
    if x.ndim != 1 or len(x) < 2:
        raise InvalidParameter(f"{name} must be a line of two or more numbers")
    if not np.isfinite(x).all() or not (np.diff(x) > 0).all():
        raise InvalidParameter(f"{name} must be finite and strictly increasing")
    return x


def _tridiagonal(lower, middle, upper, right):
    """The solution of a tridiagonal system, by elimination without pivoting;
    lower[0] and upper[-1] are not used. right is (n, m)."""
    n = len(middle)
    c = np.zeros(n)
    d = np.zeros_like(right)
    c[0] = upper[0] / middle[0]
    d[0] = right[0] / middle[0]
    for i in range(1, n):
        denominator = middle[i] - lower[i] * c[i - 1]
        c[i] = upper[i] / denominator if i < n - 1 else 0.0
        d[i] = (right[i] - lower[i] * d[i - 1]) / denominator
    for i in range(n - 2, -1, -1):
        d[i] -= c[i] * d[i + 1]
    return d


def _cyclic(lower, middle, upper, right):
    """The solution of a tridiagonal system with corners lower[0] (row 0,
    last column) and upper[-1] (last row, column 0), by Sherman-Morrison."""
    n = len(middle)
    gamma = -middle[0]
    middle = middle.copy()
    middle[0] -= gamma
    middle[-1] -= lower[0] * upper[-1] / gamma
    y = _tridiagonal(lower, middle, upper, right)
    u = np.zeros((n, 1))
    u[0, 0], u[-1, 0] = gamma, upper[-1]
    z = _tridiagonal(lower, middle, upper, u)
    v_last = lower[0] / gamma
    factor = (y[0] + v_last * y[-1]) / (1.0 + z[0, 0] + v_last * z[-1, 0])
    return y - z * factor


def _slopes(x, y, boundary, slopes):
    """The first derivative at each point of the cubic spline through the
    points, for y of shape (n, m)."""
    n = len(x)
    h = np.diff(x)
    secant = np.diff(y, axis=0) / h[:, None]
    if n == 2 and boundary != "clamped":
        return np.repeat(secant, 2, axis=0)
    if n == 3 and boundary == "not-a-knot":  # one parabola through the three
        a = (secant[1] - secant[0]) / (x[2] - x[0])
        first = secant[0] - a * h[0]
        return np.stack([first, first + 2 * a * h[0], first + 2 * a * (x[2] - x[0])])
    lower, middle, upper = np.zeros(n), np.zeros(n), np.zeros(n)
    right = np.zeros_like(y)
    lower[1:-1] = h[1:]
    middle[1:-1] = 2 * (h[:-1] + h[1:])
    upper[1:-1] = h[:-1]
    right[1:-1] = 3 * (h[1:, None] * secant[:-1] + h[:-1, None] * secant[1:])
    if boundary == "periodic":
        # the last point is the first: unknowns s_0 ... s_(n-2)
        m = n - 1
        lower, middle, upper = np.zeros(m), np.zeros(m), np.zeros(m)
        right = np.zeros((m, y.shape[1]))
        for i in range(m):
            before, after = h[i - 1], h[i]
            lower[i], upper[i] = after, before
            middle[i] = 2 * (before + after)
            right[i] = 3 * (after * secant[i - 1] + before * secant[i])
        # row 0 couples to s_(m-1) through lower[0], row m-1 to s_0 through upper[-1]
        solved = _cyclic(lower, middle, upper, right)
        return np.vstack([solved, solved[:1]])
    if boundary == "natural":
        middle[0], upper[0] = 2.0, 1.0
        right[0] = 3 * secant[0]
        lower[-1], middle[-1] = 1.0, 2.0
        right[-1] = 3 * secant[-1]
    elif boundary == "clamped":
        middle[0], upper[0] = 1.0, 0.0
        right[0] = slopes[0]
        lower[-1], middle[-1] = 0.0, 1.0
        right[-1] = slopes[1]
    else:  # not-a-knot: the third derivative continuous at x_1 and x_(n-2)
        middle[0], upper[0] = h[1], h[0] + h[1]
        right[0] = (
            (h[0] + 2 * (h[0] + h[1])) * h[1] * secant[0] + h[0] ** 2 * secant[1]
        ) / (h[0] + h[1])
        lower[-1], middle[-1] = h[-1] + h[-2], h[-2]
        right[-1] = (
            h[-1] ** 2 * secant[-2] + (2 * (h[-2] + h[-1]) + h[-1]) * h[-2] * secant[-1]
        ) / (h[-2] + h[-1])
    return _tridiagonal(lower, middle, upper, right)


def _coefficients(x, y, s):
    """(4, n - 1, m) the coefficients of the cubic on each interval, in powers
    of the distance from its start."""
    h = np.diff(x)[:, None]
    secant = np.diff(y, axis=0) / h
    c2 = (3 * secant - 2 * s[:-1] - s[1:]) / h
    c3 = (s[:-1] + s[1:] - 2 * secant) / h**2
    return np.stack([y[:-1], s[:-1], c2, c3])


def _fitted(x, values, boundary, ends=None):
    """(4, n - 1, m) the coefficients of the spline through values (n, m)."""
    if boundary == "periodic" and not np.allclose(
        values[0], values[-1], rtol=1e-12, atol=1e-12 * np.abs(values).max()
    ):
        raise InvalidParameter("a periodic spline needs equal first and last values")
    return _coefficients(x, values, _slopes(x, values, boundary, ends))


def _evaluate(coefficients, x, at, derivative):
    """The piecewise cubic or its derivative at the points at, (p, m)."""
    interval = np.clip(np.searchsorted(x, at, side="right") - 1, 0, len(x) - 2)
    t = (at - x[interval])[:, None]
    c = coefficients[:, interval]  # (4, p, m)
    if derivative == 0:
        return c[0] + t * (c[1] + t * (c[2] + t * c[3]))
    if derivative == 1:
        return c[1] + t * (2 * c[2] + 3 * t * c[3])
    if derivative == 2:
        return 2 * c[2] + 6 * t * c[3]
    return np.broadcast_to(6 * c[3], c[3].shape).copy()


class Spline:
    """A cubic spline through values at points on a line: piecewise cubic,
    with continuous first and second derivatives, through every point.

    Parameters
    ----------
    x : array_like
        (n,) float64 points, strictly increasing, two or more.
    y : array_like
        (n,) float64 values at the points, or (n, m) for m curves through the
        same points.
    boundary : str, optional
        At the two ends: ``"not-a-knot"``, the third derivative continuous
        at the second and the second-last point; ``"natural"``, second
        derivatives zero; ``"clamped"``, first derivatives given by
        ``slopes``; ``"periodic"``, a periodic curve whose first and last
        values are equal. Default ``"not-a-knot"``.
    slopes : array_like, optional
        (2,) or (2, m) float64 first derivatives at the two ends, for
        ``"clamped"``.
    extrapolate : bool, optional
        Continue the end cubics beyond the points; otherwise values there are
        NaN, or for ``"periodic"`` repeat with the period. Default False.

    Attributes
    ----------
    x : numpy.ndarray
        (n,) float64 points.
    y : numpy.ndarray
        (n,) or (n, m) float64 values.
    boundary : str
        The condition at the ends.

    Raises
    ------
    InvalidParameter
        For points that are not strictly increasing, values of another
        length, an unknown boundary, slopes that are missing or of the
        wrong shape, or periodic values whose ends differ.
    """

    def __init__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        *,
        boundary: str = "not-a-knot",
        slopes: ArrayLike | None = None,
        extrapolate: bool = False,
    ):
        x = _grid(x, "x")
        try:
            y = np.asarray(y, dtype=np.float64)
        except (TypeError, ValueError):
            raise InvalidParameter("y must be numbers") from None
        if y.ndim not in (1, 2) or len(y) != len(x):
            raise InvalidParameter(f"y must have {len(x)} values, one at each point")
        if not np.isfinite(y).all():
            raise InvalidParameter("y must be finite")
        if boundary not in BOUNDARIES:
            raise InvalidParameter(
                f"boundary must be one of {BOUNDARIES}, not {boundary!r}"
            )
        values = y.reshape(len(x), -1)
        ends = None
        if boundary == "clamped":
            if slopes is None:
                raise InvalidParameter("a clamped spline needs slopes at its two ends")
            ends = np.asarray(slopes, dtype=np.float64).reshape(2, -1)
            if ends.shape[1] not in (1, values.shape[1]):
                raise InvalidParameter("slopes must be 2 values, or 2 for each curve")
            ends = np.broadcast_to(ends, (2, values.shape[1]))
        elif slopes is not None:
            raise InvalidParameter("slopes are for a clamped spline only")
        self.__coefficients = _fitted(x, values, boundary, ends)
        self.x = x
        self.y = y
        self.boundary = boundary
        self.__extrapolate = bool(extrapolate)
        self.__shape = y.shape[1:]
        for array in (self.x, self.y):
            array.flags.writeable = False

    def __repr__(self) -> str:
        return f"Spline({len(self.x)} points, {self.boundary})"

    def __call__(self, at: ArrayLike, derivative: int = 0) -> np.ndarray:
        """The spline or one of its derivatives at points.

        Parameters
        ----------
        at : array_like
            float64 points, of any shape.
        derivative : int32, optional
            0 for the values, 1 to 3 for the derivatives. Default 0.

        Returns
        -------
        numpy.ndarray
            float64 of the shape of ``at``, followed by (m,) for m curves.
        """
        if derivative not in (0, 1, 2, 3):
            raise InvalidParameter(
                f"derivative must be 0, 1, 2 or 3, not {derivative!r}"
            )
        at = np.asarray(at, dtype=np.float64)
        flat = at.reshape(-1)
        low, high = self.x[0], self.x[-1]
        where = flat
        if self.boundary == "periodic" and not self.__extrapolate:
            where = low + np.mod(flat - low, high - low)
        values = _evaluate(self.__coefficients, self.x, where, derivative)
        if not self.__extrapolate and self.boundary != "periodic":
            values[(flat < low) | (flat > high)] = np.nan
        return values.reshape(at.shape + self.__shape)

    def integral(self, start: float, end: float) -> np.ndarray:
        """The integral of the spline from start to end, both within the
        points unless the spline extrapolates.

        Parameters
        ----------
        start, end : float64
            The limits.

        Returns
        -------
        numpy.ndarray
            float64, of shape () or (m,) for m curves.
        """
        sign = 1.0
        if end < start:
            start, end, sign = end, start, -1.0
        if not self.__extrapolate and (start < self.x[0] or end > self.x[-1]):
            raise InvalidParameter("the limits must lie within the points")
        c = self.__coefficients
        h = np.diff(self.x)[:, None]
        whole = c[0] * h + c[1] * h**2 / 2 + c[2] * h**3 / 3 + c[3] * h**4 / 4
        total = np.concatenate(
            [np.zeros((1, whole.shape[1])), np.cumsum(whole, axis=0)]
        )

        def antiderivative(point):
            i = int(
                np.clip(
                    np.searchsorted(self.x, point, side="right") - 1, 0, len(self.x) - 2
                )
            )
            t = point - self.x[i]
            k = c[:, i]
            return (
                total[i]
                + k[0] * t
                + k[1] * t**2 / 2
                + k[2] * t**3 / 3
                + k[3] * t**4 / 4
            )

        result = sign * (antiderivative(end) - antiderivative(start))
        return result.reshape(self.__shape)


class GridSpline:
    """A cubic spline through values on a rectangular grid: along each axis
    the cubic spline of ``Spline``, so that the surface goes through every
    value with continuous derivatives.

    Parameters
    ----------
    x : array_like
        (nx,) float64 points along the first axis, strictly increasing, two
        or more.
    y : array_like
        (ny,) float64 points along the second axis, likewise.
    z : array_like
        (nx, ny) float64 values at the points of the grid.
    boundary : str or sequence of 2 str, optional
        The condition at the ends along each axis, as for ``Spline``, except
        ``"clamped"``; ``"periodic"`` for a surface repeating with the grid,
        such as an energy over the cells of a lattice. Default
        ``"not-a-knot"``.
    extrapolate : bool, optional
        Continue the end cubics beyond the grid; otherwise values there are
        NaN, or repeat along periodic axes. Default False.

    Attributes
    ----------
    x, y : numpy.ndarray
        (nx,) and (ny,) float64 points.
    z : numpy.ndarray
        (nx, ny) float64 values.

    Raises
    ------
    InvalidParameter
        For points that are not strictly increasing, values of another
        shape, or an unknown boundary.
    """

    def __init__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        *,
        boundary: str | tuple[str, str] = "not-a-knot",
        extrapolate: bool = False,
    ):
        x, y = _grid(x, "x"), _grid(y, "y")
        try:
            z = np.asarray(z, dtype=np.float64)
        except (TypeError, ValueError):
            raise InvalidParameter("z must be numbers") from None
        if z.shape != (len(x), len(y)):
            raise InvalidParameter(f"z must be ({len(x)}, {len(y)}), not {z.shape}")
        if not np.isfinite(z).all():
            raise InvalidParameter("z must be finite")
        both = (boundary, boundary) if isinstance(boundary, str) else tuple(boundary)
        if len(both) != 2 or any(b not in BOUNDARIES or b == "clamped" for b in both):
            raise InvalidParameter(
                "boundary must be 'not-a-knot', 'natural' or 'periodic', or two of them"
            )
        # along x for every y, then the coefficients along y
        first = _fitted(x, z, both[0])  # (4, nx - 1, ny)
        cells = first.transpose(2, 0, 1).reshape(len(y), -1)  # (ny, 4 (nx - 1))
        second = _fitted(y, cells, both[1])  # (4, ny - 1, 4 (nx - 1))
        # (4 in y, ny - 1, 4 in x, nx - 1) -> (nx - 1, ny - 1, 4 in x, 4 in y)
        patches = second.reshape(4, len(y) - 1, 4, len(x) - 1)
        self.__patches = patches.transpose(3, 1, 2, 0).copy()
        self.x, self.y, self.z = x, y, z
        self.boundary = both
        self.__extrapolate = bool(extrapolate)
        for array in (self.x, self.y, self.z):
            array.flags.writeable = False

    def __repr__(self) -> str:
        return f"GridSpline({len(self.x)} x {len(self.y)} points)"

    def __call__(
        self, x: ArrayLike, y: ArrayLike, *, dx: int = 0, dy: int = 0
    ) -> np.ndarray:
        """The surface or a derivative of it at points.

        Parameters
        ----------
        x, y : array_like
            float64 coordinates of the points, broadcast together.
        dx, dy : int32, optional
            The order of the derivative along each axis, 0 to 3. Default 0.

        Returns
        -------
        numpy.ndarray
            float64 of the broadcast shape of x and y.
        """
        for order, name in ((dx, "dx"), (dy, "dy")):
            if order not in (0, 1, 2, 3):
                raise InvalidParameter(f"{name} must be 0, 1, 2 or 3, not {order!r}")
        x, y = np.broadcast_arrays(
            np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        )
        shape = x.shape
        px, py = x.reshape(-1).copy(), y.reshape(-1).copy()
        outside = np.zeros(len(px), dtype=bool)
        for points, grid, kind in (
            (px, self.x, self.boundary[0]),
            (py, self.y, self.boundary[1]),
        ):
            if self.__extrapolate:
                continue
            if kind == "periodic":
                points[:] = grid[0] + np.mod(points - grid[0], grid[-1] - grid[0])
            else:
                outside |= (points < grid[0]) | (points > grid[-1])
        i = np.clip(np.searchsorted(self.x, px, side="right") - 1, 0, len(self.x) - 2)
        j = np.clip(np.searchsorted(self.y, py, side="right") - 1, 0, len(self.y) - 2)
        tx, ty = px - self.x[i], py - self.y[j]
        powers = np.arange(4)

        def basis(t, order):
            # d^order/dt^order of t^k for k = 0..3
            factor = np.array(
                [
                    np.prod(np.arange(k - order + 1, k + 1)) if k >= order else 0.0
                    for k in powers
                ]
            )
            exponent = np.clip(powers - order, 0, None)
            return factor[None, :] * t[:, None] ** exponent[None, :]

        bx, by = basis(tx, dx), basis(ty, dy)  # (p, 4)
        patches = self.__patches[i, j]  # (p, 4, 4)
        values = np.einsum("pk,pkl,pl->p", bx, patches, by)
        values[outside] = np.nan
        return values.reshape(shape)
