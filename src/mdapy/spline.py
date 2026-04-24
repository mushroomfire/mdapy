# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

from mdapy import _spline
import numpy as np
from typing import Union, Tuple, List, Optional


class Spline:
    """Cubic spline interpolation on a strictly-increasing grid.

    Constructs a piecewise-cubic :math:`s(x)` that is :math:`C^2` over the whole
    range, reproduces the sample points :math:`(x_i, y_i)` exactly, and satisfies
    the chosen boundary condition at the two endpoints. The grid need not be
    uniform — if it is, prefer the internal ``UniformCubicSpline`` (used by the
    EAM code, not exposed here) which is O(1) per lookup.

    Parameters
    ----------
    x : array_like
        1-D array of x-coordinates. Must be strictly increasing and contain at
        least two points.
    y : array_like
        1-D array of y-coordinates, same length as ``x``.
    bc_type : {"not-a-knot", "natural", "clamped"}, default "not-a-knot"
        Boundary condition at the two endpoints:

        - ``"not-a-knot"`` — the third derivative is continuous at ``x[1]`` and
          ``x[n-2]`` (equivalently, the first two and last two cubic pieces are
          each a single polynomial). Same default as
          ``scipy.interpolate.CubicSpline``. Best for general data when the
          endpoint slopes are unknown.
        - ``"natural"`` — :math:`s''(x_0) = s''(x_{n-1}) = 0`. Produces a
          minimum-curvature interpolant that flattens out at the ends.
        - ``"clamped"`` — :math:`s'(x_0) = \\texttt{dy0}`,
          :math:`s'(x_{n-1}) = \\texttt{dyn}`. If ``dy0`` and ``dyn`` are not
          given, they are estimated by fitting a quadratic through the first
          (last) three points and taking its analytic derivative at the
          endpoint.
    dy0, dyn : float, optional
        Endpoint first derivatives, only used when ``bc_type="clamped"``. Both
        must be provided together; if either is ``None`` the three-point
        estimates are used.

    Notes
    -----
    Evaluation is O(log n) per point via binary search. Batch evaluation is
    OpenMP-parallelised.

    Out-of-range queries raise ``IndexError`` for scalar calls and return
    ``NaN`` element-wise for array calls. There is deliberately no silent
    extrapolation — cubic extrapolation past the last knot can swing wildly
    on smooth-looking data (see the EAM rho-clamping incident for a live
    example).

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0, 2 * np.pi, 13)
    >>> y = np.sin(x)
    >>> sp = Spline(x, y)                 # default: not-a-knot
    >>> abs(sp.evaluate(np.pi / 4) - np.sin(np.pi / 4)) < 1e-4
    True
    >>> sp.derivative(0.0)                # should be ~cos(0) = 1
    1.0000... # doctest: +SKIP

    A clamped spline with user-supplied endpoint slopes — useful when you know
    the analytic derivative at the ends (here we know :math:`\\cos(0) = 1` and
    :math:`\\cos(2\\pi) = 1`):

    >>> sp_c = Spline(x, y, bc_type="clamped", dy0=1.0, dyn=1.0)

    The natural spline, by contrast, forces :math:`s'' = 0` at the ends, which
    is appropriate when you expect the data to flatten beyond the sample range.
    """

    _BC_MAP = {
        "not-a-knot": _spline.BCType.NotAKnot,
        "natural": _spline.BCType.Natural,
        "clamped": _spline.BCType.Clamped,
    }

    def __init__(
        self,
        x: Union[List, Tuple, np.ndarray],
        y: Union[List, Tuple, np.ndarray],
        bc_type: str = "not-a-knot",
        dy0: Optional[float] = None,
        dyn: Optional[float] = None,
    ):
        self.x, self.y = self._validate(x, y)
        self.bc_type = bc_type

        if bc_type not in self._BC_MAP:
            raise ValueError(
                f"Unknown bc_type {bc_type!r}. "
                f"Expected one of {list(self._BC_MAP)}."
            )

        if bc_type == "clamped" and (dy0 is not None or dyn is not None):
            if dy0 is None or dyn is None:
                raise ValueError(
                    "For clamped with explicit derivatives both dy0 and dyn "
                    "must be given."
                )
            self._sp = _spline.CubicSpline(self.x, self.y, float(dy0), float(dyn))
        else:
            self._sp = _spline.CubicSpline(self.x, self.y, self._BC_MAP[bc_type])

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------
    def evaluate(
        self, x: Union[float, int, List, Tuple, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Evaluate :math:`s(x)` at scalar or array ``x``.

        Array inputs return an ``np.ndarray`` of the same length; entries
        outside the interpolation range become ``NaN``. Scalar inputs raise
        ``IndexError`` if out of range.
        """
        return self._call(self._sp.evaluate, x, "value")

    def derivative(
        self, x: Union[float, int, List, Tuple, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Evaluate :math:`s'(x)` at scalar or array ``x``.

        The derivative is computed analytically from the stored cubic
        coefficients, not by finite differencing.
        """
        return self._call(self._sp.derivative, x, "derivative")

    def second_derivative(
        self, x: Union[float, int, List, Tuple, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Evaluate :math:`s''(x)` at scalar or array ``x``.

        :math:`s''` is piecewise-linear between the knots (a property of
        cubic splines), so this is exact up to floating-point rounding.
        """
        return self._call(self._sp.second_derivative, x, "second derivative")

    # Convenience: ``sp(x)`` and ``sp.evaluate(x)`` do the same thing.
    __call__ = evaluate

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _call(self, backend, x, kind):
        if isinstance(x, (int, float, np.integer, np.floating)):
            xf = float(x)
            if xf < self.x[0] or xf > self.x[-1]:
                raise IndexError(
                    f"Cannot evaluate {kind} at x={xf}: outside interpolation "
                    f"range [{self.x[0]}, {self.x[-1]}]."
                )
            return backend(xf)

        if isinstance(x, np.ndarray):
            x_arr = x if x.dtype == np.float64 else x.astype(np.float64)
        elif isinstance(x, (list, tuple)):
            x_arr = np.asarray(x, dtype=np.float64)
        else:
            raise TypeError(
                f"Input type {type(x)} not supported. "
                "Expected float, int, list, tuple, or numpy.ndarray."
            )
        return backend(x_arr)

    @staticmethod
    def _validate(x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError(f"x must be 1-dimensional, got {x.ndim}D array")
        if y.ndim != 1:
            raise ValueError(f"y must be 1-dimensional, got {y.ndim}D array")
        if len(x) < 2:
            raise ValueError(f"x must have at least 2 points, got {len(x)}")
        if len(x) != len(y):
            raise ValueError(
                f"Length of x and y must match. Got x: {len(x)}, y: {len(y)}"
            )
        return x, y


if __name__ == "__main__":
    pass
