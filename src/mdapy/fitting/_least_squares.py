# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Least-squares fits of polynomials, surfaces and other functions."""

from collections.abc import Callable
from dataclasses import dataclass
from math import comb

import numpy as np
from numpy.typing import ArrayLike

from mdapy import units
from mdapy.errors import InvalidParameter


def _numbers(values: ArrayLike, name: str) -> np.ndarray:
    try:
        values = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        raise InvalidParameter(f"{name} must be numbers") from None
    if not np.isfinite(values).all():
        raise InvalidParameter(f"{name} must be finite")
    return values


def _weights(sigma: ArrayLike | None, count: int) -> np.ndarray:
    if sigma is None:
        return np.ones(count)
    sigma = np.broadcast_to(_numbers(sigma, "sigma"), (count,))
    if not (sigma > 0).all():
        raise InvalidParameter("sigma must be positive")
    return 1.0 / sigma


def _covariance(
    design: np.ndarray, residuals: np.ndarray, absolute: bool
) -> np.ndarray:
    """The covariance of least-squares parameters from the weighted design
    matrix and weighted residuals, scaled by the variance of the residuals
    unless the weights are absolute."""
    rows, size = design.shape
    covariance = np.linalg.pinv(design.T @ design)
    if absolute:
        return covariance
    if rows <= size:
        return np.full((size, size), np.nan)
    return covariance * float(residuals @ residuals) / (rows - size)


def _r_squared(y: np.ndarray, fitted: np.ndarray) -> float:
    total = float(((y - y.mean()) ** 2).sum())
    left = float(((y - fitted) ** 2).sum())
    return 1.0 - left / total if total > 0 else float(left == 0.0)


def _read_only(*arrays: np.ndarray) -> None:
    for array in arrays:
        array.flags.writeable = False


class PolynomialFit:
    """A polynomial fitted by least squares to values along a line, made by
    ``fit_polynomial``.

    Attributes
    ----------
    coefficients : numpy.ndarray
        (degree + 1,) float64 coefficients of 1, x, x², ... .
    covariance : numpy.ndarray
        (degree + 1, degree + 1) float64 covariance of the coefficients.
    errors : numpy.ndarray
        (degree + 1,) float64 standard errors of the coefficients.
    residuals : numpy.ndarray
        (n,) float64 the values less the polynomial at the points.
    r_squared : float64
        The coefficient of determination, 1 for a perfect fit.
    """

    def __init__(
        self, center, scale, scaled, transform, covariance, residuals, r_squared
    ):
        self.__center, self.__scale, self.__scaled = center, scale, scaled
        self.coefficients = transform @ scaled
        self.covariance = transform @ covariance @ transform.T
        self.errors = np.sqrt(np.diagonal(self.covariance)).copy()
        self.residuals = residuals
        self.r_squared = r_squared
        _read_only(self.coefficients, self.covariance, self.errors, self.residuals)

    def __repr__(self) -> str:
        return f"PolynomialFit(degree {len(self.coefficients) - 1})"

    def __call__(self, x: ArrayLike, derivative: int = 0) -> np.ndarray:
        """The polynomial or one of its derivatives at points.

        Parameters
        ----------
        x : array_like
            float64 points, of any shape.
        derivative : int32, optional
            The order of the derivative, 0 for the values. Default 0.

        Returns
        -------
        numpy.ndarray
            float64 of the shape of x.
        """
        if (
            isinstance(derivative, bool)
            or not isinstance(derivative, (int, np.integer))
            or derivative < 0
        ):
            raise InvalidParameter(
                f"derivative must be a whole number from 0, not {derivative!r}"
            )
        u = (np.asarray(x, dtype=np.float64) - self.__center) / self.__scale
        coefficients = np.polynomial.polynomial.polyder(self.__scaled, derivative)
        return (
            np.polynomial.polynomial.polyval(u, coefficients) / self.__scale**derivative
        )


def fit_polynomial(
    x: ArrayLike,
    y: ArrayLike,
    degree: int,
    *,
    sigma: ArrayLike | None = None,
    absolute_sigma: bool = False,
) -> PolynomialFit:
    """A polynomial fitted to values along a line by least squares.

    Parameters
    ----------
    x : array_like
        (n,) float64 points.
    y : array_like
        (n,) float64 values at the points.
    degree : int32
        The degree of the polynomial, 0 or more, below n.
    sigma : array_like, optional
        (n,) float64 standard deviations of the values, which weigh them by
        1/sigma; all equal by default.
    absolute_sigma : bool, optional
        Take sigma as the true standard deviations for the errors of the
        coefficients; otherwise scale them so that the residuals have unit
        variance. Default False.

    Returns
    -------
    PolynomialFit
        The coefficients, their covariance and errors, and the residuals.

    Raises
    ------
    InvalidParameter
        For points and values of other shapes or not finite, a degree out of
        range, or sigma not positive.
    """
    x, y = _numbers(x, "x"), _numbers(y, "y")
    if x.ndim != 1 or y.shape != x.shape:
        raise InvalidParameter("x and y must be lines of the same length")
    if (
        isinstance(degree, bool)
        or not isinstance(degree, (int, np.integer))
        or not 0 <= degree < len(x)
    ):
        raise InvalidParameter(f"degree must be from 0 to {len(x) - 1}, not {degree!r}")
    weights = _weights(sigma, len(x))
    center = 0.5 * (x.max() + x.min())
    scale = 0.5 * (x.max() - x.min()) or 1.0
    u = (x - center) / scale
    design = np.polynomial.polynomial.polyvander(u, degree)
    weighted = design * weights[:, None]
    scaled, *_ = np.linalg.lstsq(weighted, y * weights, rcond=None)
    fitted = design @ scaled
    covariance = _covariance(weighted, (y - fitted) * weights, absolute_sigma)
    # (x - center)^j / scale^j as powers of x
    transform = np.zeros((degree + 1, degree + 1))
    for j in range(degree + 1):
        for i in range(j + 1):
            transform[i, j] = comb(j, i) * (-center) ** (j - i) / scale**j
    return PolynomialFit(
        center, scale, scaled, transform, covariance, y - fitted, _r_squared(y, fitted)
    )


class SurfaceFit:
    """A polynomial in two variables fitted by least squares, made by
    ``fit_surface``.

    Attributes
    ----------
    powers : numpy.ndarray
        (k, 2) int64 the powers (i, j) of each term x^i y^j.
    coefficients : numpy.ndarray
        (k,) float64 the coefficient of each term.
    covariance : numpy.ndarray
        (k, k) float64 covariance of the coefficients.
    errors : numpy.ndarray
        (k,) float64 standard errors of the coefficients.
    residuals : numpy.ndarray
        (n,) float64 the values less the surface at the points.
    r_squared : float64
        The coefficient of determination, 1 for a perfect fit.
    """

    def __init__(
        self, powers, center, scale, scaled, transform, covariance, residuals, r_squared
    ):
        self.powers = powers
        self.__center, self.__scale, self.__scaled = center, scale, scaled
        self.coefficients = transform @ scaled
        self.covariance = transform @ covariance @ transform.T
        self.errors = np.sqrt(np.diagonal(self.covariance)).copy()
        self.residuals = residuals
        self.r_squared = r_squared
        _read_only(
            self.powers, self.coefficients, self.covariance, self.errors, self.residuals
        )

    def __repr__(self) -> str:
        return f"SurfaceFit(degree {int(self.powers.sum(axis=1).max())})"

    def __call__(
        self, x: ArrayLike, y: ArrayLike, *, dx: int = 0, dy: int = 0
    ) -> np.ndarray:
        """The surface or a derivative of it at points.

        Parameters
        ----------
        x, y : array_like
            float64 coordinates of the points, broadcast together.
        dx, dy : int32, optional
            The order of the derivative along each axis. Default 0.

        Returns
        -------
        numpy.ndarray
            float64 of the broadcast shape of x and y.
        """
        for order, name in ((dx, "dx"), (dy, "dy")):
            if (
                isinstance(order, bool)
                or not isinstance(order, (int, np.integer))
                or order < 0
            ):
                raise InvalidParameter(
                    f"{name} must be a whole number from 0, not {order!r}"
                )
        x, y = np.broadcast_arrays(
            np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        )
        u = (x - self.__center[0]) / self.__scale[0]
        v = (y - self.__center[1]) / self.__scale[1]
        total = np.zeros(x.shape)
        for (i, j), c in zip(self.powers, self.__scaled):
            if i < dx or j < dy:
                continue
            factor = np.prod(np.arange(i - dx + 1, i + 1)) * np.prod(
                np.arange(j - dy + 1, j + 1)
            )
            total = total + c * factor * u ** (i - dx) * v ** (j - dy)
        return total / (self.__scale[0] ** dx * self.__scale[1] ** dy)


def fit_surface(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    degree: int,
    *,
    sigma: ArrayLike | None = None,
    absolute_sigma: bool = False,
) -> SurfaceFit:
    """A polynomial in two variables, of every term x^i y^j with i + j up to
    the degree, fitted to values at points by least squares.

    Parameters
    ----------
    x, y : array_like
        (n,) float64 coordinates of the points.
    z : array_like
        (n,) float64 values at the points.
    degree : int32
        The total degree, 0 or more, with fewer terms than points.
    sigma : array_like, optional
        (n,) float64 standard deviations of the values; all equal by default.
    absolute_sigma : bool, optional
        As for ``fit_polynomial``. Default False.

    Returns
    -------
    SurfaceFit
        The powers and coefficients of the terms, their covariance and
        errors, and the residuals.

    Raises
    ------
    InvalidParameter
        For coordinates and values of other shapes or not finite, a degree out
        of range, or sigma not positive.
    """
    x, y, z = _numbers(x, "x"), _numbers(y, "y"), _numbers(z, "z")
    if x.ndim != 1 or y.shape != x.shape or z.shape != x.shape:
        raise InvalidParameter("x, y and z must be lines of the same length")
    powers = np.array(
        [(i, d - i) for d in range(int(degree) + 1) for i in range(d, -1, -1)]
        if not isinstance(degree, bool)
        and isinstance(degree, (int, np.integer))
        and degree >= 0
        else np.zeros((0, 2)),
        dtype=np.int64,
    ).reshape(-1, 2)
    if len(powers) == 0 or len(powers) > len(x):
        raise InvalidParameter(
            f"degree must be 0 or more, with at most {len(x)} terms, not {degree!r}"
        )
    weights = _weights(sigma, len(x))
    points = np.stack([x, y])
    center = 0.5 * (points.max(axis=1) + points.min(axis=1))
    scale = 0.5 * (points.max(axis=1) - points.min(axis=1))
    scale[scale == 0] = 1.0
    u, v = (points - center[:, None]) / scale[:, None]
    design = np.stack([u**i * v**j for i, j in powers], axis=1)
    weighted = design * weights[:, None]
    scaled, *_ = np.linalg.lstsq(weighted, z * weights, rcond=None)
    fitted = design @ scaled
    covariance = _covariance(weighted, (z - fitted) * weights, absolute_sigma)
    # u^a v^b as terms x^i y^j
    index = {(int(i), int(j)): k for k, (i, j) in enumerate(powers)}
    transform = np.zeros((len(powers), len(powers)))
    for k, (a, b) in enumerate(powers):
        for i in range(a + 1):
            for j in range(b + 1):
                transform[index[(i, j)], k] += (
                    comb(int(a), i)
                    * (-center[0]) ** (a - i)
                    / scale[0] ** a
                    * comb(int(b), j)
                    * (-center[1]) ** (b - j)
                    / scale[1] ** b
                )
    return SurfaceFit(
        powers,
        center,
        scale,
        scaled,
        transform,
        covariance,
        z - fitted,
        _r_squared(z, fitted),
    )


@dataclass(frozen=True)
class Model:
    """A function with named parameters to fit with ``fit_curve``.

    Attributes
    ----------
    name : str
        What it is called.
    function : callable
        ``function(x, *parameters)``, the values at x.
    parameters : tuple of str
        The names of the parameters, in the order the function takes them.
    guess : callable or None
        ``guess(x, y)``, a first guess of the parameters from the data.
    """

    name: str
    function: Callable
    parameters: tuple[str, ...]
    guess: Callable | None = None

    def __call__(self, x: ArrayLike, *parameters: float) -> np.ndarray:
        return self.function(np.asarray(x, dtype=np.float64), *parameters)


def _line(x, y):
    """The slope and intercept of the straight line through points, by least
    squares."""
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def _guess_exponential(x, y):
    keep = y > 0
    if keep.sum() < 2:
        return np.array([float(y[np.argmax(np.abs(y))]), float(np.ptp(x)) or 1.0])
    slope, intercept = _line(x[keep], np.log(y[keep]))
    time = -1.0 / slope if slope < 0 else float(np.ptp(x)) or 1.0
    return np.array([np.exp(intercept), time])


def _guess_stretched(x, y):
    amplitude, time = _guess_exponential(x, y)
    return np.array([amplitude, time, 1.0])


def _guess_peak(x, y):
    top = int(np.argmax(y))
    above = x[y >= 0.5 * y[top]]
    width = 0.5 * float(np.ptp(above)) or 0.1 * float(np.ptp(x)) or 1.0
    return np.array([float(y[top]), float(x[top]), width])


def _guess_arrhenius(x, y):
    slope, intercept = _line(1.0 / (units.BOLTZMANN * x), np.log(np.abs(y)))
    return np.array([np.exp(intercept), -slope])


def _guess_power(x, y):
    slope, intercept = _line(np.log(x), np.log(np.abs(y)))
    return np.array([np.exp(intercept), slope])


exponential = Model(
    "exponential",
    lambda x, amplitude, time: amplitude * np.exp(-x / time),
    ("amplitude", "time"),
    _guess_exponential,
)
stretched_exponential = Model(
    "stretched exponential",
    lambda x, amplitude, time, exponent: (
        amplitude * np.exp(-(np.abs(x / time) ** exponent))
    ),
    ("amplitude", "time", "exponent"),
    _guess_stretched,
)
gaussian = Model(
    "gaussian",
    lambda x, amplitude, center, width: (
        amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)
    ),
    ("amplitude", "center", "width"),
    _guess_peak,
)
lorentzian = Model(
    "lorentzian",
    lambda x, amplitude, center, width: (
        amplitude * width**2 / ((x - center) ** 2 + width**2)
    ),
    ("amplitude", "center", "width"),
    _guess_peak,
)
arrhenius = Model(
    "arrhenius",
    lambda x, prefactor, energy: prefactor * np.exp(-energy / (units.BOLTZMANN * x)),
    ("prefactor", "energy"),
    _guess_arrhenius,
)
power_law = Model(
    "power law",
    lambda x, prefactor, exponent: prefactor * np.abs(x) ** exponent,
    ("prefactor", "exponent"),
    _guess_power,
)
MODELS = (
    exponential,
    stretched_exponential,
    gaussian,
    lorentzian,
    arrhenius,
    power_law,
)


class CurveFit:
    """The parameters of a function fitted to values by least squares, made
    by ``fit_curve``.

    Attributes
    ----------
    model : Model
        The function, with the names of its parameters.
    parameters : numpy.ndarray
        (p,) float64 the fitted parameters.
    errors : numpy.ndarray
        (p,) float64 their standard errors.
    covariance : numpy.ndarray
        (p, p) float64 their covariance.
    residuals : numpy.ndarray
        (n,) float64 the values less the function at the points.
    r_squared : float64
        The coefficient of determination, 1 for a perfect fit.
    steps : int32
        The steps the fit took.
    """

    def __init__(
        self, model, parameters, errors, covariance, residuals, r_squared, steps
    ):
        self.model = model
        self.parameters = parameters
        self.errors = errors
        self.covariance = covariance
        self.residuals = residuals
        self.r_squared = r_squared
        self.steps = steps
        _read_only(self.parameters, self.errors, self.covariance, self.residuals)

    def __repr__(self) -> str:
        values = ", ".join(
            f"{name}={value:.6g}"
            for name, value in zip(self.model.parameters, self.parameters)
        )
        return f"CurveFit({self.model.name}: {values})"

    def __getitem__(self, name: str) -> float:
        """The fitted value of the parameter of that name."""
        if name not in self.model.parameters:
            raise KeyError(name)
        return float(self.parameters[self.model.parameters.index(name)])

    def __call__(self, x: ArrayLike) -> np.ndarray:
        """The fitted function at points.

        Parameters
        ----------
        x : array_like
            The points, as the function takes them.

        Returns
        -------
        numpy.ndarray
            float64 values.
        """
        return np.asarray(self.model(x, *self.parameters), dtype=np.float64)


def _jacobian(model, x, parameters):
    """(n, p) the derivatives of the values with respect to each parameter, by
    central differences."""
    columns = []
    for k in range(len(parameters)):
        step = np.sqrt(np.finfo(float).eps) ** (2 / 3) * max(abs(parameters[k]), 1e-3)
        up, down = parameters.copy(), parameters.copy()
        up[k] += step
        down[k] -= step
        with np.errstate(all="ignore"):
            columns.append(
                (np.asarray(model(x, *up)) - np.asarray(model(x, *down))).reshape(-1)
                / (2 * step)
            )
    return np.stack(columns, axis=1)


def fit_curve(
    model: Model | Callable,
    x: ArrayLike,
    y: ArrayLike,
    guess: ArrayLike | None = None,
    *,
    sigma: ArrayLike | None = None,
    absolute_sigma: bool = False,
    max_steps: int = 500,
    tolerance: float = 1e-12,
) -> CurveFit:
    """The parameters of a function that fit values best, by the
    Levenberg-Marquardt method of nonlinear least squares.

    Parameters
    ----------
    model : Model or callable
        One of ``MODELS``, a ``Model``, or a function ``f(x, *parameters)``.
    x : array_like
        The points, as the function takes them: (n,) float64, or for a
        function of several variables any array it takes.
    y : array_like
        (n,) float64 values at the points.
    guess : array_like, optional
        (p,) float64 first guess of the parameters; by default the guess of
        the model, which a plain function must be given.
    sigma : array_like, optional
        (n,) float64 standard deviations of the values; all equal by default.
    absolute_sigma : bool, optional
        As for ``fit_polynomial``. Default False.
    max_steps : int32, optional
        The most steps. Default 500.
    tolerance : float64, optional
        The fit has converged when a step changes the sum of squares by less
        than this fraction of it. Default 1e-12.

    Returns
    -------
    CurveFit
        The parameters, their errors and covariance, and the residuals.

    Raises
    ------
    InvalidParameter
        For values that are not finite, a plain function without a guess, a
        function whose values are not finite at the guess, or a fit that
        does not converge within ``max_steps``.
    """
    if not isinstance(model, Model):
        if not callable(model):
            raise InvalidParameter("model must be a Model or a function")
        if guess is None:
            raise InvalidParameter("a function without a model needs a guess")
        count = len(np.atleast_1d(guess))
        name = getattr(model, "__name__", "function")
        model = Model(name, model, tuple(f"p{k}" for k in range(count)))
    x = np.asarray(x, dtype=np.float64)
    y = _numbers(y, "y").reshape(-1)
    weights = _weights(sigma, len(y))
    if guess is None:
        guess = model.guess(x, y) if model.guess is not None else None
        if guess is None:
            raise InvalidParameter(f"the model {model.name!r} needs a guess")
    parameters = _numbers(guess, "guess").reshape(-1).copy()
    if len(parameters) != len(model.parameters):
        raise InvalidParameter(
            f"{model.name} has {len(model.parameters)} parameters, not {len(parameters)}"
        )

    def residuals(p):
        # trial parameters may leave the domain of the function: not finite
        with np.errstate(all="ignore"):
            values = np.asarray(model(x, *p), dtype=np.float64).reshape(-1)
        return (y - values) * weights

    current = residuals(parameters)
    if not np.isfinite(current).all():
        raise InvalidParameter("the function is not finite at the guess")
    cost = float(current @ current)
    damping = None
    for step in range(1, int(max_steps) + 1):
        jacobian = -_jacobian(model, x, parameters) * weights[:, None]
        normal = jacobian.T @ jacobian
        gradient = jacobian.T @ current
        diagonal = np.maximum(np.diagonal(normal), 1e-300)
        if damping is None:
            damping = 1e-3 * diagonal.max()
        while True:
            try:
                change = np.linalg.solve(
                    normal + damping * np.diag(diagonal), -gradient
                )
            except np.linalg.LinAlgError:
                damping *= 10
                continue
            trial = parameters + change
            moved = residuals(trial)
            new = float(moved @ moved) if np.isfinite(moved).all() else np.inf
            if new <= cost:
                break
            damping *= 10
            if damping > 1e20 * max(diagonal.max(), 1.0):
                new = cost
                trial = parameters
                break
        done = cost - new <= tolerance * max(cost, np.finfo(float).tiny) or cost == 0.0
        parameters, current, cost = trial, moved if new < cost else current, new
        damping = max(damping / 10, 1e-15 * diagonal.max())
        if done:
            break
    else:
        raise InvalidParameter(f"the fit did not converge within {max_steps} steps")
    jacobian = _jacobian(model, x, parameters) * weights[:, None]
    covariance = _covariance(jacobian, current, absolute_sigma)
    fitted = y - current / weights
    return CurveFit(
        model,
        parameters,
        np.sqrt(np.diagonal(covariance)).copy(),
        covariance,
        y - fitted,
        _r_squared(y, fitted),
        step,
    )
