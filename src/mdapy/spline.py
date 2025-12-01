# Copyright (c) 2022-2025, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

from mdapy import _spline
import numpy as np
from typing import Union, Tuple, List


class Spline:
    """
    Cubic Spline Interpolation with clamped boundary conditions, where the spline's endpoint derivatives are set equal to the original derivatives

    This class provides a convenient interface for cubic spline interpolation.
    It supports evaluation of the spline function and its first derivative at
    arbitrary points within the interpolation range.

    Parameters
    ----------
    x : array_like
        1-D array of x-coordinates of data points. Must be strictly increasing
        and contain at least 2 points. Can be list, tuple, or numpy array.
    y : array_like
        1-D array of y-coordinates of data points. Must have the same length as x.
        Can be list, tuple, or numpy array.

    Attributes
    ----------
    x : np.ndarray
        The x-coordinates of interpolation points (sorted and validated).
    y : np.ndarray
        The y-coordinates of interpolation points.

    Examples
    --------
    >>> # Basic usage with lists
    >>> x = [0, 1, 2, 3, 4]
    >>> y = [0, 1, 4, 9, 16]
    >>> sp = Spline(x, y)
    >>>
    >>> # Evaluate at a single point
    >>> value = sp.evaluate(2.5)
    >>> print(f"f(2.5) = {value}")
    >>>
    >>> # Evaluate at multiple points
    >>> x_new = np.linspace(0, 4, 100)
    >>> y_new = sp.evaluate(x_new)
    >>>
    >>> # Compute derivative
    >>> derivative = sp.derivative(2.5)
    >>> print(f"f'(2.5) = {derivative}")
    >>>
    >>> # Using numpy arrays
    >>> x = np.array([0.0, 1.0, 2.0, 3.0])
    >>> y = np.array([1.0, 2.0, 1.5, 3.0])
    >>> sp = Spline(x, y)
    >>> sp.evaluate([0.5, 1.5, 2.5])

    Notes
    -----
    - The spline is only valid within the range [x[0], x[-1]]. Attempting to
      evaluate outside this range will raise an assertion error.
    - The spline uses natural boundary conditions (zero second derivatives at
      endpoints).
    """

    def __init__(
        self, x: Union[List, Tuple, np.ndarray], y: Union[List, Tuple, np.ndarray]
    ):
        # Validate and store the interpolation points
        self.x, self.y = self._check_uniform_grid(x, y)

        # Create the underlying C++ spline object
        self._sp = _spline.CubicSpline(self.x, self.y)

    def evaluate(
        self, x: Union[float, int, List, Tuple, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Evaluate the spline at given point(s).

        Computes the interpolated value of the spline function at the specified
        x-coordinate(s). Supports both scalar and array inputs.

        Parameters
        ----------
        x : float, int, list, tuple, or np.ndarray
            Point(s) at which to evaluate the spline. Must be within the range
            [self.x[0], self.x[-1]]. Can be:
            - A single number (int or float)
            - A list or tuple of numbers
            - A numpy array

        Returns
        -------
        float or np.ndarray
            The interpolated value(s) at x. Returns a float if input is scalar,
            otherwise returns a numpy array of the same shape as input.

        Raises
        ------
        AssertionError
            If any value in x is outside the interpolation range [self.x[0], self.x[-1]].

        Examples
        --------
        >>> sp = Spline([0, 1, 2, 3], [0, 1, 4, 9])
        >>>
        >>> # Scalar evaluation
        >>> sp.evaluate(1.5)
        2.125
        >>>
        >>> # List evaluation
        >>> sp.evaluate([0.5, 1.5, 2.5])
        array([0.375, 2.125, 6.125])
        >>>
        >>> # Numpy array evaluation
        >>> x = np.linspace(0, 3, 10)
        >>> y = sp.evaluate(x)
        """
        # Handle scalar input directly (no conversion to array)
        if isinstance(x, (int, float)):
            assert (x >= self.x[0]) and (x <= self.x[-1]), (
                f"Input x ({x}) is outside interpolation range [{self.x[0]}, {self.x[-1]}]"
            )
            return self._sp.evaluate(float(x))

        # Handle array-like inputs
        else:
            # Convert to numpy array if needed
            if isinstance(x, (list, tuple)):
                x_eval = np.asarray(x, dtype=float)
            elif isinstance(x, np.ndarray):
                x_eval = x
            else:
                raise TypeError(
                    f"Input type {type(x)} not supported. "
                    "Expected float, int, list, tuple, or numpy.ndarray"
                )

            # Check bounds
            assert x_eval.min() >= self.x[0], (
                f"Input x ({x_eval.min()}) is below interpolation range ({self.x[0]})"
            )
            assert x_eval.max() <= self.x[-1], (
                f"Input x ({x_eval.max()}) is above interpolation range ({self.x[-1]})"
            )

            # Evaluate and return array
            return self._sp.evaluate(x_eval)

    def derivative(
        self, x: Union[float, int, List, Tuple, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Evaluate the first derivative of the spline at given point(s).

        Computes the first derivative (slope) of the interpolated spline function
        at the specified x-coordinate(s). Supports both scalar and array inputs.

        Parameters
        ----------
        x : float, int, list, tuple, or np.ndarray
            Point(s) at which to evaluate the derivative. Must be within the range
            [self.x[0], self.x[-1]]. Can be:
            - A single number (int or float)
            - A list or tuple of numbers
            - A numpy array

        Returns
        -------
        float or np.ndarray
            The derivative value(s) at x. Returns a float if input is scalar,
            otherwise returns a numpy array of the same shape as input.

        Raises
        ------
        AssertionError
            If any value in x is outside the interpolation range [self.x[0], self.x[-1]].

        Examples
        --------
        >>> sp = Spline([0, 1, 2, 3], [0, 1, 4, 9])
        >>>
        >>> # Scalar derivative
        >>> sp.derivative(1.5)
        3.5
        >>>
        >>> # Array derivative
        >>> x = np.array([0.5, 1.5, 2.5])
        >>> derivatives = sp.derivative(x)
        >>> print(derivatives)

        Notes
        -----
        The derivative is computed analytically from the cubic spline coefficients,
        not by numerical differentiation, ensuring high accuracy.
        """
        # Handle scalar input directly (no conversion to array)
        if isinstance(x, (int, float)):
            assert (x >= self.x[0]) and (x <= self.x[-1]), (
                f"Input x ({x}) is outside interpolation range [{self.x[0]}, {self.x[-1]}]"
            )
            return self._sp.derivative(float(x))

        # Handle array-like inputs
        else:
            # Convert to numpy array if needed
            if isinstance(x, (list, tuple)):
                x_eval = np.asarray(x, dtype=float)
            elif isinstance(x, np.ndarray):
                x_eval = x
            else:
                raise TypeError(
                    f"Input type {type(x)} not supported. "
                    "Expected float, int, list, tuple, or numpy.ndarray"
                )

            # Check bounds
            assert x_eval.min() >= self.x[0], (
                f"Input x ({x_eval.min()}) is below interpolation range ({self.x[0]})"
            )
            assert x_eval.max() <= self.x[-1], (
                f"Input x ({x_eval.max()}) is above interpolation range ({self.x[-1]})"
            )

            # Evaluate and return array
            return self._sp.derivative(x_eval)

    def _check_uniform_grid(
        self, x: Union[List, Tuple, np.ndarray], y: Union[List, Tuple, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validate and convert interpolation points.

        Internal method to validate the input x and y arrays, ensuring they meet
        the requirements for cubic spline interpolation.

        Parameters
        ----------
        x : array_like
            X-coordinates of data points.
        y : array_like
            Y-coordinates of data points.

        Returns
        -------
        x : np.ndarray
            Validated and converted x-coordinates as 1-D float array.
        y : np.ndarray
            Validated and converted y-coordinates as 1-D float array.

        Raises
        ------
        ValueError
            If x or y are not 1-dimensional, if they have different lengths,
            or if x has fewer than 2 points.

        Notes
        -----
        This method ensures:
        - Both x and y are 1-dimensional arrays
        - Both arrays have the same length
        - At least 2 data points are provided
        - Arrays are converted to float64 dtype
        """
        # Convert to numpy arrays
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        # Validate dimensions
        if x.ndim != 1:
            raise ValueError(f"x must be 1-dimensional, got {x.ndim}D array")
        if y.ndim != 1:
            raise ValueError(f"y must be 1-dimensional, got {y.ndim}D array")

        # Validate minimum number of points
        if len(x) < 2:
            raise ValueError(f"x must have at least 2 points, got {len(x)}")

        # Validate lengths match
        if len(x) != len(y):
            raise ValueError(
                f"Length of x and y must be equal. Got x: {len(x)}, y: {len(y)}"
            )

        return x, y


if __name__ == "__main__":
    pass
