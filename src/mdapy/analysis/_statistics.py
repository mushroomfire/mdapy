# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""One number from a column, and the histogram of a column."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from mdapy.core._atoms import _mask
from mdapy.core.result import Result
from mdapy.core.system import System
from mdapy.errors import InvalidParameter

OPERATIONS = (
    "mean",
    "sum",
    "min",
    "max",
    "median",
    "variance",
    "std",
    "nonzero",
    "argmin",
    "argmax",
)


def _numbers(system: System, column: str, atoms: ArrayLike | None) -> np.ndarray:
    """(n,) or (n, k) the values of a column, a component or an expression of
    the atoms given, as float64."""
    try:
        values = np.asarray(system[column], dtype=np.float64)
    except (TypeError, ValueError):
        raise InvalidParameter(f"{column} is not numbers") from None
    if atoms is not None:
        values = values[_mask(atoms, len(system))]
    return values


def reduce(
    system: System,
    column: str,
    operation: str = "mean",
    *,
    atoms: ArrayLike | None = None,
) -> Any:
    """One number from the values of a column, as OVITO's reduce property.

    Parameters
    ----------
    system : System
        The atoms.
    column : str
        A column of numbers, a component such as ``"positions.z"`` or an
        expression over the columns.
    operation : str, optional
        One of ``OPERATIONS``: "mean", "sum", "min", "max", "median",
        "variance" (of a sample), "std" (of a sample), "nonzero" (the number
        of values not zero), "argmin" or "argmax" (the index of the atom of
        the lowest or highest value). Default mean.
    atoms : array_like, optional
        A boolean mask or indices of the atoms to take; all by default.

    Returns
    -------
    float, int or numpy.ndarray
        The number; for an array column, (k,) one per component. NaN values
        are left out; the mean of no values is NaN, their sum 0. argmin and
        argmax index the atoms of the system.

    Raises
    ------
    InvalidParameter
        For an unknown operation, a column that is not numbers, or argmin or
        argmax of no values.
    """
    if operation not in OPERATIONS:
        raise InvalidParameter(
            f"operation must be one of {OPERATIONS}, not {operation!r}"
        )
    values = _numbers(system, column, None)
    chosen = (
        np.arange(len(system))
        if atoms is None
        else np.flatnonzero(_mask(atoms, len(system)))
    )
    values = values[chosen]
    columns = values.reshape(len(values), int(np.prod(values.shape[1:])))
    found = []
    for k in range(columns.shape[1]):
        v = columns[:, k]
        kept = ~np.isnan(v)
        finite = v[kept]
        if operation == "sum":
            found.append(float(finite.sum()))
        elif operation == "nonzero":
            found.append(int(np.count_nonzero(finite)))
        elif operation in ("argmin", "argmax"):
            if not len(finite):
                raise InvalidParameter(f"{operation} of no values")
            at = np.argmin(finite) if operation == "argmin" else np.argmax(finite)
            found.append(int(chosen[np.flatnonzero(kept)[at]]))
        elif not len(finite):
            found.append(float("nan"))
        elif operation == "mean":
            found.append(float(finite.mean()))
        elif operation == "min":
            found.append(float(finite.min()))
        elif operation == "max":
            found.append(float(finite.max()))
        elif operation == "median":
            found.append(float(np.median(finite)))
        elif operation == "variance":
            found.append(float(finite.var(ddof=1)) if len(finite) > 1 else float("nan"))
        else:
            found.append(float(finite.std(ddof=1)) if len(finite) > 1 else float("nan"))
    if values.ndim == 1:
        return found[0]
    return np.asarray(found)


@dataclass(frozen=True)
class HistogramResult(Result):
    """The histogram of a column.

    Attributes
    ----------
    center : numpy.ndarray
        (bins,) centers of the bins.
    edges : numpy.ndarray
        (bins + 1,) edges of the bins.
    counts : numpy.ndarray
        (bins,) int64 number of values in each bin.
    frequency : numpy.ndarray
        (bins,) the counts divided by the number of values counted.
    density : numpy.ndarray
        (bins,) the frequency divided by the width of a bin.
    nonfinite : int
        The number of values left out as NaN or infinite.
    name : str
        The column.
    """

    center: np.ndarray
    edges: np.ndarray
    counts: np.ndarray
    frequency: np.ndarray
    density: np.ndarray
    nonfinite: int
    name: str


class Histogram:
    """The histogram of the values of a column, as OVITO's histogram
    modifier.

    Parameters
    ----------
    column : str
        A column of numbers, a component such as ``"positions.z"`` or an
        expression over the columns.
    bins : int, optional
        Number of bins. Default 50.
    range : tuple of float, optional
        Lowest and highest value of the bins; values outside are not
        counted. By default from the lowest to the highest finite value.
    """

    def __init__(
        self,
        column: str,
        bins: int = 50,
        range: tuple[float, float] | None = None,  # noqa: A002 (numpy's name)
    ):
        if bins < 1:
            raise InvalidParameter(f"bins must be at least 1, not {bins!r}")
        if range is not None:
            low, high = (float(v) for v in range)
            if not high > low:
                raise InvalidParameter(f"range must increase, not {range!r}")
            range = (low, high)
        self.column = column
        self.bins = int(bins)
        self.range = range

    def compute(
        self, system: System, *, atoms: ArrayLike | None = None
    ) -> HistogramResult:
        """The histogram of a system.

        Parameters
        ----------
        system : System
            The atoms.
        atoms : array_like, optional
            A boolean mask or indices of the atoms to count; all by default.

        Returns
        -------
        HistogramResult
            The counts in the bins. Without a range, the bins span the lowest
            to the highest finite value, or one unit above a single value.
        """
        values = _numbers(system, self.column, atoms)
        if values.ndim != 1:
            raise InvalidParameter(f"{self.column} has components: give one of them")
        finite = values[np.isfinite(values)]
        if self.range is not None:
            low, high = self.range
        elif len(finite):
            low, high = float(finite.min()), float(finite.max())
        else:
            low, high = 0.0, 1.0
        if high <= low:
            high = low + 1.0
        counts, edges = np.histogram(finite, bins=self.bins, range=(low, high))
        total = counts.sum()
        frequency = counts / total if total else np.zeros(self.bins)
        return HistogramResult(
            center=0.5 * (edges[1:] + edges[:-1]),
            edges=edges,
            counts=counts.astype(np.int64),
            frequency=frequency,
            density=frequency / (edges[1] - edges[0]),
            nonfinite=int(len(values) - len(finite)),
            name=self.column,
        )

    def __repr__(self) -> str:
        return f"Histogram({self.column!r}, bins={self.bins}, range={self.range})"
