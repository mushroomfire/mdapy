# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""A property of the atoms in bins along the box vectors."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from mdapy.core.result import Result
from mdapy.core.system import System
from mdapy.errors import InvalidParameter

REDUCTIONS = ("mean", "sum", "density", "min", "max")


@dataclass(frozen=True)
class SpatialBinningResult(Result):
    """A property in bins along box vectors.

    Attributes
    ----------
    values : numpy.ndarray
        The property reduced over the atoms of each bin: (n1[, n2[, n3]])
        for one value per atom, with a last axis for its components; NaN
        for the mean, minimum and maximum of an empty bin.
    counts : numpy.ndarray
        (n1[, n2[, n3]]) int64 the number of atoms in each bin.
    centers : tuple of numpy.ndarray
        For each axis, the centers of its bins in Å along its box vector,
        from the origin of the box.
    """

    values: np.ndarray
    counts: np.ndarray
    centers: tuple


class SpatialBinning:
    """A property of the atoms in bins of equal size along one, two or three
    box vectors: its mean, sum, sum per volume, minimum or maximum in each.

    Parameters
    ----------
    property : str
        A column, a component such as "positions.z", or an expression over
        the columns.
    bins : int or sequence of int
        The number of bins along each axis, or one number for all.
    axes : int, str or sequence of them, optional
        The box vectors binned along: 0, 1, 2 or x, y, z. Default z.
    reduction : str, optional
        One of ``REDUCTIONS``: "mean", "sum", "density" (the sum per Å³ of
        the bin), "min" or "max". Default mean.
    """

    def __init__(
        self,
        property: str,
        bins: int | Sequence[int],
        *,
        axes: int | str | Sequence[int | str] = 2,
        reduction: str = "mean",
    ):
        names = [axes] if isinstance(axes, (int, str)) else list(axes)
        if not 1 <= len(names) <= 3:
            raise InvalidParameter("axes must be one to three box vectors")
        indices = []
        for name in names:
            if isinstance(name, str) and name in "xyz" and len(name) == 1:
                name = "xyz".index(name)
            if isinstance(name, bool) or name not in (0, 1, 2):
                raise InvalidParameter(
                    f"an axis must be 0, 1, 2, x, y or z, not {name!r}"
                )
            indices.append(int(name))
        if len(set(indices)) != len(indices):
            raise InvalidParameter("the axes must differ")
        counts = [bins] * len(indices) if np.ndim(bins) == 0 else list(bins)
        if len(counts) != len(indices) or any(
            isinstance(n, bool) or int(n) != n or n < 1 for n in counts
        ):
            raise InvalidParameter(
                f"bins must be a positive integer, or one per axis, not {bins!r}"
            )
        if reduction not in REDUCTIONS:
            raise InvalidParameter(
                f"reduction must be one of {REDUCTIONS}, not {reduction!r}"
            )
        self.property = str(property)
        self.axes = tuple(indices)
        self.bins = tuple(int(n) for n in counts)
        self.reduction = reduction

    def compute(self, system: System) -> SpatialBinningResult:
        """The property of a system in bins.

        Parameters
        ----------
        system : System
            The atoms. Along periodic vectors the atoms are wrapped into the
            box; along the others those outside it are left out.

        Returns
        -------
        SpatialBinningResult
            The values, the counts and the centers of the bins.
        """
        raw = np.asarray(system[self.property], dtype=float)
        values = raw.reshape(len(system), int(np.prod(raw.shape[1:])))
        box = system.box
        fractional = box.to_fractional(system["positions"])[:, self.axes]
        periodic = box.pbc[list(self.axes)]
        fractional[:, periodic] -= np.floor(fractional[:, periodic])
        inside = ((fractional >= 0) & (fractional < 1)).all(axis=1)
        sizes = np.array(self.bins)
        cell = np.minimum((fractional[inside] * sizes).astype(np.int64), sizes - 1)
        flat = np.ravel_multi_index(tuple(cell.T), self.bins)
        values = values[inside]
        total = int(np.prod(sizes))
        counts = np.bincount(flat, minlength=total)
        k = values.shape[1]
        if self.reduction in ("mean", "sum", "density"):
            out = np.stack(
                [np.bincount(flat, values[:, c], minlength=total) for c in range(k)], 1
            )
            if self.reduction == "mean":
                with np.errstate(invalid="ignore", divide="ignore"):
                    out = out / counts[:, None]
            elif self.reduction == "density":
                out = out / (box.volume / total)
        else:
            out = np.full((total, k), np.nan)
            order = np.argsort(flat, kind="stable")
            filled = np.flatnonzero(counts)
            if len(filled):
                starts = np.concatenate([[0], np.cumsum(counts[filled])[:-1]])
                reduce = np.minimum if self.reduction == "min" else np.maximum
                out[filled] = reduce.reduceat(values[order], starts, axis=0)
        shape = self.bins + ((k,) if raw.ndim > 1 else ())
        lengths = np.linalg.norm(box.matrix[list(self.axes)], axis=1)
        centers = tuple(
            (np.arange(n) + 0.5) / n * length for n, length in zip(self.bins, lengths)
        )
        return SpatialBinningResult(
            values=out.reshape(shape),
            counts=counts.reshape(self.bins),
            centers=centers,
        )

    def __repr__(self) -> str:
        return (
            f"SpatialBinning({self.property!r}, {self.bins!r}, axes={self.axes!r}, "
            f"reduction={self.reduction!r})"
        )
