# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""A property of the atoms in bins along the box vectors."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from mdapy import units
from mdapy.core.result import Result
from mdapy.core.system import System
from mdapy.elements import atomic_masses
from mdapy.errors import InvalidParameter, MissingProperty

REDUCTIONS = (
    "mean",
    "sum",
    "density",
    "min",
    "max",
    "mass_mean",
    "temperature",
    "stress",
)
# the reductions about the center of mass of each bin, which need the masses
_MOVING = ("mass_mean", "temperature", "stress")


@dataclass(frozen=True)
class SpatialBinningResult(Result):
    """A property in bins along box vectors.

    Attributes
    ----------
    values : numpy.ndarray
        The float64 property reduced over the atoms of each bin: (n1[, n2[, n3]])
        for one value per atom, with a last axis for its components, and
        (..., 3, 3) of a stress; NaN for the mean, minimum and maximum of an
        empty bin, and for the temperature of a bin of fewer than two atoms.
    counts : numpy.ndarray
        (n1[, n2[, n3]]) int64 the number of atoms in each bin.
    centers : tuple of numpy.ndarray
        For each axis, (n,) float64 centers of its bins in Å along its box vector:
        from the origin of the box, or with bounds from (0, 0, 0).
    """

    values: np.ndarray
    counts: np.ndarray
    centers: tuple


class SpatialBinning:
    """A property of the atoms in bins of equal size along one, two or three
    box vectors: its mean, sum, sum per volume, minimum or maximum in each;
    or, of moving atoms such as those of a shock, the mean weighted by mass,
    and the temperature and the stress about the center of mass of each
    bin, so that the motion of the bin as a whole is not counted.

    Parameters
    ----------
    column : str
        A column, a component such as "positions.z", or an expression over
        the columns.
    bins : int32 or sequence of int32
        The number of bins along each axis, or one number for all.
    axes : int32, str or sequence of them, optional
        The box vectors binned along: 0, 1, 2 or x, y, z. Default z.
    reduction : str, optional
        One of ``REDUCTIONS``: "mean", "sum", "density" (the sum per Å³ of
        the bin), "min", "max", "mass_mean" (the mean weighted by the mass
        of the atoms), "temperature" (of a column of velocities, in K,
        about the velocity of the center of mass of the bin) or "stress" (of
        a column of the virials of each atom in eV, as calculators give
        them, with the velocities about the center of mass of the bin, in
        GPa, positive under tension). Default mean.
    bounds : sequence of (float64, float64) or None, optional
        For each axis, the range binned, (lo, hi) in Å along its box vector
        from (0, 0, 0), the coordinate itself for a box vector along x, y or
        z; or None for the box. Bins that stay where they are as the box
        changes, as that of molecular dynamics along a direction that is not
        periodic does. Default None: the box along every axis.
    """

    def __init__(
        self,
        column: str,
        bins: int | Sequence[int],
        *,
        axes: int | str | Sequence[int | str] = 2,
        reduction: str = "mean",
        bounds: Sequence[tuple[float, float] | None] | None = None,
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
        ranges = [None] * len(indices) if bounds is None else list(bounds)
        if len(ranges) != len(indices):
            raise InvalidParameter("bounds must give one range or None for each axis")
        for k, pair in enumerate(ranges):
            if pair is None:
                continue
            try:
                lo, hi = (float(v) for v in pair)
            except (TypeError, ValueError):
                raise InvalidParameter(
                    f"each of bounds must be (lo, hi) or None, not {pair!r}"
                ) from None
            if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
                raise InvalidParameter(
                    f"each of bounds must be two finite numbers, lo < hi, not {pair!r}"
                )
            ranges[k] = (lo, hi)
        self.column = str(column)
        self.axes = tuple(indices)
        self.bins = tuple(int(n) for n in counts)
        self.reduction = reduction
        self.bounds = tuple(ranges)

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
        raw = np.asarray(system[self.column], dtype=float)
        if self.reduction == "temperature" and raw.shape[1:] != (3,):
            raise InvalidParameter("a temperature needs a column of (N, 3) velocities")
        if self.reduction == "stress" and raw.shape[1:] != (3, 3):
            raise InvalidParameter("a stress needs a column of (N, 3, 3) virials")
        values = raw.reshape(len(system), int(np.prod(raw.shape[1:])))
        box = system.box
        fractional = box.to_fractional(system["positions"])[:, self.axes]
        periodic = box.pbc[list(self.axes)]
        fractional[:, periodic] -= np.floor(fractional[:, periodic])
        lengths = np.linalg.norm(box.matrix[list(self.axes)], axis=1)
        # with bounds, the place along the vector from (0, 0, 0), as a fraction
        # of the range
        start = box.to_fractional(np.zeros((1, 3)))[0, list(self.axes)]
        spans = lengths.copy()
        for k, pair in enumerate(self.bounds):
            if pair is not None:
                lo, hi = pair
                fractional[:, k] = ((fractional[:, k] - start[k]) * lengths[k] - lo) / (
                    hi - lo
                )
                spans[k] = hi - lo
        inside = ((fractional >= 0) & (fractional < 1)).all(axis=1)
        # the volume of a bin: that of the box times the fraction of each vector
        # its range covers, over the bins
        volume = box.volume * float(np.prod(spans / lengths))
        sizes = np.array(self.bins)
        cell = np.minimum((fractional[inside] * sizes).astype(np.int64), sizes - 1)
        flat = np.ravel_multi_index(tuple(cell.T), self.bins)
        values = values[inside]
        total = int(np.prod(sizes))
        counts = np.bincount(flat, minlength=total)
        k = values.shape[1]
        if self.reduction in _MOVING:
            out = self.__moving(system, inside, flat, values, total, counts, volume)
        elif self.reduction in ("mean", "sum", "density"):
            out = np.stack(
                [np.bincount(flat, values[:, c], minlength=total) for c in range(k)], 1
            )
            if self.reduction == "mean":
                with np.errstate(invalid="ignore", divide="ignore"):
                    out = out / counts[:, None]
            elif self.reduction == "density":
                out = out / (volume / total)
        else:
            out = np.full((total, k), np.nan)
            order = np.argsort(flat, kind="stable")
            filled = np.flatnonzero(counts)
            if len(filled):
                starts = np.concatenate([[0], np.cumsum(counts[filled])[:-1]])
                reduce = np.minimum if self.reduction == "min" else np.maximum
                out[filled] = reduce.reduceat(values[order], starts, axis=0)
        if self.reduction == "stress":
            shape = self.bins + (3, 3)
        elif self.reduction == "temperature":
            shape = self.bins
        else:
            shape = self.bins + ((k,) if raw.ndim > 1 else ())
        centers = tuple(
            (np.arange(n) + 0.5) / n * span + (0.0 if pair is None else pair[0])
            for n, span, pair in zip(self.bins, spans, self.bounds)
        )
        return SpatialBinningResult(
            values=out.reshape(shape),
            counts=counts.reshape(self.bins),
            centers=centers,
        )

    def __moving(
        self,
        system: System,
        inside: np.ndarray,
        flat: np.ndarray,
        values: np.ndarray,
        total: int,
        counts: np.ndarray,
        volume: float,
    ) -> np.ndarray:
        """The mean weighted by mass, or the temperature or the stress about
        the center of mass of each bin, (bins, components)."""
        table = system.table
        if "mass" in table.columns:
            masses = table["mass"].to_numpy().astype(float)
        elif "element" in table.columns:
            masses = atomic_masses(table["element"].to_physical().to_numpy())
        else:
            raise MissingProperty(
                f"the {self.reduction} needs the masses: a mass or an element column"
            )
        masses = masses[inside]
        mass = np.bincount(flat, masses, minlength=total)

        def weighted(data: np.ndarray) -> np.ndarray:
            """The mean of data weighted by mass in each bin."""
            sums = np.stack(
                [
                    np.bincount(flat, masses * data[:, c], minlength=total)
                    for c in range(data.shape[1])
                ],
                1,
            )
            with np.errstate(invalid="ignore", divide="ignore"):
                return sums / mass[:, None]

        if self.reduction == "mass_mean":
            return weighted(values)
        if self.reduction == "temperature":
            velocities = values
        else:
            if "velocities" not in table.columns:
                raise MissingProperty("the stress of moving atoms needs the velocities")
            velocities = np.asarray(system["velocities"], dtype=float)[inside]
        # the velocities about the center of mass of their bin
        relative = velocities - weighted(velocities)[flat]
        if self.reduction == "temperature":
            twice = np.bincount(
                flat,
                masses * np.einsum("ij,ij->i", relative, relative),
                minlength=total,
            )
            # the directions of the atoms of a bin less those of its center of mass
            with np.errstate(invalid="ignore", divide="ignore"):
                kelvin = (
                    twice
                    * units.AMU_A2_PER_PS2_TO_EV
                    / (3 * (counts - 1) * units.BOLTZMANN)
                )
            return np.where(counts > 1, kelvin, np.nan)[:, None]
        kinetic = np.einsum("i,ij,ik->ijk", masses, relative, relative).reshape(-1, 9)
        work = values + kinetic * units.AMU_A2_PER_PS2_TO_EV
        sums = np.stack(
            [np.bincount(flat, work[:, c], minlength=total) for c in range(9)], 1
        )
        return -sums / (volume / total) * units.EV_PER_A3_TO_GPA

    def __repr__(self) -> str:
        return (
            f"SpatialBinning({self.column!r}, {self.bins!r}, axes={self.axes!r}, "
            f"reduction={self.reduction!r}"
            + (
                ""
                if all(b is None for b in self.bounds)
                else f", bounds={self.bounds!r}"
            )
            + ")"
        )
