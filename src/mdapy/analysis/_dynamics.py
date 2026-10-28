# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Analyses of trajectories: mean squared displacement, Lindemann index
and time averages."""

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mdapy._core import analysis as _native
from mdapy.core._atoms import _mask
from mdapy.core.result import Result
from mdapy.core.system import System
from mdapy.core.threads import _native_token
from mdapy.errors import InvalidParameter


#: Values of the positions over the frames transformed together by the
#: mean squared displacement, some atoms at a time.
_BLOCK = 1 << 20


def _follow(
    frames: Iterable[System], unwrap: bool, column: str = "positions"
) -> Iterator[np.ndarray]:
    """The values of a column in each frame, the atoms in the order of the
    first frame, matched by their ids where it has an id column. With
    unwrap, positions follow each atom from frame to frame by its image
    flags where both frames have an image column, else by the periodic image
    nearest to its previous position."""
    unwrap = unwrap and column == "positions"
    n = ids = None
    last = previous = before = None
    for t, frame in enumerate(frames):
        table = frame.table
        if t == 0:
            n = len(frame)
            if "id" in table.columns:
                ids = table["id"].to_numpy()
                first_order = np.argsort(ids, kind="stable")
                sorted_ids = ids[first_order]
                if (sorted_ids[1:] == sorted_ids[:-1]).any():
                    raise InvalidParameter("frame 0 has repeated ids")
        if len(frame) != n:
            raise InvalidParameter(f"frame {t} has {len(frame)} atoms, frame 0 has {n}")
        try:
            x = np.asarray(frame[column], dtype=np.float64)
        except (TypeError, ValueError):
            raise InvalidParameter(f"{column} is not numbers") from None
        image = table["image"].to_numpy() if "image" in table.columns else None
        if ids is not None and t > 0:
            if "id" not in table.columns:
                raise InvalidParameter(f"frame {t} has no id column, frame 0 has one")
            mine = table["id"].to_numpy()
            order = np.argsort(mine, kind="stable")
            if not np.array_equal(mine[order], sorted_ids):
                raise InvalidParameter(f"frame {t} has other ids than frame 0")
            # the atom of frame t with the id of each atom of frame 0
            match = np.empty(n, dtype=np.int64)
            match[first_order] = order
            x = x[match]
            image = None if image is None else image[match]
        if not unwrap:
            yield x
            continue
        now = None if image is None else x + image @ frame.box.matrix
        if t == 0:
            last = x if now is None else now
        elif now is not None and before is not None:
            last = last + (now - before)
        else:
            # the step in the fractional coordinates of this frame, by the
            # nearest image
            step = frame.box.to_fractional(x) - frame.box.to_fractional(previous)
            periodic = frame.box.pbc
            step[:, periodic] -= np.round(step[:, periodic])
            last = last + step @ frame.box.matrix
        previous, before = x, now
        yield last
    if n is None:
        raise InvalidParameter("there are no frames")


def _followed(frames: Iterable[System], unwrap: bool) -> np.ndarray:
    """(T, N, 3) positions of the frames, as _follow gives them."""
    return np.stack(list(_follow(frames, unwrap)))


@dataclass(frozen=True)
class DisplacementResult(Result):
    """Mean squared displacement against the lag.

    Attributes
    ----------
    lag : numpy.ndarray
        (T,) lag in frames.
    msd : numpy.ndarray
        (T,) mean squared displacement in Å², averaged over the atoms.
    components : numpy.ndarray
        (T, 3) its x, y and z parts.
    """

    lag: np.ndarray
    msd: np.ndarray
    components: np.ndarray


class MeanSquaredDisplacement:
    """Mean squared displacement of the atoms of a trajectory.

    Parameters
    ----------
    window : bool, optional
        Average over every time origin, by fast Fourier transforms (True,
        default), or measure from the first frame only (False).
    unwrap : bool, optional
        Follow the atoms through the periodic boundaries (True, default), or
        take the positions as they are, already unwrapped (False).
    """

    def __init__(self, *, window: bool = True, unwrap: bool = True):
        self.window = bool(window)
        self.unwrap = bool(unwrap)

    def compute(self, frames: Sequence[System]) -> DisplacementResult:
        """The displacement of a sequence of frames, such as a Trajectory.

        Parameters
        ----------
        frames : sequence of System
            The same atoms in every frame: matched by the ``id`` column where
            the first frame has one, else in the same order. With ``unwrap``
            and without an ``image`` column, atoms must move less than half
            the box between frames.

        Returns
        -------
        DisplacementResult
            The mean squared displacement for each lag.
        """
        steps = list(_follow(frames, self.unwrap))  # (N, 3) in each frame
        count, n = len(steps), len(steps[0])
        if n == 0:
            raise InvalidParameter("the frames have no atoms")
        # some atoms at a time, so that the Fourier transforms stay small
        block = max(1, _BLOCK // count)
        parts = np.zeros((count, 3))
        for first in range(0, n, block):
            r = np.stack([x[first : first + block] for x in steps])  # (T, b, 3)
            if not self.window:
                parts += ((r - r[0]) ** 2).sum(axis=1)
                continue
            # MSD(m) = S1(m) - 2 S2(m) per atom and direction, with S2 the
            # autocorrelation of the positions from their Fourier transform
            size = 2 * count
            spectrum = np.fft.rfft(r, n=size, axis=0)
            auto = np.fft.irfft(spectrum * np.conj(spectrum), n=size, axis=0)[:count]
            s2 = auto / (count - np.arange(count))[:, None, None]
            square = r * r
            total = 2.0 * square.sum(axis=0)
            s1 = np.empty_like(s2)
            for m in range(count):
                if m > 0:
                    total = total - square[m - 1] - square[count - m]
                s1[m] = total / (count - m)
            parts += (s1 - 2.0 * s2).sum(axis=1)
        parts /= n
        return DisplacementResult(
            lag=np.arange(count), msd=parts.sum(axis=1), components=parts
        )

    def __repr__(self) -> str:
        return f"MeanSquaredDisplacement(window={self.window}, unwrap={self.unwrap})"


@dataclass(frozen=True)
class LindemannResult(Result):
    """Lindemann index of a trajectory.

    Attributes
    ----------
    index : float
        The index averaged over every pair of atoms.
    atomic : numpy.ndarray
        (N,) the index of each atom, averaged over its pairs, the atoms in
        the order of the first frame.
    running : numpy.ndarray
        (T,) the index of the frames up to each one.
    """

    index: float
    atomic: np.ndarray
    running: np.ndarray

    def columns(self) -> dict[str, np.ndarray]:
        return {"lindemann": self.atomic}


class LindemannIndex:
    """Lindemann index: the fluctuation of the distances between atoms
    relative to their mean, which rises sharply at melting.

    Parameters
    ----------
    unwrap : bool, optional
        Follow the atoms through the periodic boundaries (True, default), or
        take the positions as they are, already unwrapped (False).
    """

    def __init__(self, *, unwrap: bool = True):
        self.unwrap = bool(unwrap)

    def compute(
        self, frames: Sequence[System], *, progress: Any = None, cancel: Any = None
    ) -> LindemannResult:
        """The index of a sequence of frames.

        Parameters
        ----------
        frames : sequence of System
            The same atoms in every frame: matched by the ``id`` column where
            the first frame has one, else in the same order. With ``unwrap``
            and without an ``image`` column, atoms must move less than half
            the box between frames.
            Time grows as the square of the number of atoms.

        Returns
        -------
        LindemannResult
            The index over every pair, of each atom and of each prefix of the
            frames.
        """
        r = _followed(frames, self.unwrap)
        if r.shape[1] < 2:
            raise InvalidParameter("the Lindemann index needs two atoms or more")
        atomic, running = _native.lindemann(
            np.ascontiguousarray(r.transpose(1, 0, 2)), progress, _native_token(cancel)
        )
        return LindemannResult(index=float(running[-1]), atomic=atomic, running=running)

    def __repr__(self) -> str:
        return f"LindemannIndex(unwrap={self.unwrap})"


def trajectory_lines(
    frames: Iterable[System], atoms: Any = None, *, unwrap: bool = True
) -> np.ndarray:
    """The paths of atoms over the frames of a trajectory, as OVITO's
    generate trajectory lines.

    Parameters
    ----------
    frames : iterable of System
        The same atoms in every frame: matched by the ``id`` column where
        the first frame has one, else in the same order.
    atoms : array_like, optional
        A boolean mask or indices of the atoms of the first frame whose
        paths are wanted. Default all.
    unwrap : bool, optional
        Follow the atoms through the periodic boundaries (True, default), or
        take the positions as they are.

    Returns
    -------
    numpy.ndarray
        (M, T, 3) the positions of each atom in each frame, the atoms in the
        order of the first frame.
    """
    chosen = None
    paths = []
    for positions in _follow(frames, unwrap):
        if chosen is None:
            n = len(positions)
            chosen = np.arange(n) if atoms is None else np.flatnonzero(_mask(atoms, n))
        paths.append(positions[chosen])
    return np.ascontiguousarray(np.stack(paths, axis=1))


@dataclass(frozen=True)
class TimeAverageResult(Result):
    """Time average of a quantity.

    Attributes
    ----------
    value : numpy.ndarray
        The mean over the frames: of each atom in the order of the first
        frame for a column, (3, 3) for the box, of the shape the function
        gives otherwise.
    frames : int
        The number of frames.
    name : str or None
        The column averaged; None for the box or a function.
    """

    value: np.ndarray
    frames: int
    name: str | None = None

    def columns(self) -> dict[str, np.ndarray]:
        return {} if self.name is None else {f"{self.name}_average": self.value}


class TimeAverage:
    """The mean of a quantity over the frames of a trajectory, as OVITO's
    time averaging modifier.

    Parameters
    ----------
    quantity : str or callable
        A column of numbers of the atoms, such as ``"positions"`` or
        ``"csp"``; ``"box"`` for the lattice vectors; or a function of a frame
        that gives a number, or an array of the same shape in every frame,
        such as the g(r) of a radial distribution function.
    unwrap : bool, optional
        For positions: follow the atoms through the periodic boundaries
        (True, default), or average the positions as they are.
    """

    def __init__(self, quantity: str | Callable[[System], Any], *, unwrap: bool = True):
        if not (isinstance(quantity, str) or callable(quantity)):
            raise InvalidParameter(
                f"quantity must be a column name, box or a function, not {quantity!r}"
            )
        self.quantity = quantity
        self.unwrap = bool(unwrap)

    def compute(self, frames: Iterable[System]) -> TimeAverageResult:
        """The mean over frames, such as a Trajectory or a slice of one.

        Parameters
        ----------
        frames : iterable of System
            For a column, the same atoms in every frame: matched by the
            ``id`` column where the first frame has one, else in the same
            order. With ``unwrap`` and without an ``image`` column, atoms
            must move less than half the box between frames.

        Returns
        -------
        TimeAverageResult
            The mean and the number of frames.
        """
        quantity = self.quantity
        if quantity == "box":
            values = (frame.box.matrix for frame in frames)
        elif isinstance(quantity, str):
            values = _follow(frames, self.unwrap, quantity)
        else:
            values = (quantity(frame) for frame in frames)
        total, count = None, 0
        for value in values:
            value = np.asarray(value, dtype=np.float64)
            if total is None:
                total = value.copy()
            elif value.shape != total.shape:
                raise InvalidParameter(
                    f"frame {count} gives shape {value.shape}, frame 0 {total.shape}"
                )
            else:
                total += value
            count += 1
        if total is None:
            raise InvalidParameter("there are no frames")
        name = quantity if isinstance(quantity, str) and quantity != "box" else None
        return TimeAverageResult(value=total / count, frames=count, name=name)

    def __repr__(self) -> str:
        return f"TimeAverage({self.quantity!r}, unwrap={self.unwrap})"
