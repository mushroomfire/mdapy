# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Analyses of trajectories: mean squared displacement and Lindemann index."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mdapy._core import analysis as _native
from mdapy.core.result import Result
from mdapy.core.system import System
from mdapy.core.threads import _native_token
from mdapy.errors import InvalidParameter


def _followed(frames: Sequence[System], unwrap: bool) -> np.ndarray:
    """(T, N, 3) positions of the frames, the atoms in the order of the first
    frame, matched by their ids where it has an id column. With unwrap, each
    atom is followed from frame to frame by its image flags where both frames
    have an image column, else by the periodic image nearest to its previous
    position."""
    count = len(frames)
    if count == 0:
        raise InvalidParameter("there are no frames")
    first = frames[0]
    n = len(first)
    ids = first.table["id"].to_numpy() if "id" in first.table.columns else None
    if ids is not None:
        first_order = np.argsort(ids, kind="stable")
        sorted_ids = ids[first_order]
        if (sorted_ids[1:] == sorted_ids[:-1]).any():
            raise InvalidParameter("frame 0 has repeated ids")
    positions = np.empty((count, n, 3))
    previous = before = None
    for t in range(count):
        frame = frames[t]
        if len(frame) != n:
            raise InvalidParameter(f"frame {t} has {len(frame)} atoms, frame 0 has {n}")
        table = frame.table
        x = frame["positions"]
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
            positions[t] = x
            continue
        now = None if image is None else x + image @ frame.box.matrix
        if t == 0:
            positions[0] = x if now is None else now
        elif now is not None and before is not None:
            positions[t] = positions[t - 1] + (now - before)
        else:
            # the step in the fractional coordinates of this frame, by the
            # nearest image
            step = frame.box.to_fractional(x) - frame.box.to_fractional(previous)
            periodic = frame.box.pbc
            step[:, periodic] -= np.round(step[:, periodic])
            positions[t] = positions[t - 1] + step @ frame.box.matrix
        previous, before = x, now
    return positions


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
        r = _followed(frames, self.unwrap)  # (T, N, 3)
        if r.shape[1] == 0:
            raise InvalidParameter("the frames have no atoms")
        count = len(r)
        if not self.window:
            parts = ((r - r[0]) ** 2).mean(axis=1)
        else:
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
            parts = (s1 - 2.0 * s2).mean(axis=1)
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
