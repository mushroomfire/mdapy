# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Analyses of trajectories: mean squared displacement and Lindemann index."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from mdapy.core.result import Result
from mdapy.core.system import System
from mdapy.errors import InvalidParameter


def unwrapped(frames: Sequence[System]) -> np.ndarray:
    """(T, N, 3) positions of the frames, each moved to the periodic image
    nearest to its position in the previous frame (atoms move less than half
    the box between frames)."""
    count = len(frames)
    if count == 0:
        raise InvalidParameter("there are no frames")
    first = frames[0]
    positions = np.empty((count, len(first), 3))
    positions[0] = first.positions
    previous = first
    for t in range(1, count):
        frame = frames[t]
        if len(frame) != len(first):
            raise InvalidParameter(
                f"frame {t} has {len(frame)} atoms, frame 0 has {len(first)}"
            )
        # the step in the fractional coordinates of this frame, by the nearest image
        step = frame.box.to_fractional(frame.positions) - frame.box.to_fractional(
            previous.positions
        )
        periodic = frame.box.pbc
        step[:, periodic] -= np.round(step[:, periodic])
        positions[t] = positions[t - 1] + step @ frame.box.matrix
        previous = frame
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
    """

    def __init__(self, *, window: bool = True):
        self.window = bool(window)

    def compute(self, frames: Sequence[System]) -> DisplacementResult:
        """The displacement of a sequence of frames, such as a Trajectory.

        Parameters
        ----------
        frames : sequence of System
            The same atoms in the same order in every frame, moving less than
            half the box between frames.

        Returns
        -------
        DisplacementResult
            The mean squared displacement for each lag.
        """
        r = unwrapped(frames)  # (T, N, 3)
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
        return f"MeanSquaredDisplacement(window={self.window})"


@dataclass(frozen=True)
class LindemannResult(Result):
    """Lindemann index of a trajectory.

    Attributes
    ----------
    index : float
        The index averaged over every pair of atoms.
    atomic : numpy.ndarray
        (N,) the index of each atom, averaged over its pairs.
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
    relative to their mean, which rises sharply at melting."""

    def compute(self, frames: Sequence[System]) -> LindemannResult:
        """The index of a sequence of frames.

        Parameters
        ----------
        frames : sequence of System
            The same atoms in the same order in every frame, moving less than
            half the box between frames. Memory grows as the square of the
            number of atoms.

        Returns
        -------
        LindemannResult
            The index over every pair, of each atom and of each prefix of the
            frames.
        """
        r = unwrapped(frames)
        count, n = r.shape[0], r.shape[1]
        if n < 2:
            raise InvalidParameter("the Lindemann index needs two atoms or more")
        first, second = np.triu_indices(n, 1)
        # Welford's running mean and sum of squared deviations of each distance
        mean = np.zeros(len(first))
        deviations = np.zeros(len(first))
        running = np.zeros(count)
        for t in range(count):
            d = np.linalg.norm(r[t][first] - r[t][second], axis=1)
            delta = d - mean
            mean += delta / (t + 1)
            deviations += delta * (d - mean)
            ratio = np.sqrt(deviations / (t + 1)) / mean
            running[t] = ratio.mean()
        atomic = (
            np.bincount(first, weights=ratio, minlength=n)
            + np.bincount(second, weights=ratio, minlength=n)
        ) / (n - 1)
        return LindemannResult(index=float(running[-1]), atomic=atomic, running=running)

    def __repr__(self) -> str:
        return "LindemannIndex()"
