# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Columns carried from frame to frame: values frozen at a frame, and
positions smoothed over neighboring frames."""

import operator
from collections.abc import Sequence
from typing import overload

import numpy as np
import polars as pl

from mdapy.analysis._dynamics import _follow
from mdapy.core._atoms import _matching
from mdapy.core.system import System
from mdapy.errors import InvalidParameter


def freeze(
    system: System, reference: System, column: str, name: str | None = None
) -> System:
    """The values of a column in a reference frame given to the atoms of a
    system, as OVITO's freeze property.

    Parameters
    ----------
    system : System
        The frame that takes the values.
    reference : System
        The frame of the values, such as the first of the trajectory: its
        atoms matched by the ``id`` column where both have one, else in the
        same order.
    column : str
        A column, a component such as ``"positions.z"`` or an expression
        over the columns.
    name : str, optional
        The new column; by default the column, its dots as underscores,
        followed by ``_frozen``.

    Returns
    -------
    System
        The system with the values of the reference in the new column.

    Raises
    ------
    MissingProperty
        If an atom of the system is not in the reference.
    InvalidParameter
        Without ids, if the frames have different numbers of atoms.
    """
    match = _matching(reference, system)
    values = np.asarray(reference[column])[match]
    return system.with_columns(**{name or f"{column.replace('.', '_')}_frozen": values})


class _Smoothed(Sequence[System]):
    """Frames, each with its positions averaged over a window of frames
    about it when it is used."""

    def __init__(self, frames: Sequence[System], window: int):
        self.__frames = frames
        self.__window = window

    def __len__(self) -> int:
        return len(self.__frames)

    @overload
    def __getitem__(self, index: int) -> System: ...

    @overload
    def __getitem__(self, index: slice) -> list[System]: ...

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[k] for k in range(len(self))[index]]
        count = len(self.__frames)
        k = range(count)[operator.index(index)]  # raises IndexError
        before = (self.__window - 1) // 2
        chosen = [
            min(max(k + offset, 0), count - 1)
            for offset in range(-before, self.__window // 2 + 1)
        ]
        window = [self.__frames[i] for i in chosen]
        followed = list(_follow(window, True))
        current = window[before]
        # the mean placed about the atoms where they are in this frame
        first = _matching(window[0], current)
        mean = np.mean(followed, axis=0) - followed[before]
        averaged = {"positions": current["positions"] + mean[first]}
        # every other column of real numbers averaged as it is
        for name, dtype in current.table.schema.items():
            inner = dtype
            while isinstance(inner, (pl.Array, pl.List)):
                inner = inner.inner
            if name == "positions" or not inner.is_float():
                continue
            averaged[name] = np.mean(
                [
                    np.asarray(frame[name])[_matching(frame, current)]
                    for frame in window
                ],
                axis=0,
            )
        return current.with_columns(**averaged)


def smooth_trajectory(frames: Sequence[System], window: int) -> Sequence[System]:
    """The positions and other real numbers of each frame averaged over a
    window of frames about it, the atoms followed through the periodic
    boundaries, as OVITO's smooth trajectory.

    Parameters
    ----------
    frames : sequence of System
        The same atoms in every frame, such as a Trajectory: matched by the
        ``id`` column where the frames have one, else in the same order.
        Without an ``image`` column, atoms must move less than half the box
        between frames.
    window : int
        The number of frames averaged: the frame, (window - 1) // 2 frames
        before it and window // 2 after it, the first and last frames
        repeated beyond the ends; 1 leaves the positions as they are.

    Returns
    -------
    sequence of System
        The frames, each computed when it is used: the mean positions about
        the atoms of that frame, the mean of every other column of real
        numbers (velocities, forces, values of analyses) matched by atom;
        columns of integers or text as they are.

    Raises
    ------
    InvalidParameter
        If the window is not a positive integer or there are no frames.
    """
    if not isinstance(window, (int, np.integer)) or window < 1:
        raise InvalidParameter(f"window must be a positive integer, not {window!r}")
    if len(frames) == 0:
        raise InvalidParameter("there are no frames")
    return _Smoothed(frames, int(window))
