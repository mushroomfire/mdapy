# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The drift of the center of mass removed from a trajectory."""

import operator
from collections.abc import Sequence
from typing import Any, overload

import numpy as np

from mdapy.analysis._bonds import _weights
from mdapy.analysis._dynamics import _follow
from mdapy.core._atoms import _mask
from mdapy.core.system import System
from mdapy.errors import InvalidParameter


class _Translated(Sequence[System]):
    """Frames, each moved by its own translation when it is used."""

    def __init__(self, frames: Sequence[System], shifts: np.ndarray):
        self.__frames = frames
        self.__shifts = shifts
        self.__shifts.flags.writeable = False

    @property
    def shifts(self) -> np.ndarray:
        """(T, 3) the translation of each frame in Å."""
        return self.__shifts

    def __len__(self) -> int:
        return len(self.__shifts)

    @overload
    def __getitem__(self, index: int) -> System: ...

    @overload
    def __getitem__(self, index: slice) -> "_Translated": ...

    def __getitem__(self, index):
        if isinstance(index, slice):
            return _Translated(self.__frames[index], self.__shifts[index])
        index = operator.index(index)
        shift = self.__shifts[index]  # raises IndexError
        frame = self.__frames[index]
        return frame.with_columns(positions=frame["positions"] + shift)


def remove_drift(
    frames: Sequence[System],
    reference: int = 0,
    *,
    atoms: Any = None,
    unwrap: bool = True,
) -> Sequence[System]:
    """The frames of a trajectory moved so that the center of mass of the
    atoms stays where it is in a reference frame, as OVITO's center of mass
    correction.

    Parameters
    ----------
    frames : sequence of System
        The same atoms in every frame, such as a Trajectory: matched by the
        ``id`` column where the first frame has one, else in the same order.
        The masses are the ``mass`` column of the first frame, else those of
        its elements, else all equal.
    reference : int, optional
        The frame whose center of mass is kept. Default 0.
    atoms : array_like, optional
        A boolean mask or indices of the atoms of the first frame whose
        center of mass is kept. Default all.
    unwrap : bool, optional
        Follow the atoms through the periodic boundaries to find the center
        of mass (True, default), or take the positions as they are.

    Returns
    -------
    sequence of System
        The frames, read when they are used, each with all its atoms moved
        by one translation; the boxes do not move. Its ``shifts`` are the
        (T, 3) translations in Å.

    Raises
    ------
    InvalidParameter
        If there are no frames, the reference frame is not one of them, no
        atoms are chosen or their masses add up to zero, or the frames do
        not hold the same atoms.
    """
    count = len(frames)
    if count == 0:
        raise InvalidParameter("there are no frames")
    try:
        reference = range(count)[operator.index(reference)]
    except IndexError:
        raise InvalidParameter(
            f"reference frame {reference} is not among the {count} frames"
        ) from None
    weights = _weights(frames[0])
    if atoms is not None:
        weights = np.where(_mask(atoms, len(weights)), weights, 0.0)
    total = weights.sum()
    if total <= 0:
        raise InvalidParameter("the chosen atoms have no mass")
    centers = np.array([weights @ x / total for x in _follow(frames, unwrap)])
    return _Translated(frames, centers[reference] - centers)
