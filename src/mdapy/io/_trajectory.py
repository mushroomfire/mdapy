# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Trajectories: the frames of a file, read when they are used."""

import operator
import os
import weakref
from collections.abc import Iterator, Sequence
from typing import Any, overload

from mdapy.core.system import System
from mdapy.io._files import Source
from mdapy.io._registry import Format, FramePosition, _format


class _Frames:
    """Frame positions of an open file, found as far as needed."""

    def __init__(self, source: Source, found: Format):
        self.source = source
        self.format = found
        # The file is released when no trajectory uses it any more.
        self.close = weakref.finalize(self, source.close)
        self.positions: list[FramePosition] = []
        if found.scan is None:
            self.positions.append((0, 0))
            self._scanner: Iterator[FramePosition] | None = None
        else:
            self._scanner = found.scan(source)

    def known(self, count: int) -> bool:
        """Whether at least count frames exist; scans as far as needed."""
        while len(self.positions) < count and self._scanner is not None:
            position = next(self._scanner, None)
            if position is None:
                self._scanner = None
            else:
                self.positions.append(position)
        return len(self.positions) >= count

    def total(self) -> int:
        while self.known(len(self.positions) + 1):
            pass
        return len(self.positions)


class Trajectory(Sequence[System]):
    """The frames of a file, read when they are used.

    Create it with :func:`mdapy.read_trajectory`. Indexing with an integer
    reads one frame; indexing with a slice gives a trajectory of the selected
    frames; iterating reads the frames one after the other. The file stays
    open until :meth:`close` is called, the trajectory is used as a context
    manager, or no trajectory of the file is used any more.
    """

    def __init__(self, frames: _Frames, options: dict[str, Any], indices: range | None):
        self._frames = frames
        self.__options = options
        self.__indices = indices  # None: every frame of the file

    @property
    def path(self) -> str:
        """Name of the file."""
        return self._frames.source.name

    @property
    def format(self) -> str:
        """Format of the file."""
        return self._frames.format.name

    def __len__(self) -> int:
        if self.__indices is not None:
            return len(self.__indices)
        return self._frames.total()

    @overload
    def __getitem__(self, index: int) -> System: ...

    @overload
    def __getitem__(self, index: slice) -> "Trajectory": ...

    def __getitem__(self, index):
        if isinstance(index, slice):
            if self.__indices is None:
                selected = range(self._frames.total())[index]
            else:
                selected = self.__indices[index]
            return Trajectory(self._frames, self.__options, selected)
        index = operator.index(index)
        if self.__indices is not None:
            frame = self.__indices[index]  # raises IndexError
        elif index >= 0:
            if not self._frames.known(index + 1):
                raise IndexError(f"frame {index} is out of range")
            frame = index
        else:
            frame = self._frames.total() + index
            if frame < 0:
                raise IndexError(f"frame {index} is out of range")
        return self.__read(frame)

    def __iter__(self) -> Iterator[System]:
        if self.__indices is not None:
            for frame in self.__indices:
                yield self.__read(frame)
            return
        frame = 0
        while self._frames.known(frame + 1):
            yield self.__read(frame)
            frame += 1

    def __read(self, frame: int) -> System:
        position = self._frames.positions[frame]
        return self._frames.format.read(self._frames.source, position, **self.__options)

    def close(self) -> None:
        """Releases the file, also for the slices of this trajectory."""
        self._frames.close()

    def __enter__(self) -> "Trajectory":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        count = (
            str(len(self.__indices))
            if self.__indices is not None
            else str(len(self._frames.positions))
            + ("" if self._frames._scanner is None else "+")
        )
        return f"Trajectory({self.path!r}, format={self.format!r}, frames={count})"


def read_trajectory(
    path: str | os.PathLike, format: str | None = None, **options
) -> Trajectory:
    """Opens the frames of a file without reading them.

    Parameters
    ----------
    path : str or os.PathLike
        The file.
    format : str, optional
        Format name; by default it is found from the file name with
        :func:`detect_format`.
    **options
        Options of the format, used for every frame.

    Returns
    -------
    Trajectory
        The frames. Frames are found as far as they are used, so iterating
        over a long file starts at once.

    Raises
    ------
    UnsupportedFormat
        If the format is unknown or cannot be read.
    """
    found = _format(path, format, "read")
    return Trajectory(_Frames(Source(path), found), options, None)
