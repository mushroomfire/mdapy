# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Trajectories: the frames of files, read when they are used."""

import glob
import operator
import os
import re
import weakref
from collections.abc import Iterator, Sequence
from typing import Any, overload

import numpy as np

from mdapy.core.box import Box
from mdapy.core.system import System
from mdapy.errors import InvalidParameter
from mdapy.io._files import Source
from mdapy.io._registry import Format, FramePosition, _format


def _close_all(sources: dict) -> None:
    for source in sources.values():
        source.close()
    sources.clear()


class _Frames:
    """Frame positions of files, found as far as needed: the file of each
    frame and its position there. At most two files are open at a time,
    that being scanned and that last read."""

    def __init__(self, name: str, paths: list[str], found: Format):
        self.name = name
        self.paths = paths
        self.format = found
        self.__open: dict[int, Source] = {}
        self.__scanning = 0
        # The files are released when no trajectory uses them any more.
        self.close = weakref.finalize(self, _close_all, self.__open)
        self.source(0)  # a file that cannot be opened fails at once
        self.positions: list[tuple[int, FramePosition]] = []
        self._scanner: Iterator[tuple[int, FramePosition]] | None = self.__scan()

    def source(self, file: int) -> Source:
        """The open file of this index, opened if need be."""
        if file not in self.__open:
            self.__open[file] = Source(self.paths[file])
        for other in [k for k in self.__open if k not in (file, self.__scanning)]:
            self.__open.pop(other).close()
        return self.__open[file]

    def __scan(self) -> Iterator[tuple[int, FramePosition]]:
        for file in range(len(self.paths)):
            self.__scanning = file
            if self.format.scan is None:
                yield file, (0, 0)
            else:
                for position in self.format.scan(self.source(file)):
                    yield file, position

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

    def __init__(
        self,
        frames: _Frames,
        options: dict[str, Any],
        indices: range | None,
        topology: System | None = None,
    ):
        self._frames = frames
        self.__options = options
        self.__indices = indices  # None: every frame of the files
        self.__topology = topology

    @property
    def path(self) -> str:
        """Name of the file, or the pattern or first file of several."""
        return self._frames.name

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
            return Trajectory(self._frames, self.__options, selected, self.__topology)
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
        file, position = self._frames.positions[frame]
        source = self._frames.source(file)
        system = self._frames.format.read(source, position, **self.__options)
        if self.__topology is not None:
            system = _with_topology(system, self.__topology, source.name)
        return system

    def close(self) -> None:
        """Releases the files, also for the slices of this trajectory."""
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


def _natural(name: str) -> list:
    """Sorts names by the numbers in them: dump.2 before dump.10."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def _paths(path: Any) -> tuple[str, list[str]]:
    """The name and the files of a file, a pattern or a list of files."""
    if isinstance(path, (str, os.PathLike)):
        name = os.fspath(path)
        if "*" not in name:
            return name, [name]
        files = sorted(glob.glob(name), key=_natural)
        if not files:
            raise FileNotFoundError(f"no file matches {name}")
        return name, files
    files = [os.fspath(p) for p in path]
    if not files:
        raise InvalidParameter("give at least one file")
    return files[0], files


def _with_topology(frame: System, topology: System, name: str) -> System:
    """The frame in the order of the atoms of the topology, with their
    topology and the columns the frame lacks; atoms missing from the frame
    are left out, their topology terms with them."""
    n, has = len(topology), "id" in topology.table.columns
    if has and "id" in frame.table.columns:
        known, ids = np.asarray(topology["id"]), np.asarray(frame["id"])
        if len(np.unique(known)) != n:
            raise InvalidParameter("the atoms of the topology repeat ids")
        if len(np.unique(ids)) != len(ids):
            raise InvalidParameter(f"the atoms of a frame of {name} repeat ids")
        order = np.argsort(ids, kind="stable")
        at = np.minimum(np.searchsorted(ids[order], known), len(ids) - 1)
        found = ids[order][at] == known if len(ids) else np.zeros(n, dtype=bool)
        if found.sum() < len(ids):
            extra = np.setdiff1d(ids, known)
            raise InvalidParameter(
                f"a frame of {name} has {len(extra)} atoms not in the topology, "
                f"such as id {extra[0]}"
            )
        rows, kept = order[at[found]], np.flatnonzero(found)
    elif len(frame) != n:
        raise InvalidParameter(
            f"a frame of {name} has {len(frame)} atoms and the topology {n}: "
            "without ids they must be the same"
        )
    elif "id" in frame.table.columns:
        raise InvalidParameter(
            "the atoms of the frames have ids and those of the topology not"
        )
    elif has and not np.array_equal(np.asarray(topology["id"]), np.arange(1, n + 1)):
        raise InvalidParameter(
            f"the frames of {name} have no ids, and those of the topology are "
            "not 1 to N in order"
        )
    else:
        rows = kept = np.arange(n)
    system = topology if len(kept) == n else topology.gather(kept)
    columns = frame.table[rows].drop("id", strict=False)
    if "image" in system.table.columns and "image" not in columns.columns:
        system = system.drop("image")  # the flags of the topology are stale
    box = Box(frame.box.matrix, frame.box.origin, topology.box.pbc)
    return (
        system.with_columns(*columns.get_columns())
        .with_box(box)
        .with_info({**topology.info, **frame.info})
    )


def read_trajectory(
    path: str | os.PathLike | Sequence[str | os.PathLike],
    format: str | None = None,
    *,
    topology: System | None = None,
    **options,
) -> Trajectory:
    """Opens the frames of files without reading them.

    Parameters
    ----------
    path : str, os.PathLike or sequence of them
        The file; a pattern with ``*``, such as ``"dump.*.lammpstrj"``,
        for the files it matches in the order of the numbers in their
        names; or files in order. The frames of each file follow those of
        the file before.
    format : str, optional
        Format name; by default it is found from the name of the first file
        with :func:`detect_format`.
    topology : System, optional
        A system, such as one read from a LAMMPS data file, whose atoms the
        frames move: each frame gets its atoms, matched by id, in their
        order, with their topology and the columns the frame lacks, such as
        charges, molecules and masses; the box of the frame with the
        periodicity of the system. Atoms missing from a frame are left out
        with their bonds, angles and so on.
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
    FileNotFoundError
        If no file matches a pattern.
    InvalidParameter
        For an empty list of files; on reading a frame with a topology, if
        it has atoms not in the topology, if ids repeat, or if without ids
        the numbers of atoms differ.
    """
    name, paths = _paths(path)
    found = _format(paths[0], format, "read")
    return Trajectory(_Frames(name, paths, found), options, None, topology)
