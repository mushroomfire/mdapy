# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The log of a molecular dynamics or a minimization: a text file or stream
of comment lines, which start with #, and rows of numbers, written as the
work goes and flushed at once, so that the file shows its progress and
reads back as one table, numpy.loadtxt(path)."""

import datetime
import math
import os
import time
from typing import Any, TextIO

from mdapy.errors import InvalidParameter


def _machine() -> str:
    """The processor and the memory of this machine, for the heading of a log."""
    import platform

    from mdapy import _core

    cpu = _core.core.cpu_name() or platform.machine()
    memory = _core.core.memory_bytes()
    text = (
        f"{cpu}, {os.cpu_count()} logical CPUs, {_core.core.hardware_threads()} usable"
    )
    return text + (f", {memory / 2**30:.1f} GB of memory" if memory else "")


def _on_gpu(calculator: Any) -> str | None:
    """The GPU a calculator computes on, its memory and compute capability,
    for the heading of a log; None for one on the processor."""
    if getattr(calculator, "device", "cpu") != "cuda":
        return None
    from mdapy.calculators import _gpu as gpu

    found = gpu.cuda().device_properties()
    major, minor = found["compute_capability"]
    return (
        f"{found['name']}, {found['memory'] / 2**30:.1f} GB, compute capability "
        f"{major}.{minor}, {found['multiprocessors']} multiprocessors"
    )


def _apart(log: Any, calculator: Any) -> None:
    """Raises if the log of a run is the file the calculator writes its own
    log to, such as that of a DFT: their rows differ, and the file would not
    read back as one table."""
    other = getattr(calculator, "log", None)
    paths = (str, os.PathLike)
    if (
        isinstance(log, paths)
        and isinstance(other, paths)
        and os.path.realpath(log) == os.path.realpath(other)
    ):
        raise InvalidParameter(
            f"{os.fspath(log)!r} is also the log of the calculator: give them two files"
        )


class _Log:
    """Where the lines go: a file of a path, opened here, overwritten (or
    added to, for a run continued from a restart) and closed by close; or a
    stream, such as sys.stdout, that stays open. A process that does not
    write (the others of LAMMPS under MPI) passes active=False."""

    def __init__(self, target: Any, *, active: bool = True, append: bool = False):
        self.__owned = False
        self.__stream: TextIO | None = None
        self.__columns: tuple = ()
        self.__start = time.perf_counter()
        if target is None or not active:
            return
        if isinstance(target, (str, os.PathLike)):
            try:
                self.__stream = open(target, "a" if append else "w", encoding="utf-8")
            except OSError as error:
                raise InvalidParameter(f"cannot write the log: {error}") from None
            self.__owned = True
        elif callable(getattr(target, "write", None)):
            self.__stream = target
        else:
            raise InvalidParameter(
                f"log must be a path or a stream with write, not {target!r}"
            )

    @property
    def on(self) -> bool:
        return self.__stream is not None

    def __write(self, text: str) -> None:
        if self.__stream is None:
            return
        self.__stream.write(text)
        self.__stream.flush()

    def comment(self, *lines: str) -> None:
        self.__write("".join(f"# {line}\n" if line else "#\n" for line in lines))

    def title(self, what: str) -> None:
        """The first line: MDAPY, its version, what runs, and when."""
        from mdapy import __version__

        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.comment(f"MDAPY {__version__}, {what}, {now}")

    def columns(self, columns: tuple) -> None:
        """The heading of the rows to come, (name, format, width) of each
        column, wall added: the seconds since then."""
        self.__columns = (*columns, ("wall", "{:.2f}", 10))
        heading = "".join(f"{name:>{width}}" for name, _, width in self.__columns)
        self.__write("#" + heading[1:] + "\n")
        self.__start = time.perf_counter()

    @property
    def wall(self) -> float:
        """Seconds since the heading of the rows."""
        return time.perf_counter() - self.__start

    def row(self, *values: Any) -> None:
        if self.__stream is None:
            return
        cells = []
        for (_, form, width), value in zip(self.__columns, (*values, self.wall)):
            nan = isinstance(value, float) and math.isnan(value)
            cells.append(f"{'nan' if nan else form.format(value):>{width}}")
        self.__write("".join(cells) + "\n")

    def close(self) -> None:
        if self.__owned and self.__stream is not None:
            self.__stream.close()
        self.__stream = None
