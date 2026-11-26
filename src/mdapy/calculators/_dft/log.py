# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The log of a DFT calculator: every step of the self-consistent field of
each of its computations, in the form of the logs of the simulations (comment
lines that start with #, rows of numbers that numpy.loadtxt reads as one
table), with the parameters of each computation as one line of JSON."""

import json
import math
import os
from typing import Any

from mdapy.errors import InvalidParameter

# (name, format, width) of the columns before wall, the seconds since the
# start of the computation
COLUMNS = (
    ("calculation", "{:d}", 13),
    ("step", "{:d}", 6),
    ("free_energy", "{:.8f}", 20),
    ("change", "{:.2e}", 11),
    ("residual", "{:.2e}", 11),
    ("magnetization", "{:.4f}", 15),
    ("absolute", "{:.4f}", 10),
    ("fermi", "{:.4f}", 11),
    ("davidson", "{:d}", 10),
)


def check_target(target: Any) -> None:
    """Raises unless target is None, a path or a stream with write."""
    if target is None or isinstance(target, (str, os.PathLike)):
        return
    if not callable(getattr(target, "write", None)):
        raise InvalidParameter(
            f"log must be a path or a stream with write, not {target!r}"
        )


class Writer:
    """The log of the computations of one calculator: a file of a path,
    written anew by the first computation and added to by the next ones, or a
    stream such as sys.stdout. A process that does not write (all but the
    first under MPI) passes active=False. heading, the lines of the engine and
    the machine, comes once at the top."""

    def __init__(self, target: Any, heading: list[str], *, active: bool = True):
        check_target(target)
        self.__target = target if active else None
        self.__heading = heading
        self.__log: Any = None
        self.__count = 0
        self.__parameters: str | None = None
        self.__number = 0
        self.__previous = math.nan
        self.__single = False
        self.__held = False

    @property
    def on(self) -> bool:
        return self.__target is not None

    def start(self, number: int, description: str, parameters: dict) -> None:
        """Opens the log for computation number: the heading before the first,
        then its line, its parameters when they differ from the last ones
        written, and the heading of its rows."""
        if self.__target is None:
            return
        from mdapy.simulation._log import _Log

        self.__log = _Log(self.__target, append=self.__count > 0)
        if self.__count == 0:
            self.__log.title("DFT")
            self.__log.comment(*self.__heading)
        self.__count += 1
        self.__number, self.__previous, self.__single = number, math.nan, False
        self.__held = False
        self.__log.comment("", f"calculation {number}: {description}")
        text = json.dumps(parameters, separators=(", ", ": "))
        if text != self.__parameters:
            self.__log.comment(f"parameters: {text}")
            self.__parameters = text
        self.__log.columns(COLUMNS)

    def step(
        self,
        step: int,
        energy: float,
        residual: float,
        magnetization: float,
        absolute: float,
        fermi: float,
        davidson: int,
        single: bool,
        accepted: bool,
        held: bool = False,
    ) -> None:
        """A row of a step, eV, Bohr magnetons and eV, with its events."""
        if self.__log is None:
            return
        where = f"calculation {self.__number}, step {step}"
        if single and step == 1:
            self.__log.comment(f"{where}: single precision from here")
        elif self.__single and not single:
            self.__log.comment(f"{where}: double precision from here")
        self.__single = bool(single)
        if held and step == 1:
            self.__log.comment(f"{where}: the first moments held from here")
        elif self.__held and not held:
            self.__log.comment(f"{where}: the moments free from here")
        self.__held = bool(held)
        self.__log.row(
            self.__number, int(step), float(energy), float(energy) - self.__previous,
            float(residual), float(magnetization), float(absolute), float(fermi),
            int(davidson),
        )  # fmt: skip
        if accepted:
            self.__previous = float(energy)
        else:
            self.__log.comment(f"{where}: tried back")

    def end(self, summary: str) -> None:
        """The last line of a computation, and the file closed."""
        if self.__log is None:
            return
        self.__log.comment(f"calculation {self.__number}: {summary}")
        self.__log.close()
        self.__log = None
