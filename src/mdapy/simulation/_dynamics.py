# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Molecular dynamics: in LAMMPS with a LAMMPS calculator, or in MDAPY itself
with any other calculator."""

import os
import time
import weakref
from collections.abc import Iterator
from typing import Any

import numpy as np
import polars as pl

from mdapy import units
from mdapy.calculators._lammps import LAMMPS, _close, _lammps
from mdapy.core.box import Box
from mdapy.core.system import System
from mdapy.core.threads import CancelToken
from mdapy.elements import SYMBOLS, atomic_masses
from mdapy.errors import Cancelled, InvalidParameter, MissingProperty
from mdapy.simulation._integrator import _Native
from mdapy.simulation._log import _Log

_ENSEMBLES = ("nve", "nvt", "npt", "nph")
_BAROSTATS = ("iso", "aniso", "tri", "xy")
_THERMOSTATS = ("nose-hoover", "langevin")
# thermo keywords of LAMMPS and the columns they become
# the columns of the log: name in thermo, format, width
_LOG_COLUMNS = (
    ("step", "{:d}", 10),
    ("time", "{:.4f}", 12),
    ("temperature", "{:.3f}", 14),
    ("potential_energy", "{:.6f}", 20),
    ("kinetic_energy", "{:.6f}", 18),
    ("total_energy", "{:.6f}", 20),
    ("pressure", "{:.6f}", 14),
    ("volume", "{:.4f}", 18),
)
_THERMO = (
    ("temp", "temperature"),
    ("pe", "potential_energy"),
    ("ke", "kinetic_energy"),
    ("etotal", "total_energy"),
    ("press", "pressure"),
    ("vol", "volume"),
)


def _pair(value: float | tuple[float, float] | None, name: str) -> tuple | None:
    """A value held through a run, or its values at the start and the end."""
    if value is None:
        return None
    pair = tuple(float(v) for v in np.atleast_1d(np.asarray(value, dtype=float)))
    if len(pair) == 1:
        pair = pair * 2
    if len(pair) != 2:
        raise InvalidParameter(
            f"{name} must be a number or (start, stop), not {value!r}"
        )
    return pair


def _read_written(lmp: Any, path: str | os.PathLike) -> np.ndarray:
    """The rows of numbers of a file that a fix of LAMMPS wrote, on every
    process: under MPI the main process alone writes it, into its own
    folder, and sends what it reads to the others."""

    def read() -> np.ndarray:
        return np.loadtxt(path, comments="#", ndmin=2)

    if int(lmp.extract_setting("world_size")) == 1:
        return read()
    comm = lmp.get_mpi_comm()
    if comm is None:
        raise InvalidParameter(
            "LAMMPS on several processes needs mpi4py, by which its main process "
            "shares the values it writes"
        )
    data: Any = None
    if comm.rank == 0:
        try:
            data = read()
        except Exception as error:  # sent too, or the others would wait for ever
            data = error
    data = comm.bcast(data, root=0)
    if isinstance(data, Exception):
        raise data
    return data


class _Lammps:
    """The state of the dynamics in LAMMPS and its steps."""

    def __init__(self, system: System, potential: LAMMPS, timestep: float):
        if "element" not in system.table.columns:
            raise MissingProperty("molecular dynamics needs the element column")
        if (
            "movable" in system.table.columns
            and not system.table["movable"].to_numpy().all()
        ):
            raise InvalidParameter(
                "atoms held by a movable column need MDAPY's own dynamics, with "
                "a calculator other than LAMMPS"
            )
        numbers = system.table["element"].to_physical().to_numpy()
        type_of = np.zeros(len(SYMBOLS), dtype=np.int32)
        for k, element in reversed(list(enumerate(potential.elements))):
            type_of[SYMBOLS.index(element)] = k + 1
        types = type_of[numbers]
        if (types == 0).any():
            missing = sorted({SYMBOLS[z] for z in numbers[types == 0]})
            raise InvalidParameter(
                f"no atom type has the elements {missing}; the types are "
                f"{list(potential.elements)}"
            )
        self.__n = len(system)
        self.__cmdargs = tuple(potential.cmdargs or ())
        # what the dynamics changes, or leaves out of date, is left out
        moving = (
            "positions",
            "velocities",
            "image",
            "forces",
            "atomic_energies",
            "atomic_virials",
        )
        self.__table = system.table.drop(
            [c for c in moving if c in system.table.columns]
        )
        self.__info = {
            k: v
            for k, v in system.info.items()
            if k not in ("energy", "virial", "stress")
        }
        box = system.box
        self.__swapped = bool(np.linalg.det(box.matrix) < 0)
        lammps_box = box.to_lammps()
        self.__rotation = lammps_box.rotation
        self.__lo = lammps_box.lo
        self.__pbc = box.pbc
        self.__holder: list = []
        self.__fixes: list[str] = []
        self.__finalizer = weakref.finalize(self, _close, self.__holder)
        lmp = _lammps(potential.cmdargs)
        self.__holder.append(lmp)
        pbc = box.pbc[[0, 2, 1]] if self.__swapped else box.pbc
        lengths = lammps_box.hi - lammps_box.lo
        xy, xz, yz = (float(t) for t in lammps_box.tilt)
        lx, ly, lz = (float(v) for v in lengths)
        masses = [float(m) for m in atomic_masses(potential.elements)]
        try:
            lmp.commands_string(
                f"""units metal
atom_style atomic
atom_modify map array
boundary {" ".join("p" if p else "m" for p in pbc)}
region box prism 0 {lx!r} 0 {ly!r} 0 {lz!r} {xy!r} {xz!r} {yz!r}
create_box {len(potential.elements)} box
"""
                + "".join(f"mass {k + 1} {m!r}\n" for k, m in enumerate(masses))
                + potential.commands
                + "\n"
                + potential.pair
                + f"\ntimestep {float(timestep)!r}"
                + "\nthermo_style custom step "
                + " ".join(k for k, _ in _THERMO)
                + " pxx pyy pzz pyz pxz pxy"
                + "\nthermo_modify norm no"
            )
            wrapped, _ = box.wrap(system["positions"])
            x = wrapped @ self.__rotation - self.__lo
            ids = list(range(1, self.__n + 1))
            v = None
            if "velocities" in system.table.columns:
                v = (system["velocities"] @ self.__rotation).ravel().tolist()
            # the box grows along a direction that is not periodic to hold
            # atoms outside it, which LAMMPS would otherwise leave out
            lmp.create_atoms(
                self.__n, ids, types.tolist(), x.ravel().tolist(), v, shrinkexceed=True
            )
            held = int(lmp.get_natoms())
        except Exception as error:
            self.close()
            raise InvalidParameter(f"LAMMPS: {error}") from None
        if held != self.__n:
            self.close()
            raise InvalidParameter(
                f"LAMMPS holds {held} of the {self.__n} atoms of the system"
            )

    # -- the LAMMPS instance --------------------------------------------

    @property
    def lammps(self) -> Any:
        if not self.__holder:
            raise InvalidParameter("this molecular dynamics is closed")
        return self.__holder[0]

    @property
    def closed(self) -> bool:
        return not self.__holder

    def close(self) -> None:
        _close(self.__holder)

    def command(self, commands: str) -> None:
        try:
            self.lammps.commands_string(commands)
        except Exception as error:
            raise InvalidParameter(f"LAMMPS: {error}") from None

    def describe(self) -> str:
        """What runs the dynamics, for the log."""
        lmp = self.lammps
        size = int(lmp.extract_setting("world_size"))
        text = f"LAMMPS {lmp.version()}, {size} process{'' if size == 1 else 'es'}"
        return text + (f", {' '.join(self.__cmdargs)}" if self.__cmdargs else "")

    @property
    def writes(self) -> bool:
        """Whether this process writes the log: the main one under MPI."""
        return int(self.lammps.extract_setting("world_rank")) == 0

    # -- reading the state ----------------------------------------------

    @property
    def step(self) -> int:
        return int(self.lammps.extract_global("ntimestep"))

    @property
    def system(self) -> System:
        lmp = self.lammps
        n = self.__n
        r = self.__rotation
        x = np.ctypeslib.as_array(lmp.gather_atoms("x", 1, 3)).reshape(n, 3)
        v = np.ctypeslib.as_array(lmp.gather_atoms("v", 1, 3)).reshape(n, 3)
        image = np.ctypeslib.as_array(lmp.gather_atoms("image", 0, 3)).reshape(n, 3)
        low, _, xy, yz, xz, _, _ = lmp.extract_box()
        lx, ly, lz = (lmp.get_thermo(k) for k in ("lx", "ly", "lz"))
        matrix = np.array([[lx, 0, 0], [xy, ly, 0], [xz, yz, lz]]) @ r.T
        origin = (np.asarray(low) + self.__lo) @ r.T
        pbc = self.__pbc
        if self.__swapped:
            matrix, image = matrix[[0, 2, 1]], image[:, [0, 2, 1]]
        table = self.__table.with_columns(
            pl.Series(
                "positions", (x + self.__lo) @ r.T, dtype=pl.Array(pl.Float64, 3)
            ),
            pl.Series("velocities", v @ r.T, dtype=pl.Array(pl.Float64, 3)),
            pl.Series("image", image.astype(np.int32), dtype=pl.Array(pl.Int32, 3)),
        )
        step = self.step
        info = self.__info | {"timestep": step, "time": self.__time()}
        return System(table=table, box=Box(matrix, origin=origin, pbc=pbc), info=info)

    def __time(self) -> float:
        # the time LAMMPS has run, in ps: also when the time step changed on
        # the way, as with fix dt/reset or the timestep command
        return float(self.lammps.get_thermo("time"))

    def record(self) -> dict[str, Any]:
        lmp = self.lammps
        step = self.step
        row = {"step": step, "time": self.__time()}
        for keyword, name in _THERMO:
            row[name] = float(lmp.get_thermo(keyword))
        row["pressure"] *= units.BAR_TO_GPA
        p = [lmp.get_thermo(k) for k in ("pxx", "pyy", "pzz", "pyz", "pxz", "pxy")]
        tensor = np.array([[p[0], p[5], p[4]], [p[5], p[1], p[3]], [p[4], p[3], p[2]]])
        stress = -(self.__rotation @ tensor @ self.__rotation.T) * units.BAR_TO_GPA
        row["stress"] = [
            stress[a, b] for a, b in ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
        ]
        return row

    # -- running ----------------------------------------------------------

    def velocities(self, temperature: float, seed: int) -> None:
        self.command(
            f"velocity all create {float(temperature)!r} {int(seed)} "
            f"dist gaussian mom yes rot no loop all"
        )

    def start(self, run: dict[str, Any]) -> None:
        """The fixes of a run, and its setup."""
        ensemble = run["ensemble"]
        langevin = ensemble in ("nvt", "npt") and run["thermostat"] == "langevin"
        fix = ensemble
        if langevin:
            fix = "nve" if ensemble == "nvt" else "nph"
        if ensemble in ("nvt", "npt") and not langevin:
            t0, t1 = run["temperatures"]
            fix += f" temp {t0!r} {t1!r} {float(run['thermostat_damping'])!r}"
        if ensemble in ("npt", "nph"):
            p0, p1 = (p * units.GPA_TO_BAR for p in run["pressures"])
            span = f"{p0!r} {p1!r} {float(run['barostat_damping'])!r}"
            # the second vector of the box is the z axis of LAMMPS when swapped
            other = "z" if self.__swapped else "y"
            barostat = run["barostat"]
            sheet = bool(self.__pbc[:2].all() and not self.__pbc[2])
            if barostat == "xy" or (sheet and barostat == "iso"):
                fix += f" x {span} {other} {span} couple x{other}"
            elif sheet:  # in the plane only
                fix += f" x {span} {other} {span}"
                if barostat == "tri":
                    fix += f" x{other} {span}"
            else:
                fix += f" {barostat} {span}"
            if langevin:
                # the barostat's mass from the temperature of the thermostat
                fix += f" ptemp {run['temperatures'][0]!r}"
        self.command(f"fix mdapy_integrate all {fix}")
        self.__fixes = ["mdapy_integrate"]
        if langevin:
            t0, t1 = run["temperatures"]
            seed = run["seed"]
            if seed is None:
                # the largest seed of LAMMPS's random numbers is 900000000
                seed = int(np.random.default_rng().integers(1, 900_000_000))
            self.command(
                f"fix mdapy_langevin all langevin {t0!r} {t1!r} "
                f"{float(run['thermostat_damping'])!r} {int(seed)}"
            )
            self.__fixes.append("mdapy_langevin")
        self.__start, self.__steps = self.step, run["steps"]
        self.__done = 0
        # the setup knows where the whole run ends: fixes that plan the run
        # at their setup, as deform, would otherwise plan one of 0 steps
        self.command(
            f"run 0 start {self.__start} stop {self.__start + self.__steps} post no"
        )

    def advance(self, count: int) -> None:
        self.command(
            f"run {count} start {self.__start} stop {self.__start + self.__steps} "
            f"pre no post no"
        )

    def finish(self) -> None:
        if self.__holder:
            for fix in self.__fixes:
                self.command(f"unfix {fix}")
        self.__fixes = []


class MolecularDynamics:
    """Molecular dynamics of a system: in LAMMPS with a LAMMPS calculator,
    in MDAPY itself with any other calculator.

    Parameters
    ----------
    system : System
        The atoms, with an ``element`` column (or, without LAMMPS, a
        ``mass`` column); their ``velocities``, if the system has them, in
        Å/ps; a ``movable`` column, (N, 3) bool, keeps atoms still along the
        directions where it is False (not with LAMMPS).
    potential : Calculator
        The interaction: a LAMMPS calculator, run in LAMMPS, or any other
        calculator of MDAPY, run in MDAPY, which needs its energies, forces
        and, for a barostat or the pressure, its virials.
    timestep : float64, optional
        Time step in ps. Default 0.001.
    log : str, os.PathLike or stream, optional
        Writes the runs to this file, which it overwrites, or to this
        stream, such as ``sys.stdout``: a heading, the settings of each run,
        the thermodynamic state every ``thermo`` steps with the seconds
        since the run began, and the speed of each run. Lines that are not
        rows of numbers start with #, so that ``numpy.loadtxt`` reads the
        rows of every run as one table. Under MPI the main process writes.
        Default None, writing none.

    Attributes
    ----------
    timestep : float64
        Time step in ps.

    Raises
    ------
    InvalidParameter
        If the potential is not a calculator, the time step is not
        positive, an element has no atom type, LAMMPS rejects the
        commands, or the log cannot be written.
    MissingProperty
        If the system has no ``element`` column (with LAMMPS), or neither
        a ``mass`` nor an ``element`` column.
    ImportError
        If the LAMMPS Python module is not installed, with a LAMMPS
        calculator.
    """

    def __init__(
        self,
        system: System,
        potential: Any,
        *,
        timestep: float = 0.001,
        log: Any = None,
    ):
        if not timestep > 0:
            raise InvalidParameter(f"timestep must be positive, not {timestep!r}")
        self.timestep = float(timestep)
        self.__n = len(system)
        self.__pbc = system.box.pbc
        self.__rows: list[dict] = []
        self.__frames: list[System] = []
        self.__saved: int | None = None  # the step of the last frame kept or given
        self.__reading: Any = None  # the frames of an iterate not read to the end
        self.__runs = 0
        if isinstance(potential, LAMMPS):
            self.__engine: _Lammps | _Native = _Lammps(system, potential, timestep)
        else:
            self.__engine = _Native(system, potential, timestep)
        try:
            self.__log = _Log(log, active=self.__engine.writes)
        except InvalidParameter:
            self.__engine.close()
            raise
        if self.__log.on:
            self.__heading(system, potential)

    # -- the LAMMPS instance --------------------------------------------

    def __lammps_engine(self) -> _Lammps:
        if not isinstance(self.__engine, _Lammps):
            raise InvalidParameter(
                "this molecular dynamics runs in MDAPY: LAMMPS commands need a "
                "LAMMPS calculator"
            )
        return self.__engine

    @property
    def lammps(self) -> Any:
        """The LAMMPS instance, for anything else (with a LAMMPS
        calculator)."""
        return self.__lammps_engine().lammps

    def close(self) -> None:
        """Closes the LAMMPS instance and the log; the dynamics stops, and
        the frames of an ``iterate`` end."""
        reading = self.__reading() if self.__reading is not None else None
        if reading is not None:
            reading.close()
        self.__engine.close()
        self.__log.close()

    def __heading(self, system: System, potential: Any) -> None:
        box = system.box
        lengths = " ".join(f"{v:.4f}" for v in np.linalg.norm(box.matrix, axis=1))
        periodic = " ".join("abc"[k] for k in range(3) if box.pbc[k]) or "none"
        self.__log.title("molecular dynamics")
        self.__log.comment(
            f"{self.__n} atoms; box vectors {lengths} Å long, periodic along {periodic}",
            f"engine: {self.__engine.describe()}",
            f"potential: {potential!r}",
            f"time step: {self.timestep!r} ps",
        )

    def __enter__(self) -> "MolecularDynamics":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "closed" if self.__engine.closed else f"step {self.step}"
        return f"MolecularDynamics({self.__n} atoms, {state})"

    def command(self, commands: str) -> None:
        """Runs LAMMPS commands, one or several lines (with a LAMMPS
        calculator).

        Parameters
        ----------
        commands : str
            The commands.

        Raises
        ------
        InvalidParameter
            If LAMMPS rejects them, the dynamics runs in MDAPY, or the frames
            of an ``iterate`` are being read.
        """
        self.__idle()
        self.__lammps_engine().command(commands)

    # -- reading the state ----------------------------------------------

    @property
    def step(self) -> int:
        """The time step number: the steps run so far."""
        self.__open()
        return self.__engine.step

    @property
    def system(self) -> System:
        """The atoms now, with their velocities and periodic images, and the
        step and time in ``info``."""
        self.__open()
        return self.__engine.system

    @property
    def thermo(self) -> pl.DataFrame:
        """The thermodynamic state every ``thermo`` steps of the runs: step,
        time (ps), temperature (K), potential, kinetic and total energy
        (eV), pressure (GPa), volume (Å³) and stress (GPa, Voigt order,
        positive under tension)."""
        schema = {
            "step": pl.Int64,
            "time": pl.Float64,
            **{name: pl.Float64 for _, name in _THERMO},
            "stress": pl.Array(pl.Float64, 6),
        }
        return pl.DataFrame(self.__rows, schema=schema)

    @property
    def frames(self) -> tuple[System, ...]:
        """The atoms every ``save_every`` steps of the runs."""
        return tuple(self.__frames)

    def __open(self) -> None:
        if self.__engine.closed:
            raise InvalidParameter("this molecular dynamics is closed")

    def __idle(self) -> None:
        """Refuses to change the dynamics while the frames of an iterate
        are being read: its run is still going."""
        if self.__reading is not None and self.__reading() is not None:
            raise InvalidParameter(
                "the frames of an iterate are being read: read them to the end "
                "or close them first"
            )

    # -- running ----------------------------------------------------------

    def velocities(self, temperature: float, *, seed: int) -> None:
        """Gives the atoms random velocities of a temperature, with no
        total momentum.

        Parameters
        ----------
        temperature : float64
            Temperature in K, not negative.
        seed : int32
            Seed of the random velocities, positive.

        Raises
        ------
        InvalidParameter
            For a negative temperature or a seed that is not positive, or
            while the frames of an ``iterate`` are being read.
        """
        if not temperature >= 0:
            raise InvalidParameter(
                f"temperature must not be negative, not {temperature!r}"
            )
        if int(seed) < 1:
            raise InvalidParameter(f"seed must be positive, not {seed!r}")
        self.__open()
        self.__idle()
        self.__engine.velocities(float(temperature), int(seed))

    def run(
        self,
        steps: int,
        *,
        ensemble: str = "nve",
        temperature: float | tuple[float, float] | None = None,
        pressure: float | tuple[float, float] | None = None,
        barostat: str = "iso",
        thermostat: str = "nose-hoover",
        thermostat_damping: float | None = None,
        barostat_damping: float | None = None,
        seed: int | None = None,
        thermo: int = 100,
        save_every: int | None = None,
        progress: Any = None,
        cancel: CancelToken | None = None,
    ) -> None:
        """Runs molecular dynamics.

        Parameters
        ----------
        steps : int32
            Number of steps.
        ensemble : {"nve", "nvt", "npt", "nph"}, optional
            Constant energy, or a thermostat, and barostat, or a barostat
            alone. Default "nve".
        temperature : float64 or (float64, float64), optional
            Temperature in K of the thermostat, or its values at the start
            and the end of the run; needed for "nvt" and "npt".
        pressure : float64 or (float64, float64), optional
            Pressure in GPa of the barostat, or its values at the start and
            the end; needed for "npt" and "nph".
        barostat : {"iso", "aniso", "tri", "xy"}, optional
            The box changes alike in every direction, along each axis, in
            any way, or alike along its first two vectors with the third
            kept. A sheet, periodic along its first two vectors only, changes
            in its plane only: alike along them for "iso" and "xy", along
            each for "aniso", and in any way in the plane for "tri". Default
            "iso".
        thermostat : {"nose-hoover", "langevin"}, optional
            A chain of Nosé-Hoover thermostats, or Langevin dynamics. Default
            "nose-hoover".
        thermostat_damping : float64, optional
            Damping time of the thermostat in ps. Default 100 time steps.
        barostat_damping : float64, optional
            Damping time of the barostat in ps. Default 1000 time steps.
        seed : int32, optional
            Seed of the random forces of Langevin dynamics, positive;
            unpredictable when None.
        thermo : int32, optional
            Records the thermodynamic state every this many steps, and at
            the start and the end. Default 100.
        save_every : int32, optional
            Keeps the atoms every this many steps, in ``frames``. Default
            None, keeping none.
        progress : callable, optional
            Called as ``progress(done, total)`` with the steps run.
        cancel : CancelToken, optional
            Stops the run, which raises ``mdapy.errors.Cancelled``.

        Raises
        ------
        InvalidParameter
            For invalid options, if LAMMPS fails, or while the frames of an
            ``iterate`` are being read.
        CalculationFailed
            If the energy or the forces stop being finite, in MDAPY.
        Cancelled
            If cancelled.
        """
        spec = self.__plan(
            steps,
            ensemble,
            temperature,
            pressure,
            barostat,
            thermostat,
            thermostat_damping,
            barostat_damping,
            seed,
            thermo,
            save_every,
        )
        self.__idle()
        self.__frames.extend(self.__steps(spec, progress, cancel))

    def iterate(
        self,
        steps: int,
        *,
        save_every: int,
        ensemble: str = "nve",
        temperature: float | tuple[float, float] | None = None,
        pressure: float | tuple[float, float] | None = None,
        barostat: str = "iso",
        thermostat: str = "nose-hoover",
        thermostat_damping: float | None = None,
        barostat_damping: float | None = None,
        seed: int | None = None,
        thermo: int = 100,
        progress: Any = None,
        cancel: CancelToken | None = None,
    ) -> Iterator[System]:
        """Runs molecular dynamics as ``run`` does, giving the atoms every
        save_every steps as the run reaches them instead of keeping them: the
        run goes on as the frames are read, and stops where their reading
        ends.

        Parameters
        ----------
        steps : int32
            Number of steps.
        save_every : int32
            Gives the atoms every this many steps, and at the start unless
            the run before gave them there.
        ensemble : {"nve", "nvt", "npt", "nph"}, optional
            Constant energy, or a thermostat, and barostat, or a barostat
            alone. Default "nve".
        temperature : float64 or (float64, float64), optional
            Temperature in K of the thermostat, or its values at the start
            and the end of the run; needed for "nvt" and "npt".
        pressure : float64 or (float64, float64), optional
            Pressure in GPa of the barostat, or its values at the start and
            the end; needed for "npt" and "nph".
        barostat : {"iso", "aniso", "tri", "xy"}, optional
            The box changes alike in every direction, along each axis, in
            any way, or alike along its first two vectors with the third
            kept. A sheet, periodic along its first two vectors only, changes
            in its plane only: alike along them for "iso" and "xy", along
            each for "aniso", and in any way in the plane for "tri". Default
            "iso".
        thermostat : {"nose-hoover", "langevin"}, optional
            A chain of Nosé-Hoover thermostats, or Langevin dynamics. Default
            "nose-hoover".
        thermostat_damping : float64, optional
            Damping time of the thermostat in ps. Default 100 time steps.
        barostat_damping : float64, optional
            Damping time of the barostat in ps. Default 1000 time steps.
        seed : int32, optional
            Seed of the random forces of Langevin dynamics, positive;
            unpredictable when None.
        thermo : int32, optional
            Records the thermodynamic state every this many steps, and at
            the start and the end. Default 100.
        progress : callable, optional
            Called as ``progress(done, total)`` with the steps run.
        cancel : CancelToken, optional
            Stops the run, which raises ``mdapy.errors.Cancelled``.


        Returns
        -------
        iterator of System
            The frames, each as ``system`` gives it, read once; ``frames``
            keeps none of them.

        Raises
        ------
        InvalidParameter
            For invalid options, if LAMMPS fails, or while the frames of
            another ``iterate`` are being read; the options are checked at
            once, the rest as the frames are read.
        CalculationFailed
            If the energy or the forces stop being finite, in MDAPY.
        Cancelled
            If cancelled.
        """
        if save_every is None:
            raise InvalidParameter("iterate needs save_every")
        spec = self.__plan(
            steps,
            ensemble,
            temperature,
            pressure,
            barostat,
            thermostat,
            thermostat_damping,
            barostat_damping,
            seed,
            thermo,
            save_every,
        )
        return self.__read(spec, progress, cancel)

    def __read(
        self, spec: dict[str, Any], progress: Any, cancel: CancelToken | None
    ) -> Iterator[System]:
        """The frames of an iterate: no other run, command or velocities
        while they are being read, from the first to the last or until they
        are closed."""
        self.__idle()
        frames = self.__steps(spec, progress, cancel)
        try:
            first = next(frames)
        except StopIteration:
            return
        self.__reading = weakref.ref(frames)
        try:
            yield first
            yield from frames
        finally:
            frames.close()
            self.__reading = None

    def __plan(
        self,
        steps: int,
        ensemble: str,
        temperature: float | tuple[float, float] | None,
        pressure: float | tuple[float, float] | None,
        barostat: str,
        thermostat: str,
        thermostat_damping: float | None,
        barostat_damping: float | None,
        seed: int | None,
        thermo: int,
        save_every: int | None,
    ) -> dict[str, Any]:
        """The checked options of a run."""
        steps = int(steps)
        if steps < 0:
            raise InvalidParameter(f"steps must not be negative, not {steps!r}")
        if ensemble not in _ENSEMBLES:
            raise InvalidParameter(
                f"ensemble must be one of {_ENSEMBLES}, not {ensemble!r}"
            )
        if barostat not in _BAROSTATS:
            raise InvalidParameter(
                f"barostat must be one of {_BAROSTATS}, not {barostat!r}"
            )
        if thermostat not in _THERMOSTATS:
            raise InvalidParameter(
                f"thermostat must be one of {_THERMOSTATS}, not {thermostat!r}"
            )
        if int(thermo) < 1 or (save_every is not None and int(save_every) < 1):
            raise InvalidParameter("thermo and save_every must be positive")
        if seed is not None and int(seed) < 1:
            raise InvalidParameter(f"seed must be positive, not {seed!r}")
        temperatures = _pair(temperature, "temperature")
        pressures = _pair(pressure, "pressure")
        if ensemble in ("nvt", "npt") and temperatures is None:
            raise InvalidParameter(f"{ensemble} needs a temperature")
        if ensemble in ("npt", "nph") and pressures is None:
            raise InvalidParameter(f"{ensemble} needs a pressure")
        if temperatures is not None and ensemble in ("nvt", "npt"):
            if not min(temperatures) > 0:
                raise InvalidParameter(
                    f"the temperature must be positive, not {temperature!r}"
                )
        sheet = bool(self.__pbc[:2].all() and not self.__pbc[2])
        if ensemble in ("npt", "nph"):
            if barostat == "xy" and not self.__pbc[:2].all():
                raise InvalidParameter(
                    "this barostat needs a box periodic along its first two vectors"
                )
            if not (self.__pbc.all() or sheet):
                raise InvalidParameter(
                    "a barostat needs a box periodic in every direction, or along "
                    "its first two vectors for a sheet"
                )
        self.__open()
        dt = self.timestep
        spec = {
            "steps": steps,
            "ensemble": ensemble,
            "temperatures": temperatures,
            "pressures": pressures,
            "barostat": barostat,
            "thermostat": thermostat,
            "thermostat_damping": 100 * dt
            if thermostat_damping is None
            else float(thermostat_damping),
            "barostat_damping": 1000 * dt
            if barostat_damping is None
            else float(barostat_damping),
            "seed": None if seed is None else int(seed),
            "thermo": int(thermo),
            "save_every": None if save_every is None else int(save_every),
        }
        return spec

    def __steps(
        self, spec: dict[str, Any], progress: Any, cancel: CancelToken | None
    ) -> Iterator[System]:
        """Runs, recording the thermodynamic state, and gives the atoms every
        save_every steps; the run ends where the frames stop being read."""
        engine = self.__engine
        steps, thermo, every = spec["steps"], spec["thermo"], spec["save_every"]
        done = 0
        finished = False
        log = self.__log
        self.__runs += 1
        if log.on:
            log.comment("", self.__settings(self.__runs, spec))
            log.columns(_LOG_COLUMNS)
        began = time.perf_counter()
        try:
            engine.start(spec)
            start = engine.step
            if not self.__rows or self.__rows[-1]["step"] != start:
                self.__record()
            time0 = self.__rows[-1]["time"]
            if every is not None and self.__saved != start:
                self.__saved = start
                yield engine.system
            stops = set(range(thermo, steps, thermo)) | {steps}
            if every is not None:
                stops |= set(range(every, steps + 1, every))
            stops.discard(0)  # a run of no steps: the start only, once
            for stop in sorted(stops):
                if cancel is not None and cancel.cancelled:
                    raise Cancelled("molecular dynamics cancelled")
                engine.advance(stop - done)
                done = stop
                if done % thermo == 0 or done == steps:
                    self.__record()
                if progress is not None:
                    progress(done, steps)
                if every is not None and done % every == 0:
                    self.__saved = engine.step
                    yield engine.system
            finished = True
        finally:
            if not engine.closed:
                engine.finish()
            if log.on:
                seconds = time.perf_counter() - began
                if not finished:
                    log.comment(
                        f"run {self.__runs} stopped after {done} of {steps} steps, "
                        f"{seconds:.2f} s"
                    )
                elif done == 0 or seconds <= 0.0:
                    log.comment(f"run {self.__runs}: {done} steps")
                else:
                    simulated = self.__rows[-1]["time"] - time0  # ps
                    log.comment(
                        f"run {self.__runs}: {done} steps in {seconds:.2f} s; "
                        f"{done / seconds:.1f} steps/s, "
                        f"{self.__n * done / seconds / 1e6:.4g} million atom-steps/s, "
                        f"{simulated * 86.4 / seconds:.4g} ns/day"
                    )

    def __record(self) -> None:
        row = self.__engine.record()
        self.__rows.append(row)
        self.__log.row(*(row[name] for name, _, _ in _LOG_COLUMNS))

    def __settings(self, run: int, spec: dict[str, Any]) -> str:
        """The settings of a run, for the log."""

        def span(values: tuple, unit: str) -> str:
            first, last = values
            return (
                f"{first!r} {unit}"
                if first == last
                else f"{first!r} to {last!r} {unit}"
            )

        ensemble = spec["ensemble"]
        parts = [f"run {run}: {spec['steps']} steps, {ensemble}"]
        if ensemble in ("nvt", "npt"):
            text = f"{span(spec['temperatures'], 'K')}, {spec['thermostat']}"
            text += f" with damping {spec['thermostat_damping']!r} ps"
            if spec["thermostat"] == "langevin" and spec["seed"] is not None:
                text += f", seed {spec['seed']}"
            parts.append(text)
        if ensemble in ("npt", "nph"):
            parts.append(
                f"{span(spec['pressures'], 'GPa')}, barostat {spec['barostat']} with "
                f"damping {spec['barostat_damping']!r} ps"
            )
        parts.append(f"thermo every {spec['thermo']}")
        if spec["save_every"] is not None:
            parts.append(f"atoms every {spec['save_every']}")
        return "; ".join(parts)
