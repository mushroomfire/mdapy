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
from mdapy.simulation import _controls, _restart
from mdapy.simulation._integrator import _Native
from mdapy.simulation._log import _Log, _machine

_ENSEMBLES = ("nve", "nvt", "npt", "nph", "nphug", "msst")
_BAROSTATS = ("iso", "aniso", "tri", "xy", "x", "y", "z")
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


def _hugoniot(value: Any, ensemble: str) -> tuple[float, float, float] | None:
    """The reference state of nphug, (energy, volume, pressure), or None."""
    if value is None:
        return None
    if ensemble not in ("nphug", "msst"):
        raise InvalidParameter('hugoniot is for the ensembles "nphug" and "msst"')
    try:
        energy, volume, pressure = (float(v) for v in value)
    except (TypeError, ValueError):
        raise InvalidParameter(
            f"hugoniot must be (energy, volume, pressure), not {value!r}"
        ) from None
    if not (np.isfinite([energy, pressure]).all() and volume > 0):
        raise InvalidParameter(
            "hugoniot needs a finite energy and pressure and a positive volume, "
            f"not {value!r}"
        )
    return energy, volume, pressure


# the settings of msst and their defaults, those of LAMMPS's fix msst
_MSST = {"q": 10.0, "mu": 0.0, "tscale": 0.01}


def _check_msst(
    run: dict[str, Any], barostat: str, pbc: Any, controls: tuple
) -> dict[str, float]:
    """What fix msst takes: a shock velocity, an axis, a box periodic in
    every direction, no pressure, temperature or controls; its settings with
    their defaults."""
    speed, settings = run["shock_velocity"], run["msst"]
    if speed is None:
        raise InvalidParameter("msst needs a shock_velocity")
    if isinstance(speed, bool) or not (np.isfinite(speed) and speed >= 0):
        raise InvalidParameter(
            f"shock_velocity must be a number not negative, not {speed!r}"
        )
    if barostat not in ("x", "y", "z"):
        raise InvalidParameter(
            'msst needs the axis of the shock: barostat "x", "y" or "z"'
        )
    if run["pressures"] is not None or run["temperatures"] is not None:
        raise InvalidParameter("msst takes no pressure or temperature")
    if controls:
        raise InvalidParameter("msst takes no controls")
    if not np.all(pbc):
        raise InvalidParameter("msst needs a box periodic in every direction")
    if settings is None:
        settings = {}
    if not isinstance(settings, dict) or set(settings) - set(_MSST):
        raise InvalidParameter(
            f"msst takes a dict of q, mu and tscale, not {settings!r}"
        )
    out = _MSST | {k: float(v) for k, v in settings.items()}
    if not (out["q"] > 0 and out["mu"] >= 0 and 0 <= out["tscale"] <= 1):
        raise InvalidParameter(
            "msst needs q positive, mu not negative and tscale from 0 to 1, "
            f"not {settings!r}"
        )
    return out


def _check_nphug(
    pressures: tuple, temperatures: Any, thermostat: str, barostat: str, pbc: Any
) -> None:
    """What fix nphug takes: one pressure, no temperature, the chain of
    Nosé-Hoover thermostats, a hydrostatic or uniaxial barostat, a box
    periodic in every direction."""
    if pressures[0] != pressures[1]:
        raise InvalidParameter("nphug holds one pressure: not a start and a stop")
    if temperatures is not None:
        raise InvalidParameter(
            "nphug takes its temperature from the Hugoniot: give no temperature"
        )
    if thermostat != "nose-hoover":
        raise InvalidParameter('nphug needs the "nose-hoover" thermostat')
    if barostat == "xy":
        raise InvalidParameter(
            'nphug needs a barostat "iso", "aniso", "tri", "x", "y" or "z"'
        )
    if not np.all(pbc):
        raise InvalidParameter("nphug needs a box periodic in every direction")


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
        self.__timestep = float(timestep)
        self.__adaptive = False  # whether a run changes the time step
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
        self.__computes: list[str] = []
        self.__groups: list[str] = []
        self.__moves: tuple = ()
        self.__baths: tuple = ()
        self.__temperature: str | None = None
        self.__hugoniostat = False  # whether the run is fix nphug
        self.__shock = False  # whether the run is fix msst
        self.__variables: list[str] = []  # the variables of the run
        self.__indented: tuple = ()  # (name, fix) of each named indenter
        self.__piston = False  # whether a wall/piston shrink-wraps the low side of z
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
        if self.__temperature is not None:
            row["temperature"] = float(lmp.extract_compute(self.__temperature, 0, 0))
        for name, compute in self.__moves:
            total = np.ctypeslib.as_array(
                lmp.extract_compute(compute, 0, 1), shape=(3,)
            )
            row[f"{name}_force"] = (
                self.__rotation @ np.array(total, dtype=float)
            ).tolist()
        for name, fix in self.__baths:
            # the tally of LAMMPS is the energy the bath took away
            row[f"{name}_energy"] = -float(lmp.extract_fix(fix, 0, 0))
        for name, fix in self.__indented:  # the vector and the scalar of fix indent
            row[f"{name}_force"] = [
                float(lmp.extract_fix(fix, 0, 1, k)) for k in range(3)
            ]
            row[f"{name}_energy"] = float(lmp.extract_fix(fix, 0, 0))
        if self.__shock:  # the vector and the scalar of fix msst
            names = ("hugoniot", "rayleigh", "particle_velocity", "lagrangian_position")
            for k, name in enumerate(names):
                row[name] = float(lmp.extract_fix("mdapy_integrate", 0, 1, k))
            row["rayleigh"] *= units.BAR_TO_GPA
            row["msst_energy"] = float(lmp.extract_fix("mdapy_integrate", 0, 0))
        if self.__hugoniostat:  # the first three of the vector of fix nphug
            for k, name in enumerate(
                ("hugoniot", "shock_velocity", "particle_velocity")
            ):
                row[name] = float(lmp.extract_fix("mdapy_integrate", 0, 1, k))
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
        if ensemble == "nphug":
            # the temperatures are those of the Hugoniot: these two are not used
            fix += f" temp 1.0 1.0 {float(run['thermostat_damping'])!r}"
        if ensemble == "msst":
            fix += self.__msst(run)
        if ensemble in ("npt", "nph", "nphug"):
            p0, p1 = (p * units.GPA_TO_BAR for p in run["pressures"])
            span = f"{p0!r} {p1!r} {float(run['barostat_damping'])!r}"
            # the second vector of the box is the z axis of LAMMPS when swapped
            other = "z" if self.__swapped else "y"
            barostat = run["barostat"]
            sheet = bool(self.__pbc[:2].all() and not self.__pbc[2])
            deform = next(
                (c for c in run["controls"] if isinstance(c, _controls.Deform)), None
            )
            if deform is not None:
                if barostat not in ("iso", "aniso") or deform.rates[3:].any():
                    raise InvalidParameter(
                        'a barostat with a deformation needs "iso" or "aniso" and no '
                        "shear"
                    )
                if sheet:
                    raise InvalidParameter(
                        "a deformed sheet with a barostat is not supported"
                    )
                left = [d for k, d in enumerate("xyz") if deform.rates[k] == 0.0]
                if not left:
                    raise InvalidParameter(
                        "the deformation leaves the barostat no length"
                    )
                fix += "".join(f" {d} {span}" for d in left)
                if barostat == "iso" and len(left) > 1:
                    fix += f" couple {''.join(left)}"
            elif barostat in ("x", "y", "z"):
                aligned = np.allclose(self.__rotation, np.eye(3), rtol=0, atol=1e-12)
                if not (aligned and not self.__swapped):
                    raise InvalidParameter(
                        "a barostat along one axis needs a box whose first vector is "
                        "along x and second in the xy plane"
                    )
                fix += f" {barostat} {span}"
            elif barostat == "xy" or (sheet and barostat == "iso"):
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
        self.__fixes, self.__computes, self.__groups = [], [], []
        group, extra = self.__set_controls(
            run["controls"], ensemble in ("npt", "nph", "nphug")
        )
        self.command(f"fix mdapy_integrate {group} {fix}")
        self.__fixes.insert(0, "mdapy_integrate")
        self.__hugoniostat = ensemble == "nphug"
        self.__shock = ensemble == "msst"
        if self.__hugoniostat and run["hugoniot"] is not None:
            e0, v0, p0 = run["hugoniot"]
            self.command(
                f"fix_modify mdapy_integrate e0 {e0!r} v0 {v0!r} "
                f"p0 {p0 * units.GPA_TO_BAR!r}"
            )
        if extra is not None and ensemble in ("nvt", "npt") and not langevin:
            # the thermostat counts the directions that move freely
            self.command(f"compute_modify mdapy_integrate_temp extra/dof {extra}")
        if langevin:
            t0, t1 = run["temperatures"]
            seed = run["seed"]
            if seed is None:
                # the largest seed of LAMMPS's random numbers is 900000000
                seed = int(np.random.default_rng().integers(1, 900_000_000))
            self.command(
                f"fix mdapy_langevin {group} langevin {t0!r} {t1!r} "
                f"{float(run['thermostat_damping'])!r} {int(seed)}"
                + (" zero yes" if run.get("keep_momentum") else "")
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
            for compute in self.__computes:
                self.command(f"uncompute {compute}")
            for group in self.__groups:
                self.command(f"group {group} delete")
            for variable in self.__variables:
                self.command(f"variable {variable} delete")
        self.__fixes, self.__computes, self.__groups = [], [], []
        self.__variables = []
        self.__indented = ()
        if self.__holder and self.__piston:
            flags = " ".join("p" if p else "m" for p in self.__pbc)
            self.command(f"change_box all boundary {flags}")
        self.__piston = False
        if self.__holder and self.__adaptive:
            # fix dt/reset leaves its last time step
            self.command(f"timestep {self.__timestep!r}")
        self.__adaptive = False
        self.__hugoniostat = False
        self.__shock = False
        self.__moves = ()
        self.__baths = ()
        self.__temperature = None

    def __indenter(self, name: str, indenter: Any, aligned: bool) -> None:
        """A virtual indenter as fix indent, its moving coordinates variables
        of the steps of the run."""
        if not (aligned and not self.__swapped):
            raise InvalidParameter(
                "an indenter needs a box whose first vector is along x and second "
                "in the xy plane"
            )
        dt = float(self.lammps.extract_global("dt"))

        def coordinate(value: float, velocity: float, k: int) -> str:
            start = float(value - self.__lo[k])  # LAMMPS's box starts at 0
            if velocity == 0.0:
                return repr(start)
            variable = f"{name}_{'xyz'[k]}"
            self.command(
                f"variable {variable} equal {start!r}+{float(velocity) * dt!r}*elaplong"
            )
            self.__variables.append(variable)
            return f"v_{variable}"

        if indenter.shape == "plane":
            k = "xyz".index(indenter.axis)
            if self.__pbc[k]:
                raise InvalidParameter(
                    f"a plane indenter across {indenter.axis} needs a box not periodic "
                    f"along {indenter.axis}"
                )
            where = coordinate(indenter.position, indenter.velocity, k)
            side = "hi" if indenter.side == "high" else "lo"
            geometry = f"plane {indenter.axis} {where} {side}"
        else:
            c, v = indenter.center, indenter.velocity
            if indenter.shape == "sphere":
                axes = (0, 1, 2)
                geometry = "sphere"
            else:
                axes = tuple(k for k in range(3) if "xyz"[k] != indenter.axis)
                geometry = f"cylinder {indenter.axis}"
            geometry += "".join(f" {coordinate(c[k], v[k], k)}" for k in axes)
            geometry += f" {indenter.radius!r} side {indenter.side}"
        self.command(
            f"fix {name} all indent {indenter.stiffness!r} {geometry} units box"
        )
        self.__fixes.append(name)

    def __msst(self, run: dict[str, Any]) -> str:
        """The arguments of fix msst, in a box in the form of LAMMPS."""
        aligned = np.allclose(self.__rotation, np.eye(3), rtol=0, atol=1e-12)
        if not (aligned and not self.__swapped):
            raise InvalidParameter(
                "msst needs a box whose first vector is along x and second in the "
                "xy plane"
            )
        settings = run["msst"]
        words = f" {run['barostat']} {run['shock_velocity']!r}" + "".join(
            f" {k} {settings[k]!r}" for k in ("q", "mu", "tscale")
        )
        if run["hugoniot"] is not None:
            e0, v0, p0 = run["hugoniot"]
            words += f" p0 {p0 * units.GPA_TO_BAR!r} v0 {v0!r} e0 {e0!r}"
        return words

    def __wall(self, name: str, wall: Any, aligned: bool) -> None:
        """A still wall as fix wall/reflect, a moving one as fix wall/piston,
        which LAMMPS has on the low side of z, moving up, only."""
        if not (aligned and not self.__swapped):
            raise InvalidParameter(
                "a wall needs, in LAMMPS, a box whose first vector is along x and "
                "second in the xy plane"
            )
        low, _, xy, yz, xz, _, _ = self.lammps.extract_box()
        lx, ly, lz = (self.lammps.get_thermo(k) for k in ("lx", "ly", "lz"))
        matrix = np.array([[lx, 0, 0], [xy, ly, 0], [xz, yz, lz]])
        k = "xyz".index(wall.axis)
        # in the coordinates of LAMMPS, which start at the corner of the box
        at = float(
            wall._start(matrix, np.asarray(low) + self.__lo, self.__pbc) - self.__lo[k]
        )
        if wall.velocity == 0.0:
            face = wall.axis + ("lo" if wall.side == "low" else "hi")
            self.command(f"fix {name} all wall/reflect {face} {at!r} units box")
        elif wall.axis == "z" and wall.side == "low" and wall.velocity > 0.0:
            # the piston needs the low side of z shrink-wrapped
            flags = ["p" if p else "m" for p in self.__pbc]
            self.command(f"change_box all boundary {flags[0]} {flags[1]} sm")
            self.__piston = True
            self.command(
                f"fix {name} all wall/piston zlo pos {at!r} vel {wall.velocity!r} "
                "units box"
            )
        else:
            raise InvalidParameter(
                "a moving wall needs, in LAMMPS, the low side of z and a velocity "
                "up z, as fix wall/piston; MDAPY's own dynamics move any wall"
            )
        self.__fixes.append(name)

    def __group(self, name: str, mask: np.ndarray) -> str:
        """A group of LAMMPS of the atoms of a mask, by their ids."""
        ids = np.flatnonzero(mask) + 1
        self.__groups.append(name)
        if ids.size == 0:
            self.command(f"group {name} empty")
            return name
        # runs of consecutive ids as first:last
        breaks = np.flatnonzero(np.diff(ids) != 1)
        firsts = np.concatenate([ids[:1], ids[breaks + 1]])
        lasts = np.concatenate([ids[breaks], ids[-1:]])
        words = [f"{a}" if a == b else f"{a}:{b}" for a, b in zip(firsts, lasts)]
        for k in range(0, len(words), 2000):
            self.command(f"group {name} id {' '.join(words[k : k + 2000])}")
        return name

    def __set_controls(self, controls: tuple, barostat: bool) -> tuple[str, int | None]:
        """The groups, fixes and computes of the controls of a run: the group
        the ensemble moves, and the directions its thermostat leaves out
        beyond those of the atoms held or moved whole, or None."""
        self.__moves = ()
        self.__baths = ()
        self.__temperature = None
        if not controls:
            return "all", None
        if len(controls) > 24:
            raise InvalidParameter("LAMMPS takes at most 24 controls in a run")
        n = self.__n
        r = self.__rotation
        aligned = np.allclose(r, np.eye(3), rtol=0, atol=1e-12)
        held = np.zeros((n, 3), dtype=bool)
        moved = np.zeros(n, dtype=bool)
        moves, baths = [], []
        constrained = False
        # the indenters first, so that the setforce of held atoms comes after
        indented = []
        for k, control in enumerate(controls):
            if isinstance(control, _controls.Indenter):
                self.__indenter(f"mdapy_c{k}", control, aligned)
                if control.name is not None:
                    indented.append((control.name, f"mdapy_c{k}"))
        self.__indented = tuple(indented)
        # then the forces added, and the rest in the order given: LAMMPS applies
        # its fixes in the order they are defined, and the setforce of atoms held
        # along some directions must come after every force, as MDAPY holds them
        # last, whatever the order of the controls
        numbered = list(enumerate(controls))
        added = [(k, c) for k, c in numbered if isinstance(c, _controls.AddForce)]
        others = [(k, c) for k, c in numbered if not isinstance(c, _controls.AddForce)]
        for k, control in added + others:
            name = f"mdapy_c{k}"
            if isinstance(control, _controls.Indenter):
                continue
            if isinstance(control, _controls.Hold):
                if not aligned and control.directions != "xyz":
                    raise InvalidParameter(
                        "atoms held along some directions need, in LAMMPS, a box whose "
                        "first vector is along x and second in the xy plane"
                    )
                axes = control.axes  # the axes of LAMMPS, the box aligned
                held[control.mask] |= control.axes
                constrained = constrained or bool(control.mask.any())
                group = self.__group(name, control.mask)
                values = " ".join("0.0" if a else "NULL" for a in axes)
                self.command(f"velocity {group} set {values}")
                self.command(f"fix {name} {group} setforce {values}")
                self.__fixes.append(name)
            elif isinstance(control, _controls.Move):
                moved |= control.mask
                constrained = constrained or bool(control.mask.any())
                group = self.__group(name, control.mask)
                vx, vy, vz = (float(c) for c in control.velocity @ r)
                # their velocities from the start, as fix move sets them at the
                # first step only
                self.command(f"velocity {group} set {vx!r} {vy!r} {vz!r}")
                self.command(f"fix {name} {group} move linear {vx!r} {vy!r} {vz!r}")
                self.__fixes.append(name)
                if control.name is not None:
                    self.command(f"compute {name}_force {group} reduce sum fx fy fz")
                    self.__computes.append(f"{name}_force")
                    moves.append((control.name, f"{name}_force"))
            elif isinstance(control, _controls.Deform):
                if not (aligned and not self.__swapped):
                    raise InvalidParameter(
                        "a deformation needs a box whose first vector is along x and "
                        "second in the xy plane"
                    )
                if control.kind == "true" and control.rates[3:].any():
                    raise InvalidParameter("tilts deform at engineering rates only")
                control._periodic(self.__pbc)
                style = "trate" if control.kind == "true" else "erate"
                words = []
                for k, part in enumerate(("x", "y", "z", "yz", "xz", "xy")):
                    rate = float(control.rates[k])
                    if rate != 0.0:
                        words.append(f"{part} {'erate' if k > 2 else style} {rate!r}")
                self.command(
                    f"fix {name} all deform 1 {' '.join(words)} remap x flip no"
                )
                self.__fixes.append(name)
            elif isinstance(control, _controls.Wall):
                self.__wall(name, control, aligned)
            elif isinstance(control, _controls._HeatField):
                raise InvalidParameter(
                    "HNEMD needs MDAPY's own dynamics, with a calculator other than "
                    "LAMMPS"
                )
            elif isinstance(control, _controls.Switch):
                raise InvalidParameter(
                    "a Switch needs MDAPY's own dynamics, with a calculator other "
                    "than LAMMPS"
                )
            elif isinstance(control, _controls.AdaptiveTimestep):
                low = (
                    "NULL"
                    if control.min_timestep is None
                    else repr(control.min_timestep)
                )
                high = (
                    self.__timestep
                    if control.max_timestep is None
                    else control.max_timestep
                )
                self.command(
                    f"fix {name} all dt/reset {control.every} {low} {high!r} "
                    f"{control.max_distance!r} units box"
                )
                self.__fixes.append(name)
                self.__adaptive = True
            elif isinstance(control, _controls.Langevin):
                group = self.__group(name, control.mask)
                t0, t1 = control.temperature
                dt = float(self.lammps.extract_global("dt"))
                damping = control.damping if control.damping is not None else 100 * dt
                seed = control.seed
                if seed is None:
                    seed = int(np.random.default_rng().integers(1, 900_000_000))
                self.command(
                    f"fix {name} {group} langevin {t0!r} {t1!r} {damping!r} {seed} tally yes"
                )
                self.__fixes.append(name)
                if control.name is not None:
                    baths.append((control.name, name))
            else:
                group = self.__group(name, control.mask)
                fx, fy, fz = (float(c) for c in control.force @ r)
                self.command(f"fix {name} {group} addforce {fx!r} {fy!r} {fz!r}")
                self.__fixes.append(name)
        if barostat and constrained:
            raise InvalidParameter(
                "a barostat with held or moved atoms is not supported"
            )
        self.__moves = tuple(moves)
        self.__baths = tuple(baths)
        if not constrained:
            return "all", None
        free = ~(held.all(axis=1) | moved)
        group = self.__group("mdapy_free", free)
        # LAMMPS counts 3 directions of each atom of the group, less 3: the
        # directions held of the atoms it still moves are left out instead
        extra = int(held[free].sum())
        self.command(f"compute mdapy_temp {group} temp")
        self.command(f"compute_modify mdapy_temp extra/dof {extra}")
        self.__computes.append("mdapy_temp")
        self.__temperature = "mdapy_temp"
        return group, extra


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
        self.__potential = potential
        # a run a restart file stopped in: its settings and the steps it ran
        self.__pending: tuple[dict[str, Any], int] | None = None
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
            f"machine: {_machine()}",
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
        (eV), pressure (GPa), volume (Å³), stress (GPa, Voigt order,
        positive under tension), ``<name>_force``, (3,) eV/Å, for each named
        group a ``Move`` moved, and ``<name>_energy`` (eV), the heat each
        named ``Langevin`` bath gave since its run began, null in the other
        runs."""
        schema = {
            "step": pl.Int64,
            "time": pl.Float64,
            **{name: pl.Float64 for _, name in _THERMO},
            "stress": pl.Array(pl.Float64, 6),
        }
        # the forces on the groups the controls of some runs move
        for row in self.__rows:
            for key, value in row.items():
                if key not in schema:
                    schema[key] = (
                        pl.Array(pl.Float64, 3)
                        if isinstance(value, list)
                        else pl.Float64
                    )
        rows = [{key: row.get(key) for key in schema} for row in self.__rows]
        return pl.DataFrame(rows, schema=schema)

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

    # -- restarts ---------------------------------------------------------

    @staticmethod
    def __restart_settings(
        restart: str | os.PathLike | None, every: int | None
    ) -> tuple[str, int] | None:
        if restart is None:
            if every is not None:
                raise InvalidParameter("restart_every needs restart, the file")
            return None
        if every is None:
            every = 1000
        if isinstance(every, bool) or not isinstance(every, int) or every < 1:
            raise InvalidParameter(
                f"restart_every must be a positive integer, not {every!r}"
            )
        if not isinstance(restart, (str, os.PathLike)):
            raise InvalidParameter(f"restart must be a path, not {restart!r}")
        return os.fspath(restart), every

    def __native_only(self) -> None:
        if not isinstance(self.__engine, _Native):
            raise InvalidParameter(
                "restart files keep runs in MDAPY; LAMMPS writes its own, with "
                "md.command('write_restart file')"
            )

    def __state(self, pending: tuple[dict[str, Any], int] | None) -> dict[str, Any]:
        self.__native_only()
        engine = self.__engine
        return {
            "timestep": self.timestep,
            "system": engine.system,
            "engine": engine.state(),
            "rows": self.__rows,
            "runs": self.__runs,
            "saved": self.__saved,
            "pending": pending,
        }

    def __write_restart(
        self, path: str, pending: tuple[dict[str, Any], int] | None
    ) -> None:
        _restart.write(path, self.__state(pending), first=self.__potential)

    def save(self, path: str | os.PathLike) -> None:
        """Writes the state of the dynamics to a restart file, between runs,
        for ``load`` to go on from.

        Parameters
        ----------
        path : str or os.PathLike
            The file, replaced once the new one is whole.

        Raises
        ------
        InvalidParameter
            While the frames of an ``iterate`` are being read, for a
            LAMMPS calculator, or if the file cannot be written.
        """
        self.__idle()
        self.__write_restart(os.fspath(path), self.__pending)

    @classmethod
    def load(
        cls,
        path: str | os.PathLike,
        potential: Any,
        *,
        others: Any = (),
        log: Any = None,
    ) -> "MolecularDynamics":
        """The dynamics of a restart file, to go on from where it was written.

        Parameters
        ----------
        path : str or os.PathLike
            A restart file that this version of MDAPY wrote.
        potential : Calculator
            The potential of the dynamics, as given when it began.
        others : sequence of Calculator, optional
            The other potentials its runs use, those of ``Switch`` controls,
            in the order they first came. Default none.
        log : str, os.PathLike or stream, optional
            The log, added to rather than overwritten. Default None.

        Returns
        -------
        MolecularDynamics
            At the step the file was written, with the thermodynamic state
            recorded so far; ``resume`` finishes a run the file stopped in.

        Raises
        ------
        InvalidParameter
            For a file that is not a restart file of this version of MDAPY,
            a potential of another kind than the file's, or too few others.
        """
        state, names = _restart.read(path, [potential, *others])
        if names and not names[0].startswith(type(potential).__name__ + "("):
            raise InvalidParameter(
                f"this restart file ran with {names[0]}, not a {type(potential).__name__}"
            )
        md = cls(state["system"], potential, timestep=state["timestep"])
        if state["pending"] is not None:
            # the controls of the run set up, on the GPU too, as at its start;
            # its variables then as the file kept them
            md.__engine.start(state["pending"][0])
        md.__engine.restore(state["engine"])
        md.__rows = list(state["rows"])
        md.__runs = state["runs"]
        md.__saved = state["saved"]
        md.__pending = state["pending"]
        md.__log = _Log(log, active=md.__engine.writes, append=True)
        if md.__log.on:
            md.__log.comment("", f"continued from {os.fspath(path)} at step {md.step}")
        return md

    def resume(
        self, *, progress: Any = None, cancel: CancelToken | None = None
    ) -> None:
        """Finishes the run that a restart file stopped in, with its settings,
        controls and restart files, as if it had not stopped.

        Parameters
        ----------
        progress : callable, optional
            Called as ``progress(done, total)`` with the steps of the run.
        cancel : CancelToken, optional
            Stops the run, which raises ``mdapy.errors.Cancelled``.

        Raises
        ------
        InvalidParameter
            If no run was stopped: the file was written between runs.
        """
        if self.__pending is None:
            raise InvalidParameter(
                "no run to resume: the restart file was written between runs"
            )
        self.__idle()
        spec, done = self.__pending
        self.__pending = None
        self.__frames.extend(self.__steps(spec, progress, cancel, resumed=done))

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
        keep_momentum: bool = False,
        hugoniot: tuple[float, float, float] | None = None,
        shock_velocity: float | None = None,
        msst: dict[str, float] | None = None,
        thermo: int = 100,
        save_every: int | None = None,
        controls: Any = None,
        progress: Any = None,
        cancel: CancelToken | None = None,
        restart: str | os.PathLike | None = None,
        restart_every: int | None = None,
    ) -> None:
        """Runs molecular dynamics.

        Parameters
        ----------
        steps : int32
            Number of steps.
        ensemble : {"nve", "nvt", "npt", "nph", "nphug", "msst"}, optional
            Constant energy, or a thermostat, and barostat, or a barostat
            alone, or a barostat and a thermostat that keeps the atoms on the
            Hugoniot of a reference state, as LAMMPS's fix nphug, or the box
            compressed along an axis as by a steady shock of shock_velocity,
            as fix msst. Default "nve".
        temperature : float64 or (float64, float64), optional
            Temperature in K of the thermostat, or its values at the start
            and the end of the run; needed for "nvt" and "npt".
        pressure : float64 or (float64, float64), optional
            Pressure in GPa of the barostat, or its values at the start and
            the end; needed for "npt", "nph" and "nphug", one value for
            "nphug".
        barostat : {"iso", "aniso", "tri", "xy", "x", "y", "z"}, optional
            The box changes alike in every direction, along each axis, in
            any way, alike along its first two vectors with the third kept,
            or along one axis alone, the others kept. A sheet, periodic along
            its first two vectors only, changes in its plane only: alike
            along them for "iso" and "xy", along each for "aniso", and in
            any way in the plane for "tri". Along one axis the box needs its
            first vector along x and its second in the xy plane, periodic
            along that axis. "nphug" takes "iso", "aniso" and "tri" under
            the same pressure along every axis, or one axis for a uniaxial
            compression; "msst" the axis of the shock. Default "iso".
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
        keep_momentum : bool, optional
            With Langevin dynamics, random forces that sum to zero, which keep
            the total momentum, as LAMMPS's fix langevin zero yes; with no
            atoms held or moved. Default False.
        hugoniot : (float64, float64, float64), optional
            For "nphug" and "msst", the reference state of the Hugoniot: its
            total energy in eV, volume in Å³ and pressure in GPa, along the
            axis of a barostat along one axis or of the shock. Default None,
            the state at the start of the run.
        shock_velocity : float64, optional
            For "msst", the velocity of the shock in Å/ps, not negative.
        msst : dict, optional
            For "msst", its settings: "q", the mass of the box in amu²/Å⁴
            (default 10), "mu", an artificial viscosity in amu/(Å ps)
            (default 0), and "tscale", the share of the kinetic energy given
            to the compression at the start (default 0.01), as in fix msst.
        thermo : int32, optional
            Records the thermodynamic state every this many steps, and at
            the start and the end. Default 100.
        save_every : int32, optional
            Keeps the atoms every this many steps, in ``frames``. Default
            None, keeping none.
        controls : sequence of controls, optional
            Hold, Move, AddForce, Langevin, Deform or AdaptiveTimestep: what
            is done to groups of atoms, the box or the time step during the
            run: atoms held, moved at a velocity, pushed, or kept at a
            temperature, the box deformed at constant rates, the time step
            set by the fastest atom. Held and
            moved atoms take no part in the thermostat, and the temperature
            counts the directions that move freely; held or moved atoms not
            with a barostat or a deformation, baths not with the thermostat
            of "nvt" or "npt". Default none.
        progress : callable, optional
            Called as ``progress(done, total)`` with the steps run.
        cancel : CancelToken, optional
            Stops the run, which raises ``mdapy.errors.Cancelled``.
        restart : str or os.PathLike, optional
            Writes the state of the run to this file every ``restart_every``
            steps and at its end, each time replacing the last whole, for
            ``MolecularDynamics.load`` to go on from; in MDAPY only. Default
            None, writing none.
        restart_every : int32, optional
            Steps between restart files. Default 1000 with ``restart``.

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
            controls,
            keep_momentum,
            hugoniot,
            shock_velocity,
            msst,
        )
        spec["restart"] = self.__restart_settings(restart, restart_every)
        if spec["restart"] is not None:
            self.__native_only()
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
        keep_momentum: bool = False,
        hugoniot: tuple[float, float, float] | None = None,
        shock_velocity: float | None = None,
        msst: dict[str, float] | None = None,
        thermo: int = 100,
        controls: Any = None,
        progress: Any = None,
        cancel: CancelToken | None = None,
        restart: str | os.PathLike | None = None,
        restart_every: int | None = None,
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
        ensemble : {"nve", "nvt", "npt", "nph", "nphug", "msst"}, optional
            Constant energy, or a thermostat, and barostat, or a barostat
            alone, or a barostat and a thermostat that keeps the atoms on the
            Hugoniot of a reference state, as LAMMPS's fix nphug, or the box
            compressed along an axis as by a steady shock of shock_velocity,
            as fix msst. Default "nve".
        temperature : float64 or (float64, float64), optional
            Temperature in K of the thermostat, or its values at the start
            and the end of the run; needed for "nvt" and "npt".
        pressure : float64 or (float64, float64), optional
            Pressure in GPa of the barostat, or its values at the start and
            the end; needed for "npt", "nph" and "nphug", one value for
            "nphug".
        barostat : {"iso", "aniso", "tri", "xy", "x", "y", "z"}, optional
            The box changes alike in every direction, along each axis, in
            any way, alike along its first two vectors with the third kept,
            or along one axis alone, the others kept. A sheet, periodic along
            its first two vectors only, changes in its plane only: alike
            along them for "iso" and "xy", along each for "aniso", and in
            any way in the plane for "tri". Along one axis the box needs its
            first vector along x and its second in the xy plane, periodic
            along that axis. "nphug" takes "iso", "aniso" and "tri" under
            the same pressure along every axis, or one axis for a uniaxial
            compression; "msst" the axis of the shock. Default "iso".
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
        keep_momentum : bool, optional
            With Langevin dynamics, random forces that sum to zero, which keep
            the total momentum, as LAMMPS's fix langevin zero yes; with no
            atoms held or moved. Default False.
        hugoniot : (float64, float64, float64), optional
            For "nphug" and "msst", the reference state of the Hugoniot: its
            total energy in eV, volume in Å³ and pressure in GPa, along the
            axis of a barostat along one axis or of the shock. Default None,
            the state at the start of the run.
        shock_velocity : float64, optional
            For "msst", the velocity of the shock in Å/ps, not negative.
        msst : dict, optional
            For "msst", its settings: "q", the mass of the box in amu²/Å⁴
            (default 10), "mu", an artificial viscosity in amu/(Å ps)
            (default 0), and "tscale", the share of the kinetic energy given
            to the compression at the start (default 0.01), as in fix msst.
        thermo : int32, optional
            Records the thermodynamic state every this many steps, and at
            the start and the end. Default 100.
        controls : sequence of controls, optional
            Hold, Move, AddForce, Langevin, Deform or AdaptiveTimestep: what
            is done to groups of atoms, the box or the time step during the
            run: atoms held, moved at a velocity, pushed, or kept at a
            temperature, the box deformed at constant rates, the time step
            set by the fastest atom. Held and
            moved atoms take no part in the thermostat, and the temperature
            counts the directions that move freely; held or moved atoms not
            with a barostat or a deformation, baths not with the thermostat
            of "nvt" or "npt". Default none.
        progress : callable, optional
            Called as ``progress(done, total)`` with the steps run.
        cancel : CancelToken, optional
            Stops the run, which raises ``mdapy.errors.Cancelled``.
        restart : str or os.PathLike, optional
            Writes a restart file, as ``run`` does. Default None.
        restart_every : int32, optional
            Steps between restart files. Default 1000 with ``restart``.

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
            controls,
            keep_momentum,
            hugoniot,
            shock_velocity,
            msst,
        )
        spec["restart"] = self.__restart_settings(restart, restart_every)
        if spec["restart"] is not None:
            self.__native_only()
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
        controls: Any = None,
        keep_momentum: bool = False,
        hugoniot: tuple[float, float, float] | None = None,
        shock_velocity: float | None = None,
        msst: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """The checked options of a run."""
        checked = _controls.checked(controls, self.__n)
        if ensemble in ("nvt", "npt", "nphug") and any(
            isinstance(c, _controls.Langevin) for c in checked
        ):
            raise InvalidParameter(
                "Langevin controls take the place of the thermostat: use them with "
                "nve or nph"
            )
        reference = _hugoniot(hugoniot, ensemble)
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
        if not isinstance(keep_momentum, bool):
            raise InvalidParameter(
                f"keep_momentum must be a bool, not {keep_momentum!r}"
            )
        if keep_momentum and not (
            thermostat == "langevin" and ensemble in ("nvt", "npt")
        ):
            raise InvalidParameter(
                'keep_momentum is for the "langevin" thermostat of "nvt" or "npt"'
            )
        if keep_momentum and any(
            isinstance(c, (_controls.Hold, _controls.Move)) for c in checked
        ):
            raise InvalidParameter(
                "keep_momentum with atoms held or moved: there is no momentum to keep"
            )
        if int(thermo) < 1 or (save_every is not None and int(save_every) < 1):
            raise InvalidParameter("thermo and save_every must be positive")
        if seed is not None and int(seed) < 1:
            raise InvalidParameter(f"seed must be positive, not {seed!r}")
        temperatures = _pair(temperature, "temperature")
        pressures = _pair(pressure, "pressure")
        if ensemble in ("nvt", "npt") and temperatures is None:
            raise InvalidParameter(f"{ensemble} needs a temperature")
        if ensemble in ("npt", "nph", "nphug") and pressures is None:
            raise InvalidParameter(f"{ensemble} needs a pressure")
        if ensemble == "nphug":
            _check_nphug(pressures, temperatures, thermostat, barostat, self.__pbc)
        if ensemble != "msst" and (shock_velocity is not None or msst is not None):
            raise InvalidParameter(
                'shock_velocity and msst are for the ensemble "msst"'
            )
        settings = None
        if ensemble == "msst":
            settings = _check_msst(
                {
                    "shock_velocity": shock_velocity,
                    "msst": msst,
                    "pressures": pressures,
                    "temperatures": temperatures,
                },
                barostat,
                self.__pbc,
                checked,
            )
        if temperatures is not None and ensemble in ("nvt", "npt"):
            if not min(temperatures) > 0:
                raise InvalidParameter(
                    f"the temperature must be positive, not {temperature!r}"
                )
        sheet = bool(self.__pbc[:2].all() and not self.__pbc[2])
        if ensemble in ("npt", "nph", "nphug"):
            if barostat in ("x", "y", "z") and not self.__pbc["xyz".index(barostat)]:
                raise InvalidParameter(
                    f"a barostat along {barostat} needs a box periodic along it"
                )
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
            "keep_momentum": bool(keep_momentum),
            "hugoniot": reference,
            "shock_velocity": None if shock_velocity is None else float(shock_velocity),
            "msst": settings,
            "thermo": int(thermo),
            "save_every": None if save_every is None else int(save_every),
            "controls": checked,
        }
        return spec

    def __steps(
        self,
        spec: dict[str, Any],
        progress: Any,
        cancel: CancelToken | None,
        resumed: int | None = None,
    ) -> Iterator[System]:
        """Runs, recording the thermodynamic state, and gives the atoms every
        save_every steps; the run ends where the frames stop being read. A
        run resumed from a restart goes on from the steps it had run."""
        engine = self.__engine
        steps, thermo, every = spec["steps"], spec["thermo"], spec["save_every"]
        restart, restart_every = spec.get("restart") or (None, None)
        done = 0 if resumed is None else resumed
        finished = False
        log = self.__log
        if resumed is None:
            self.__runs += 1
        if log.on:
            if resumed is None:
                log.comment("", self.__settings(self.__runs, spec))
            else:
                log.comment(
                    "", f"run {self.__runs} resumed after {resumed} of {steps} steps"
                )
            log.columns(_LOG_COLUMNS)
        began = time.perf_counter()
        try:
            if resumed is None:
                engine.start(spec)
                start = engine.step
                if not self.__rows or self.__rows[-1]["step"] != start:
                    self.__record()
                if every is not None and self.__saved != start:
                    self.__saved = start
                    yield engine.system
            time0 = self.__rows[-1]["time"]
            stops = set(range(thermo, steps, thermo)) | {steps}
            if every is not None:
                stops |= set(range(every, steps + 1, every))
            if restart is not None:
                stops |= set(range(restart_every, steps, restart_every))
            stops = {s for s in stops if s > done} | ({0} if steps == 0 else set())
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
                if restart is not None and done % restart_every == 0 and done < steps:
                    self.__write_restart(restart, (spec, done))
            finished = True
        finally:
            if not engine.closed:
                engine.finish()
            if finished and restart is not None:
                self.__write_restart(restart, None)
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
        if ensemble == "msst":
            settings = ", ".join(f"{k} {v!r}" for k, v in spec["msst"].items())
            parts.append(
                f"shock along {spec['barostat']} at {spec['shock_velocity']!r} Å/ps, "
                f"{settings}"
            )
        if ensemble == "nphug":
            parts.append(
                f"thermostat on the Hugoniot with damping "
                f"{spec['thermostat_damping']!r} ps"
            )
        if ensemble in ("npt", "nph", "nphug"):
            parts.append(
                f"{span(spec['pressures'], 'GPa')}, barostat {spec['barostat']} with "
                f"damping {spec['barostat_damping']!r} ps"
            )
        parts.extend(repr(control) for control in spec["controls"])
        parts.append(f"thermo every {spec['thermo']}")
        if spec["save_every"] is not None:
            parts.append(f"atoms every {spec['save_every']}")
        return "; ".join(parts)
