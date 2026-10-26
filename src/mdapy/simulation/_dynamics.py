# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Molecular dynamics in LAMMPS, with the interaction of a LAMMPS calculator."""

import weakref
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

_ENSEMBLES = ("nve", "nvt", "npt")
_BAROSTATS = ("iso", "aniso", "tri")
# thermo keywords of LAMMPS and the columns they become
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


class MolecularDynamics:
    """Molecular dynamics of a system in LAMMPS, with the interaction of a
    LAMMPS calculator.

    Parameters
    ----------
    system : System
        The atoms, with an ``element`` column; their ``velocities``, if the
        system has them, in Å/ps.
    potential : LAMMPS
        The interaction: its commands, the element of each atom type and
        the command-line arguments of LAMMPS.
    timestep : float, optional
        Time step in ps. Default 0.001.

    Attributes
    ----------
    timestep : float
        Time step in ps.

    Raises
    ------
    InvalidParameter
        If the time step is not positive, an element has no atom type, or
        LAMMPS rejects the commands.
    MissingProperty
        If the system has no ``element`` column.
    ImportError
        If the LAMMPS Python module is not installed.
    """

    def __init__(self, system: System, potential: LAMMPS, *, timestep: float = 0.001):
        if not isinstance(potential, LAMMPS):
            raise InvalidParameter("potential must be a mdapy.calculators.LAMMPS")
        if not timestep > 0:
            raise InvalidParameter(f"timestep must be positive, not {timestep!r}")
        if "element" not in system.table.columns:
            raise MissingProperty("molecular dynamics needs the element column")
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
        self.timestep = float(timestep)
        self.__n = len(system)
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
        self.__rows: list[dict] = []
        self.__frames: list[System] = []
        self.__holder: list = []
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
boundary {" ".join("p" if p else "s" for p in pbc)}
region box prism 0 {lx!r} 0 {ly!r} 0 {lz!r} {xy!r} {xz!r} {yz!r}
create_box {len(potential.elements)} box
"""
                + "".join(f"mass {k + 1} {m!r}\n" for k, m in enumerate(masses))
                + potential.commands
                + "\n"
                + potential.pair
                + f"\ntimestep {self.timestep!r}"
                + "\nthermo_style custom step "
                + " ".join(k for k, _ in _THERMO)
                + " pxx pyy pzz pyz pxz pxy"
                + "\nthermo_modify norm no"
            )
            wrapped, _ = box.wrap(system["positions"])
            x = wrapped @ self.__rotation - self.__lo
            ids = list(range(1, self.__n + 1))
            if "velocities" in system.table.columns:
                v = system["velocities"] @ self.__rotation
                lmp.create_atoms(
                    self.__n,
                    ids,
                    types.tolist(),
                    x.ravel().tolist(),
                    v.ravel().tolist(),
                )
            else:
                lmp.create_atoms(self.__n, ids, types.tolist(), x.ravel().tolist())
        except Exception as error:
            self.close()
            raise InvalidParameter(f"LAMMPS: {error}") from None

    # -- the LAMMPS instance --------------------------------------------

    @property
    def lammps(self) -> Any:
        """The LAMMPS instance, for anything else."""
        if not self.__holder:
            raise InvalidParameter("this molecular dynamics is closed")
        return self.__holder[0]

    def close(self) -> None:
        """Closes the LAMMPS instance."""
        _close(self.__holder)

    def __enter__(self) -> "MolecularDynamics":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "closed" if not self.__holder else f"step {self.step}"
        return f"MolecularDynamics({self.__n} atoms, {state})"

    def command(self, commands: str) -> None:
        """Runs LAMMPS commands, one or several lines.

        Parameters
        ----------
        commands : str
            The commands.

        Raises
        ------
        InvalidParameter
            If LAMMPS rejects them.
        """
        try:
            self.lammps.commands_string(commands)
        except Exception as error:
            raise InvalidParameter(f"LAMMPS: {error}") from None

    # -- reading the state ----------------------------------------------

    @property
    def step(self) -> int:
        """The time step number of LAMMPS: the steps run so far."""
        return int(self.lammps.extract_global("ntimestep"))

    @property
    def system(self) -> System:
        """The atoms now, with their velocities and periodic images, and the
        step and time in ``info``."""
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
        info = self.__info | {"timestep": step, "time": step * self.timestep}
        return System(table=table, box=Box(matrix, origin=origin, pbc=pbc), info=info)

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
        """The atoms every ``frames`` steps of the runs."""
        return tuple(self.__frames)

    def __record(self) -> None:
        lmp = self.lammps
        step = self.step
        row = {"step": step, "time": step * self.timestep}
        for keyword, name in _THERMO:
            row[name] = float(lmp.get_thermo(keyword))
        row["pressure"] *= units.BAR_TO_GPA
        p = [lmp.get_thermo(k) for k in ("pxx", "pyy", "pzz", "pyz", "pxz", "pxy")]
        tensor = np.array([[p[0], p[5], p[4]], [p[5], p[1], p[3]], [p[4], p[3], p[2]]])
        stress = -(self.__rotation @ tensor @ self.__rotation.T) * units.BAR_TO_GPA
        row["stress"] = [
            stress[a, b] for a, b in ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
        ]
        self.__rows.append(row)

    # -- running ----------------------------------------------------------

    def velocities(self, temperature: float, *, seed: int) -> None:
        """Gives the atoms random velocities of a temperature, with no
        total momentum.

        Parameters
        ----------
        temperature : float
            Temperature in K, not negative.
        seed : int
            Seed of the random velocities, positive.

        Raises
        ------
        InvalidParameter
            For a negative temperature or a seed that is not positive.
        """
        if not temperature >= 0:
            raise InvalidParameter(
                f"temperature must not be negative, not {temperature!r}"
            )
        if int(seed) < 1:
            raise InvalidParameter(f"seed must be positive, not {seed!r}")
        self.command(
            f"velocity all create {float(temperature)!r} {int(seed)} "
            f"dist gaussian mom yes rot no loop all"
        )

    def run(
        self,
        steps: int,
        *,
        ensemble: str = "nve",
        temperature: float | tuple[float, float] | None = None,
        pressure: float | tuple[float, float] | None = None,
        barostat: str = "iso",
        thermostat_damping: float | None = None,
        barostat_damping: float | None = None,
        thermo: int = 100,
        frames: int | None = None,
        progress: Any = None,
        cancel: CancelToken | None = None,
    ) -> None:
        """Runs molecular dynamics.

        Parameters
        ----------
        steps : int
            Number of steps.
        ensemble : {"nve", "nvt", "npt"}, optional
            Constant energy, or a Nosé-Hoover thermostat, and barostat.
            Default "nve".
        temperature : float or (float, float), optional
            Temperature in K of the thermostat, or its values at the start
            and the end of the run; needed for "nvt" and "npt".
        pressure : float or (float, float), optional
            Pressure in GPa of the barostat, or its values at the start and
            the end; needed for "npt".
        barostat : {"iso", "aniso", "tri"}, optional
            The box changes alike in every direction, along each axis, or
            in any way. Default "iso".
        thermostat_damping : float, optional
            Damping time of the thermostat in ps. Default 100 time steps.
        barostat_damping : float, optional
            Damping time of the barostat in ps. Default 1000 time steps.
        thermo : int, optional
            Records the thermodynamic state every this many steps, and at
            the start and the end. Default 100.
        frames : int, optional
            Keeps the atoms every this many steps. Default None, keeping
            none.
        progress : callable, optional
            Called as ``progress(done, total)`` with the steps run.
        cancel : CancelToken, optional
            Stops the run, which raises :class:`mdapy.errors.Cancelled`.

        Raises
        ------
        InvalidParameter
            For invalid options, or if LAMMPS fails.
        Cancelled
            If cancelled.
        """
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
        if int(thermo) < 1 or (frames is not None and int(frames) < 1):
            raise InvalidParameter("thermo and frames must be positive")
        temperatures = _pair(temperature, "temperature")
        pressures = _pair(pressure, "pressure")
        if ensemble in ("nvt", "npt") and temperatures is None:
            raise InvalidParameter(f"{ensemble} needs a temperature")
        if ensemble == "npt" and pressures is None:
            raise InvalidParameter("npt needs a pressure")
        if barostat != "iso" and ensemble == "npt" and not self.__pbc.all():
            raise InvalidParameter("a barostat needs a box periodic in every direction")
        tdamp = (
            100 * self.timestep if thermostat_damping is None else thermostat_damping
        )
        pdamp = 1000 * self.timestep if barostat_damping is None else barostat_damping
        if ensemble == "nve":
            fix = "nve"
        else:
            t0, t1 = temperatures
            fix = f"{ensemble} temp {t0!r} {t1!r} {float(tdamp)!r}"
            if ensemble == "npt":
                p0, p1 = (p * units.GPA_TO_BAR for p in pressures)
                fix += f" {barostat} {p0!r} {p1!r} {float(pdamp)!r}"
        self.command(f"fix mdapy_integrate all {fix}")
        thermo, start, done = int(thermo), self.step, 0
        try:
            self.command("run 0 post no")
            if not self.__rows or self.__rows[-1]["step"] != start:
                self.__record()
            if frames is not None and (
                not self.__frames or self.__frames[-1].info["timestep"] != start
            ):
                self.__frames.append(self.system)
            stops = set(range(thermo, steps, thermo)) | {steps}
            if frames is not None:
                stops |= set(range(int(frames), steps + 1, int(frames)))
            for stop in sorted(stops):
                if cancel is not None and cancel.cancelled:
                    raise Cancelled("molecular dynamics cancelled")
                self.command(
                    f"run {stop - done} start {start} stop {start + steps} pre no post no"
                )
                done = stop
                if done % thermo == 0 or done == steps:
                    self.__record()
                if frames is not None and done % int(frames) == 0:
                    self.__frames.append(self.system)
                if progress is not None:
                    progress(done, steps)
        finally:
            if self.__holder:
                self.command("unfix mdapy_integrate")
