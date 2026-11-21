# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Path-integral molecular dynamics in MDAPY: each atom a ring polymer of P
beads, neighbors on the ring tied by springs of frequency w_P = P k_B T /
hbar, each bead feeling the whole potential, sampled at the temperature
P T (Ceriotti et al., J. Chem. Phys. 133, 124104 (2010)). The steps are
BAOAB in the normal modes of the free ring polymer: half a kick of the
forces, half a step of the free ring polymer, exact, the PILE_L thermostat
of each mode, the other half of the free ring polymer, the forces, the
other half kick. Without the thermostat it is ring-polymer molecular
dynamics."""

import math
import secrets
from typing import Any

import numpy as np
import polars as pl

from mdapy import _core, units
from mdapy.calculators._base import WITHOUT_VIRIAL, Calculator
from mdapy.core.system import System
from mdapy.core.threads import get_num_threads
from mdapy.elements import atomic_masses
from mdapy.errors import CalculationFailed, InvalidParameter, MissingProperty
from mdapy.neighbor import NeighborList
from mdapy.simulation import _integrator
from mdapy.simulation._log import _on_gpu

_native = _core.core
_HBAR = units.PLANCK / (2.0 * math.pi)  # eV ps
_PIMD_STREAM = 2  # of the random numbers of the thermostat of the normal modes
_MOVING = ("positions", "velocities", "image", "forces", "atomic_energies",
           "atomic_virials")  # fmt: skip


def normal_modes(beads: int) -> tuple[np.ndarray, np.ndarray]:
    """The orthogonal (P, P) matrix C that takes the beads to the normal
    modes of the free ring polymer, q~ = C q, and the frequency of each mode
    over w_P, 2 sin(pi k / P): the centroid first, (1 / sqrt(P)) the sum,
    then cosines up to P / 2, the alternating mode of an even P, and
    sines."""
    p = beads
    j = np.arange(p)
    c = np.zeros((p, p))
    c[0] = 1.0 / math.sqrt(p)
    for k in range(1, p):
        if 2 * k < p:
            c[k] = math.sqrt(2.0 / p) * np.cos(2.0 * math.pi * k * j / p)
        elif 2 * k == p:
            c[k] = (-1.0) ** j / math.sqrt(p)
        else:
            c[k] = math.sqrt(2.0 / p) * np.sin(2.0 * math.pi * k * j / p)
    return c, 2.0 * np.sin(math.pi * np.arange(p) / p)


def _same(values: np.ndarray) -> np.ndarray:
    """What a process sends to its ghosts, or back from them: without
    ghosts, the values as they are."""
    return values


class _RingPolymer:
    """The state of a path-integral dynamics in MDAPY and its steps, with
    the interface of _Native for MolecularDynamics."""

    def __init__(
        self, system: System, calculator: Calculator, timestep: float, beads: int
    ):
        if not isinstance(calculator, Calculator):
            raise InvalidParameter("potential must be an MDAPY calculator")
        from mdapy.simulation import _domain

        # under MPI the beads are shared among the processes, each computing
        # the forces of its own, and every process takes the same steps
        self.__comm = _domain.world()
        table = system.table
        n = len(system)
        if "mass" in table.columns:
            masses = table["mass"].to_numpy().astype(float)
        elif "element" in table.columns:
            masses = atomic_masses(table["element"].to_physical().to_numpy())
        else:
            raise MissingProperty(
                "molecular dynamics needs the masses: a mass or an element column"
            )
        if "movable" in table.columns and not table["movable"].to_numpy().all():
            raise InvalidParameter(
                "path-integral dynamics moves every atom: no movable column"
            )
        self.__properties = tuple(
            p for p in ("energy", "forces", "virial") if p in calculator.implemented
        )
        if not {"energy", "forces"} <= set(self.__properties):
            raise InvalidParameter(
                f"{type(calculator).__name__} does not compute energies and forces"
            )
        self.__n, self.__p = n, int(beads)
        self.__calculator = calculator
        self.__masses = masses
        self.__table = table.drop([c for c in _MOVING if c in table.columns])
        self.__info = {
            k: v
            for k, v in system.info.items()
            if k not in ("energy", "virial", "stress")
        }
        self.__box = system.box
        p = self.__p
        x = np.ascontiguousarray(system["positions"], dtype=float)
        v = (
            np.asarray(system["velocities"], dtype=float)
            if "velocities" in table.columns
            else np.zeros((n, 3))
        )
        # every bead at the atom, with its velocity
        self.__x = np.repeat(x[None], p, axis=0)
        self.__v = np.repeat(v[None], p, axis=0)
        self.__f = np.zeros((p, n, 3))
        self.__energies = np.zeros(p)  # of the potential at each bead
        self.__virials = np.full((p, 3, 3), np.nan)
        # the atoms of each bead, with its neighbor lists, for its forces
        self.__beads = [
            _integrator._HostAtoms(calculator, self.__x[k], self.__v[k], masses,
                                   lambda vectors, rows=None: vectors)
            for k in range(p)
        ]  # fmt: skip
        # MDAPY's potentials on the processor compute each bead from arrays,
        # with a neighbor list of its own kept until a bead has moved half
        # the skin: (list, positions when built) of each bead
        self.__direct = (
            callable(getattr(calculator, "_split", None))
            and getattr(calculator, "_takes_wider_lists", False)
            and calculator.device == "cpu"
        )
        self.__lists: list = [None] * p
        if self.__direct:
            self.__species = np.ascontiguousarray(calculator._species(system))
            self.__ids = np.arange(n, dtype=np.int64)
        size = 1 if self.__comm is None else self.__comm.Get_size()
        rank = 0 if self.__comm is None else self.__comm.Get_rank()
        self.__mine = [k for k in range(p) if k % size == rank]
        self.__modes, self.__frequencies = normal_modes(p)
        self.timestep = float(timestep)
        self.__step = 0
        self.__time = 0.0
        self.__closed = False
        self.__run: dict[str, Any] = {}
        self.__temperature: float | None = None  # of the springs, K
        self.__thermostat = False
        self.__centroids = True
        self.__seed = 0
        self.__draws = 0

    # -- what MolecularDynamics asks of an engine ------------------------------

    def state(self) -> dict[str, Any]:
        """Every variable but the calculator, the atoms of the beads and
        the neighbor lists."""
        skip = (
            "_RingPolymer__calculator",
            "_RingPolymer__beads",
            "_RingPolymer__lists",
            "_RingPolymer__comm",
            "_RingPolymer__mine",
        )
        return {"pimd": {k: v for k, v in vars(self).items() if k not in skip}}

    def restore(self, state: dict[str, Any]) -> None:
        vars(self).update(state["pimd"])
        self.__lists = [None] * self.__p
        for k, bead in enumerate(self.__beads):
            bead.restore({"_x": self.__x[k], "_v": self.__v[k]})

    def describe(self) -> str:
        calculator = self.__calculator
        if _on_gpu(calculator) is not None:
            where = f"the forces on the GPU, {_on_gpu(calculator)}"
        else:
            where = f"the processor, {get_num_threads()} threads"
        if self.__comm is not None:
            where += f", the beads on {self.__comm.Get_size()} processes"
        return f"MDAPY, path integrals of {self.__p} beads; {where}"

    @property
    def writes(self) -> bool:
        """Whether this process writes the log and files: the main one."""
        return self.__comm is None or self.__comm.Get_rank() == 0

    def anywhere(self, flag: bool) -> bool:
        """Whether any process has the flag."""
        if self.__comm is None:
            return flag
        return bool(self.__comm.allreduce(int(flag)))

    def shared(self, value: Any) -> Any:
        """What the main process has, on every process."""
        return value if self.__comm is None else self.__comm.bcast(value, root=0)

    @property
    def closed(self) -> bool:
        return self.__closed

    def close(self) -> None:
        self.__closed = True

    @property
    def step(self) -> int:
        return self.__step

    @property
    def beads(self) -> int:
        return self.__p

    def __system(self, x: np.ndarray, v: np.ndarray, info: dict) -> System:
        box = self.__box
        wrapped, image = box.wrap(x)
        table = self.__table.with_columns(
            pl.Series("positions", wrapped, dtype=pl.Array(pl.Float64, 3)),
            pl.Series("velocities", v, dtype=pl.Array(pl.Float64, 3)),
            pl.Series("image", image.astype(np.int32), dtype=pl.Array(pl.Int32, 3)),
        )
        info = self.__info | {"timestep": self.__step, "time": self.__time} | info
        return System(table=table, box=box, info=info)

    @property
    def system(self) -> System:
        """The centroids: the mean place and velocity of the beads of each
        atom."""
        return self.__system(self.__x.mean(axis=0), self.__v.mean(axis=0), {})

    def bead_systems(self) -> list[System]:
        """The atoms at each bead, its index in info["bead"]."""
        return [
            self.__system(self.__x[k], self.__v[k], {"bead": k})
            for k in range(self.__p)
        ]

    # -- velocities ------------------------------------------------------------

    def velocities(self, temperature: float, seed: int) -> None:
        """Gaussian velocities of the beads at P times the temperature, as
        the ring polymer samples them, with no total momentum of the
        centroids: their temperature is the one given."""
        rng = np.random.default_rng(int(seed))
        m = self.__masses
        v = rng.normal(size=(self.__p, self.__n, 3)) / np.sqrt(m)[None, :, None]
        v -= (m @ v.mean(axis=0)) / m.sum()
        now = self.temperature_of(v)
        if temperature == 0.0 or not now > 0.0:
            v[:] = 0.0
        else:
            v *= math.sqrt(temperature / now)
        self.__v[:] = v

    def temperature_of(self, v: np.ndarray) -> float:
        """The temperature the velocities of the beads stand for: their
        kinetic energy over 3 N P k_B, divided by P."""
        twice = (
            float(np.einsum("i,kij,kij->", self.__masses, v, v)) * _integrator._MVV2E
        )
        return twice / (3 * self.__n * self.__p * _integrator._KB) / self.__p

    # -- forces ----------------------------------------------------------------

    def __compute(self, full: bool = True) -> None:
        """The forces on every bead; their energies, and their virials when
        full; each bead refuses forces that are not finite."""
        box = self.__box
        if self.__direct:
            self.__compute_direct(full)
        else:
            for k in self.__mine:
                bead = self.__beads[k]
                bead._x = self.__x[k]
                energy, virial = bead.compute(self.__table, box, True, full)
                self.__f[k] = bead.forces
                self.__energies[k] = energy
                self.__virials[k] = virial
        if self.__comm is not None:
            # the forces, energies and virials of every bead, on every process
            mine = self.__mine
            parts = self.__comm.allgather(
                (mine, self.__f[mine], self.__energies[mine], self.__virials[mine])
            )
            for beads, f, e, w in parts:
                self.__f[beads], self.__energies[beads] = f, e
                self.__virials[beads] = w

    def __compute_direct(self, full: bool) -> None:
        """Each bead from arrays, as a process of MPI computes its atoms,
        here without ghosts."""
        box, calculator = self.__box, self.__calculator
        matrix = np.ascontiguousarray(box.matrix)
        cutoff = float(calculator.cutoff)
        wanted = ("energy", "forces") + (("virial",) if full else (WITHOUT_VIRIAL,))
        for k in self.__mine:
            x = np.ascontiguousarray(self.__x[k])
            kept = self.__lists[k]
            if kept is not None:
                step = x - kept[1]
                moved = math.sqrt(float((step * step).sum(axis=1).max(initial=0.0)))
                if not 2.0 * moved < _integrator._SKIN:
                    kept = None
            if kept is None:
                current = System(table=self.__table.with_columns(
                    pl.Series("positions", x.copy(), dtype=pl.Array(pl.Float64, 3))
                ), box=box)  # fmt: skip
                kept = self.__lists[k] = (
                    NeighborList.build(current, cutoff + _integrator._SKIN),
                    x.copy(),
                )
            part = calculator._split((self.__species, x, matrix), wanted, kept[0],
                                     self.__n, _same, _same, self.__ids)  # fmt: skip
            forces = np.asarray(part.forces, dtype=float)
            if not (math.isfinite(part.energy) and np.isfinite(forces).all()):
                raise CalculationFailed(
                    "the energy or the forces are not finite: atoms may overlap"
                )
            self.__f[k] = forces
            self.__energies[k] = float(part.energy)
            self.__virials[k] = part.virial if full else np.full((3, 3), np.nan)

    # -- a run -----------------------------------------------------------------

    def start(self, run: dict[str, Any]) -> None:
        ensemble = run["ensemble"]
        if ensemble not in ("nvt", "nve"):
            raise InvalidParameter(
                f'path-integral dynamics runs "nvt" or "nve", not {ensemble!r}'
            )
        if ensemble == "nvt" and run["thermostat"] not in ("langevin", "trpmd"):
            raise InvalidParameter(
                'path-integral dynamics at a temperature takes the "langevin" '
                'thermostat, the PILE_L of the normal modes, or "trpmd"'
            )
        if run["controls"]:
            raise InvalidParameter("path-integral dynamics takes no controls")
        temperatures = run["temperatures"]
        if temperatures is None:
            raise InvalidParameter(
                "path-integral dynamics needs a temperature, that of the springs of "
                "its ring polymers"
            )
        t0, t1 = temperatures
        if t0 != t1 or not t0 > 0:
            raise InvalidParameter(
                "path-integral dynamics needs one positive temperature, not "
                f"{temperatures!r}"
            )
        self.__run = run
        self.__temperature = float(t0)
        self.__thermostat = ensemble == "nvt"
        # thermostatted ring-polymer molecular dynamics: the centroids left
        # to their dynamics, the other modes thermostatted
        self.__centroids = run["thermostat"] != "trpmd"
        self.__damping = float(run["thermostat_damping"])
        self.__dt = self.timestep
        # the time from the start of the run
        self.__atime, self.__astep = self.__time, self.__step
        if self.__thermostat:
            seed = run["seed"]
            self.__seed = (
                self.shared(secrets.randbits(63)) if seed is None else int(seed)
            )
            self.__draws = 0
        self.__compute()

    def finish(self) -> None:
        self.__run = {}

    def advance(self, count: int) -> None:
        for k in range(count):
            self.__one_step(k == count - 1)

    def __omega(self) -> float:
        """w_P in 1/ps."""
        return self.__p * _integrator._KB * self.__temperature / _HBAR

    def __to_modes(self, a: np.ndarray) -> np.ndarray:
        return np.einsum("kj,jia->kia", self.__modes, a)

    def __to_beads(self, a: np.ndarray) -> np.ndarray:
        return np.einsum("kj,kia->jia", self.__modes, a)

    def __free(self, q: np.ndarray, v: np.ndarray, h: float) -> None:
        """The free ring polymer for a time h, exactly, in the normal modes,
        in place: the centroid drifts, the others turn."""
        q[0] += h * v[0]
        w = self.__omega() * self.__frequencies[1:, None, None]
        c, s = np.cos(w * h), np.sin(w * h)
        q1, v1 = q[1:].copy(), v[1:].copy()
        q[1:] = c * q1 + s / w * v1
        v[1:] = -w * s * q1 + c * v1

    def __pile(self, v: np.ndarray, h: float) -> None:
        """The PILE_L thermostat of the normal modes for a time h, in place:
        the centroid damped in thermostat_damping, each other mode
        critically, 2 w_k, with the noise of P times the temperature."""
        centroid = 1.0 / self.__damping if self.__centroids else 0.0
        gamma = np.concatenate([[centroid],
                                2.0 * self.__omega() * self.__frequencies[1:]])  # fmt: skip
        c1 = np.exp(-gamma * h)[:, None, None]
        c2 = np.sqrt(1.0 - c1 * c1)
        kt = self.__p * _integrator._KB * self.__temperature * _integrator._FTM2V
        sigma = np.sqrt(kt / self.__masses)[None, :, None]
        ids = np.arange(self.__p * self.__n, dtype=np.int64)
        noise = _native.normals(ids, self.__seed, _PIMD_STREAM, self.__draws)
        self.__draws += 1
        v[:] = c1 * v + c2 * sigma * noise.reshape(self.__p, self.__n, 3)

    def __one_step(self, full: bool) -> None:
        dt = self.__dt
        kick = 0.5 * dt * _integrator._FTM2V / self.__masses[None, :, None]
        self.__v += kick * self.__f
        q, v = self.__to_modes(self.__x), self.__to_modes(self.__v)
        self.__free(q, v, 0.5 * dt)
        if self.__thermostat:
            self.__pile(v, dt)
        self.__free(q, v, 0.5 * dt)
        self.__x[:] = self.__to_beads(q)
        self.__v[:] = self.__to_beads(v)
        self.__step += 1
        self.__time = self.__atime + (self.__step - self.__astep) * dt
        self.__compute(full)
        self.__v += kick * self.__f

    # -- what it records -------------------------------------------------------

    def record(self) -> dict[str, Any]:
        """The quantum estimators: the potential energy, the mean over the
        beads; the kinetic energy of the centroid virial, and that of the
        primitive estimator; the pressure and the stress of the centroid
        virial; the temperature of the velocities of the beads. Of the forces
        of the last step of a run, which computes them at its start."""
        p, n = self.__p, self.__n
        kt = _integrator._KB * self.__temperature
        centroid = self.__x.mean(axis=0)
        away = self.__x - centroid[None]
        # sum over the beads of (q - q_c) f, over P
        spread = np.einsum("kia,kib->ab", away, self.__f) / p
        kinetic = 1.5 * n * kt - 0.5 * float(np.trace(spread))
        q = self.__to_modes(self.__x)
        omega = self.__omega()
        spring = (
            0.5
            * omega**2
            * float(
                np.einsum("k,i,kia,kia->", self.__frequencies**2, self.__masses, q, q)
            )
            * _integrator._MVV2E
        )
        primitive = 1.5 * n * p * kt - spring / p
        potential = float(self.__energies.mean())
        volume = float(self.__box.volume)
        tensor = (
            n * kt * np.eye(3) - 0.5 * (spread + spread.T)
            + self.__virials.mean(axis=0)
        ) / volume  # fmt: skip
        stress = -tensor * _integrator._EV_PER_A3_TO_GPA
        return {
            "step": self.__step,
            "time": self.__time,
            "temperature": self.temperature(),
            "potential_energy": potential,
            "kinetic_energy": kinetic,
            "total_energy": potential + kinetic,
            "pressure": float(np.trace(tensor)) / 3.0 * _integrator._EV_PER_A3_TO_GPA,
            "volume": volume,
            "stress": [float(stress[a, b]) for a, b in
                       ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))],
            "primitive_kinetic_energy": primitive,
            "spring_energy": spring,
            # what RPMD keeps: the kinetic energy of the beads, the energy of
            # the potential at each and of the springs
            "ring_polymer_energy": 0.5 * float(
                np.einsum("i,kia,kia->", self.__masses, self.__v, self.__v)
            ) * _integrator._MVV2E + float(self.__energies.sum()) + spring,
        }  # fmt: skip

    def temperature(self) -> float:
        return self.temperature_of(self.__v)
