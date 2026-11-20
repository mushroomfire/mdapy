# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Molecular dynamics in MDAPY itself, with any calculator: velocity Verlet,
Nosé-Hoover chains, Langevin dynamics and the barostat of Martyna, Tobias and
Klein, for small systems.

The equations of motion and their splitting follow Martyna, Tobias and Klein,
J. Chem. Phys. 101, 4177 (1994), Shinoda, Shiga and Mikami, Phys. Rev. B 69,
134103 (2004) and Tuckerman et al., J. Phys. A 39, 5629 (2006), with the
choices of LAMMPS for its fix nvt, npt and nph (chains of three, one loop, the
masses of the thermostats and barostat), so that both give the same runs; the
Langevin steps follow Bussi and Parrinello, Phys. Rev. E 75, 056707 (2007).
The work runs in the frame of the LAMMPS box (a along x, b in the xy plane),
where the barostat acts on the lengths and tilts of the box."""

import math
import secrets
from typing import Any, NamedTuple

import numpy as np
import polars as pl

from mdapy import units
from mdapy.calculators import _gpu
from mdapy.calculators._base import HEAT_FLUX_TERMS, Calculator
from mdapy.calculators._springs import Springs
from mdapy.core.box import Box
from mdapy.core.system import System
from mdapy.core.threads import get_num_threads
from mdapy.elements import atomic_masses
from mdapy.errors import CalculationFailed, InvalidParameter, MissingProperty
from mdapy.neighbor import NeighborList
from mdapy.simulation import _controls, _device

_KB = units.BOLTZMANN
_MVV2E = units.AMU_A2_PER_PS2_TO_EV
_FTM2V = 1.0 / _MVV2E
_GPA_TO_EV_PER_A3 = units.GPA_TO_EV_PER_A3
_EV_PER_A3_TO_GPA = units.EV_PER_A3_TO_GPA
# the thermostats of a chain, on the atoms and on the barostat
_CHAIN = 3
# the skin of the neighbor list kept between steps, in Å
_SKIN = 1.0
# the columns the dynamics changes, or leaves out of date
_MOVING = ("positions", "velocities", "image", "forces", "atomic_energies",
           "atomic_virials")  # fmt: skip
# the components of a symmetric tensor in the order of LAMMPS: xx, yy, zz,
# yz, xz, xy
_VOIGT = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


class _Barostat(NamedTuple):
    """What the barostat controls, as the fix of LAMMPS would: the
    components (xx, yy, zz, yz, xz, xy) it changes, how the diagonal ones
    are coupled ("xyz", "xy", "xz" or "none"), its style ("iso", "aniso" or
    "tri"), whether the target stress has a deviatoric part, and whether
    the tilts yz, xz and xy scale with the lengths of the box."""

    flags: np.ndarray
    couple: str
    style: str
    deviatoric: bool
    scale: np.ndarray


def _barostat(
    name: str,
    pbc: np.ndarray,
    swapped: bool,
    h: np.ndarray,
    deformed: np.ndarray | None = None,
) -> _Barostat:
    """The barostat of a name, for a box with these periodic directions in
    the frame of LAMMPS, as the fix of the LAMMPS dynamics sets it; with
    deformed, (3,) bool, the lengths a deformation changes, on the others
    alone, coupled for "iso", as fix npt y ... z ... couple yz; "x", "y"
    or "z" on that length alone."""
    flags = np.zeros(6, dtype=bool)
    if name in ("x", "y", "z"):  # along one axis, as fix npt z ...
        deformed = np.array([axis != name for axis in "xyz"])
    if deformed is not None:
        flags[:3] = ~deformed
        left = "".join("xyz"[k] for k in range(3) if flags[k])
        couple = left if name == "iso" and len(left) > 1 else "none"
        scale = np.array(
            [
                bool(flags[2] and pbc[2] and h[2, 1] != 0.0),
                bool(flags[2] and pbc[2] and h[2, 0] != 0.0),
                bool(flags[1] and pbc[1] and h[1, 0] != 0.0),
            ]
        )
        return _Barostat(flags, couple, "aniso", True, scale)
    # tilts that scale with the lengths: yz and xz with z, xy with y
    scale = np.array(
        [
            pbc[2] and h[2, 1] != 0.0,
            pbc[2] and h[2, 0] != 0.0,
            pbc[1] and h[1, 0] != 0.0,
        ]
    )
    original = pbc[[0, 2, 1]] if swapped else pbc
    sheet = bool(original[:2].all() and not original[2])
    # the second vector of the box is the z axis of LAMMPS when swapped
    other = 2 if swapped else 1
    if name == "xy" or (sheet and name == "iso"):
        flags[[0, other]] = True
        return _Barostat(flags, "xz" if swapped else "xy", "aniso", True, scale)
    if sheet:
        flags[[0, other]] = True
        if name == "tri":
            flags[4 if swapped else 5] = True
            scale[1 if swapped else 2] = False
        return _Barostat(
            flags, "none", "tri" if name == "tri" else "aniso", True, scale
        )
    if name == "tri":
        flags[:] = True
        return _Barostat(flags, "none", "tri", False, np.zeros(3, dtype=bool))
    flags[:3] = True
    return _Barostat(flags, "xyz" if name == "iso" else "none", name, False, scale)


# the time step of an atom that neither moves nor feels a force, as LAMMPS's
_BIG = 1.0e20


def _step_limit(
    v: np.ndarray, f: np.ndarray, masses: np.ndarray, distance: float
) -> float:
    """The largest time step in which no atom moves more than distance by
    its velocity and its force, as LAMMPS's fix dt/reset finds it: the
    smaller of distance / |v| and sqrt(2 distance m / |f|), scaled down if
    the move of v dt + f dt^2 / 2m is still longer."""
    inverse = (1.0 / masses)[:, None]
    vsq = np.einsum("ij,ij->i", v, v)
    fsq = np.einsum("ij,ij->i", f, f)
    with np.errstate(divide="ignore"):
        dtv = np.where(vsq > 0.0, distance / np.sqrt(vsq), _BIG)
        dtf = np.where(
            fsq > 0.0,
            np.sqrt(2.0 * distance / (_FTM2V * np.sqrt(fsq) * inverse[:, 0])),
            _BIG,
        )
    dt = np.minimum(dtv, dtf)[:, None]
    move = dt * v + 0.5 * dt * dt * inverse * f * _FTM2V
    length = np.sqrt(np.einsum("ij,ij->i", move, move))
    dt = dt[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        dt = np.where(length > distance, dt * distance / length, dt)
    return float(dt.min()) if len(dt) else _BIG


def _upper(h: np.ndarray) -> np.ndarray:
    """The box (rows a, b, c, lower triangular) as the upper triangular
    matrix of LAMMPS, columns a, b, c."""
    return h.T


def _voigt(tensor: np.ndarray) -> np.ndarray:
    return np.array([tensor[a, b] for a, b in _VOIGT])


def _symmetric(voigt: np.ndarray) -> np.ndarray:
    out = np.empty((3, 3))
    for k, (a, b) in enumerate(_VOIGT):
        out[a, b] = out[b, a] = voigt[k]
    return out


class _Msst:
    """The variables of the multiscale shock technique of a run, as LAMMPS's
    fix msst keeps them: the rate of the volume, the sum of the squares of
    the velocities, the position behind the front, and the reference."""

    def __init__(self, axis: int, speed: float, settings: dict, mass: float,
                 reference: tuple[float, float, float]):  # fmt: skip
        self.axis, self.speed = axis, speed
        self.q, self.mu, self.tscale = (settings[k] for k in ("q", "mu", "tscale"))
        self.mass = mass
        self.e0, self.v0, self.p0 = reference
        self.omega = 0.0
        self.vsum = 0.0
        self.position = 0.0

    def cell(self, p: float, volume: float, half: float) -> None:
        """The rate of the volume half a step on, at a fixed volume and atoms."""
        mass, v0 = self.mass, self.v0
        driving = _MVV2E * self.speed * self.speed * mass * (v0 - volume) / (v0 * v0)
        a = mass * (p - self.p0 - driving) / (self.q * _MVV2E)
        b = mass * self.mu / (self.q * volume)
        if volume > v0 and a > 0.0:  # no blow-up of the volume
            a = -a
        w = self.omega
        if b * half > 1.0e-06:
            self.omega = (w + a * (math.exp(b * half) - 1.0) / b) * math.exp(-b * half)
        else:
            self.omega = (
                w + (a - b * w) * half + 0.5 * (b * b * w - a * b) * half * half
            )

    def damping(self, volume: float, vsum: float) -> tuple[float, float]:
        """The terms of the velocities: the viscous one, over the mass of an
        atom, and that of the rate of the volume along the axis."""
        base = self.mu * self.omega * self.omega / (vsum * volume) if self.mu else 0.0
        return base, 2.0 * self.omega / volume

    def records(self, e: float, p: float, volume: float, dof: int) -> dict[str, float]:
        mass, v0, p0, us = self.mass, self.v0, self.p0, self.speed
        strain = 1.0 - volume / v0
        return {
            "hugoniot": (0.5 * (p + p0) * (v0 - volume) + self.e0 - e) / (dof * _KB),
            "rayleigh": (p - p0 - mass * us * us * _MVV2E * strain / v0)
            * _EV_PER_A3_TO_GPA,
            "particle_velocity": us * strain,
            "lagrangian_position": self.position,
            "msst_energy": self.q * self.omega**2 / (2.0 * mass) * _MVV2E
            - 0.5 * mass * us * us * strain * strain * _MVV2E
            - p0 * (v0 - volume),
        }


def _msst_velocities(
    v: np.ndarray, f: np.ndarray, masses: np.ndarray, half: float, terms: tuple
) -> np.ndarray:
    """The velocities of fix msst half a step on: exp(D t) (C + D v - C exp(-D t)) / D,
    or its expansion for a small D t, with C = ftm2v f / m and D the terms."""
    base, axial, axis, ftm2v = terms
    m = masses[:, None]
    c = f * ftm2v / m
    d = np.repeat(base / m, 3, axis=1)
    d[:, axis] -= axial
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        e = np.exp(d * half)
        exact = e * (c + d * v - c / e) / d
    small = v + (c + d * v) * half + 0.5 * (d * d * v + c * d) * half * half
    return np.where(np.abs(half * d) > 1.0e-06, exact, small)


def _minimum_image(d: np.ndarray, h: np.ndarray, pbc: Any) -> np.ndarray:
    """Displacements in their minimum image as LAMMPS takes it in a box h of
    its form: along z, then y, then x, the tilts carried."""
    d = d.copy()
    if pbc[2]:
        d -= np.rint(d[:, 2] / h[2, 2])[:, None] * h[2]
    if pbc[1]:
        d -= np.rint(d[:, 1] / h[1, 1])[:, None] * h[1]
    if pbc[0]:
        d[:, 0] -= np.rint(d[:, 0] / h[0, 0]) * h[0, 0]
    return d


def _indented(
    x: np.ndarray, h: np.ndarray, pbc: Any, spec: tuple, at: Any
) -> tuple[np.ndarray, float, np.ndarray]:
    """The forces of a virtual indenter on the atoms, its energy and the force
    on it, as fix indent: spec (kind, axis, sign, k, radius), kind 0 a sphere,
    1 a cylinder, 2 a plane, at its center or the position of the plane."""
    kind, axis, sign, k, radius = spec
    forces = np.zeros_like(x)
    if kind == 2:
        dr = sign * (at - x[:, axis])
        past = dr < 0.0
        push = -sign * k * dr[past] ** 2
        forces[past, axis] = push
        on = np.zeros(3)
        on[axis] = -push.sum()
        return forces, float(-(k / 3.0 * dr[past] ** 3).sum()), on
    d = x - at
    if kind == 1:
        d[:, axis] = 0.0
    d = _minimum_image(d, h, pbc)
    r = np.sqrt(np.einsum("ij,ij->i", d, d))
    dr = r - radius if sign > 0 else radius - r
    past = dr < 0.0
    push = d[past] * (sign * k * dr[past] ** 2 / r[past])[:, None]
    forces[past] = push
    return forces, float(-(k / 3.0 * dr[past] ** 3).sum()), -push.sum(axis=0)


class _HostAtoms:
    """The atoms of MDAPY's dynamics on the processor: NumPy arrays of the
    positions, velocities and forces in the frame of LAMMPS, and the steps
    of the integrator on each atom."""

    def __init__(
        self,
        calculator: Calculator,
        x: np.ndarray,
        v: np.ndarray,
        masses: np.ndarray,
        restrict: Any,
    ):
        self.__calculator = calculator
        self.__x, self.__v, self.__masses = x, v, masses
        self.__restrict = restrict
        self.__forces: np.ndarray | None = None
        self.__wide: NeighborList | None = None
        self.__rng: np.random.Generator | None = None
        # a Switch: the other interaction, its neighbor list, and λ
        self.__field: np.ndarray | None = None  # of HNEMD, in the frame
        self.__virials: np.ndarray | None = None  # of each atom, for the field
        self.__target: Any = False  # False: no switch; None: the ideal gas
        self.__target_wide: NeighborList | None = None
        self.weight = 0.0
        self.parts = (math.nan, math.nan)  # the energies of the two
        self.__properties = tuple(
            p for p in ("energy", "forces", "virial") if p in calculator.implemented
        )
        # what the controls of a run set, in the frame of LAMMPS: forces added
        # to the atoms, the velocities of the components that do not move
        # freely (zero where they do), and groups whose forces are summed
        self.added: np.ndarray | None = None
        self.prescribed: np.ndarray | None = None
        self.watched: tuple = ()
        self.sums: list = []
        self.__baths: list = []

    @property
    def positions(self) -> np.ndarray:
        return self.__x

    @property
    def velocities(self) -> np.ndarray:
        return self.__v

    @velocities.setter
    def velocities(self, v: np.ndarray) -> None:
        self.__v = v

    @property
    def forces(self) -> np.ndarray:
        """Those of the last computation, as the steps use them."""
        return self.__forces

    def twice_kinetic(self) -> float:
        return float(np.einsum("i,ij,ij->", self.__masses, self.__v, self.__v)) * _MVV2E

    def mvv(self) -> np.ndarray:
        return np.einsum("i,ij,ik->jk", self.__masses, self.__v, self.__v) * _MVV2E

    def compute(
        self, table: pl.DataFrame, box: Box, energy: bool, virial: bool
    ) -> tuple[float, np.ndarray]:
        """The energy and the virial, the forces kept; on the processor
        always all three."""
        # a copy of the positions: the System may hold the array itself, and
        # the steps change it in place, which would move the positions the
        # kept neighbor list compares with
        current = System(table=table.with_columns(
            pl.Series("positions", self.__x.copy(), dtype=pl.Array(pl.Float64, 3))
        ), box=box)  # fmt: skip
        cutoff = float(self.__calculator.cutoff)
        neighbors = None
        if 0.0 < cutoff < np.inf:
            wide = self.__wide
            neighbors = wide._follow(current, cutoff) if wide is not None else None
            if neighbors is None:
                self.__wide = NeighborList.build(current, cutoff + _SKIN)
                neighbors = self.__wide._follow(current, cutoff)
        wanted = self.__properties
        if self.__field is not None:
            wanted = wanted + ("atomic_virials", HEAT_FLUX_TERMS)
        result = self.__calculator.compute(current, wanted, neighbors=neighbors)
        forces = np.asarray(result.forces, dtype=float)
        energy = float(result.energy)
        if self.__field is not None:
            # each atom pushed by the field through its virial, W^T F, the
            # pushes less their mean, as GPUMD's HNEMD
            self.__virials = np.asarray(result.atomic_virials, dtype=float)
            push = np.einsum("a,iab->ib", self.__field, self.__virials)
            forces = forces + (push - push.mean(axis=0))
        work = (
            np.asarray(result.virial, dtype=float)
            if "virial" in self.__properties
            else np.full((3, 3), np.nan)
        )
        if self.__target is not False:
            energy, forces, work = self.__switched(current, energy, forces, work)
        if not (np.isfinite(energy) and np.isfinite(forces).all()):
            raise CalculationFailed(
                "the energy or the forces are not finite: atoms may overlap"
            )
        # the forces of the potential on the watched groups
        self.sums = [forces[mask].sum(axis=0) for mask in self.watched]
        self.indented = []
        for spec, at in zip(self.indenters, self.indenter_at):
            push, work, on = _indented(self.__x, box.matrix, box.pbc, spec, at)
            forces = forces + push
            self.indented.append((work, on))
        if self.added is not None:
            forces = forces + self.added
        self.__forces = self.__restrict(forces)
        return energy, work

    def __switched(
        self, current: System, energy: float, forces: np.ndarray, work: np.ndarray
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """(1 - λ) times the potential plus λ times the target of a Switch:
        the energy, the forces and the virial, the target giving none where
        it computes none; the energies of the two kept."""
        target, weight = self.__target, self.weight
        if target is None:  # the ideal gas
            other, pulls, push = 0.0, np.zeros_like(forces), np.zeros((3, 3))
        else:
            cutoff = float(target.cutoff)
            neighbors = None
            if 0.0 < cutoff < np.inf:
                wide = self.__target_wide
                neighbors = wide._follow(current, cutoff) if wide is not None else None
                if neighbors is None:
                    self.__target_wide = NeighborList.build(current, cutoff + _SKIN)
                    neighbors = self.__target_wide._follow(current, cutoff)
            wanted = ("energy", "forces") + (
                ("virial",) if "virial" in target.implemented else ()
            )
            result = target.compute(current, wanted, neighbors=neighbors)
            other = float(result.energy)
            pulls = np.asarray(result.forces, dtype=float)
            push = (
                np.asarray(result.virial, dtype=float)
                if result.virial is not None
                else np.zeros((3, 3))
            )
        self.parts = (energy, other)
        mixed = (1.0 - weight) * energy + weight * other
        return (
            mixed,
            (1.0 - weight) * forces + weight * pulls,
            (1.0 - weight) * work + weight * push,
        )

    def kick(self, c: float) -> None:
        self.__v += c * self.__forces / self.__masses[:, None]

    def drift(self, dt: float) -> None:
        self.__x += dt * self.__v

    def kick_drift(self, c: float, dt: float) -> None:
        self.kick(c)
        self.drift(dt)

    indenters: tuple = ()  # (kind, axis, sign, k, radius) of each indenter
    indenter_at: tuple = ()  # their centers or planes at the step
    indented: list = []  # (energy, force on it) of each, of the last computation

    def speed_squares(self) -> float:
        """The sum of the squares of the velocities."""
        return float(np.einsum("ij,ij->", self.__v, self.__v))

    def msst_kick(
        self,
        half: float,
        base: float,
        axial: float,
        axis: int,
        trial: bool,
        ftm2v: float,
    ) -> float:
        """The velocities of fix msst half a step on, kept unless trial; the
        sum of their squares."""
        v = _msst_velocities(
            self.__v, self.__forces, self.__masses, half, (base, axial, axis, ftm2v)
        )
        if not trial:
            self.__v = v
        return float(np.einsum("ij,ij->", v, v))

    def reflect(self, normal: np.ndarray, offset: float, speed: float) -> None:
        """A flat wall: each atom with x . normal < offset put back by as
        much, its velocity along the normal made 2 speed - v . normal."""
        depth = self.__x @ normal - offset
        past = depth < 0.0
        if past.any():
            self.__x[past] -= 2.0 * depth[past, None] * normal
            change = 2.0 * (speed - self.__v[past] @ normal)
            self.__v[past] += change[:, None] * normal

    def scale(self, factor: Any) -> None:
        self.__v *= factor

    def shear(self, half: float, w: np.ndarray) -> None:
        v = self.__v
        v[:, 0] -= half * (v[:, 1] * w[5] + v[:, 2] * w[4])
        v[:, 1] -= half * v[:, 2] * w[3]

    def restrict(self) -> None:
        """The velocities of the components that do not move freely set:
        zero, or those of their control."""
        self.__v = self.__restrict(self.__v)
        if self.prescribed is not None:
            self.__v = self.__v + self.prescribed

    def remap(
        self, old_lo: np.ndarray, old: np.ndarray, lo: np.ndarray, new: np.ndarray
    ) -> None:
        # fractional coordinates of the atoms in the old box, kept in the new
        fractional = np.einsum("ij,jk->ik", self.__x - old_lo, np.linalg.inv(old))
        self.__x = lo + np.einsum("ij,jk->ik", fractional, new)

    def seed(self, seed: int | None) -> None:
        self.__rng = np.random.default_rng(seed)

    def set_controls(
        self,
        free: np.ndarray | None,
        prescribed: np.ndarray | None,
        added: np.ndarray | None,
        watched: tuple,
        baths: tuple,
    ) -> None:
        """The controls of a run: the free components (in the axes of the
        system; the restriction given at the start follows them), the
        velocities of the others and the forces added (frame of LAMMPS), the
        groups whose forces are summed, and (mask, seed) of each bath."""
        self.prescribed, self.added, self.watched = prescribed, added, watched
        self.__baths = [(mask, np.random.default_rng(seed)) for mask, seed in baths]

    def clear_controls(self) -> None:
        self.prescribed = self.added = None
        self.watched = ()
        self.__baths = []

    def twice_kinetic_free(self) -> float:
        """Of the components that move freely."""
        v = self.__restrict(self.__v)
        return float(np.einsum("i,ij,ij->", self.__masses, v, v)) * _MVV2E

    def bath(self, k: int, c1: float, c2: float, kt: float) -> float:
        """v = c1 v + c2 sqrt(kt / m) g on the atoms of bath k, g Gaussian of
        its random numbers in the directions they move freely; the kinetic
        energy it added, in eV."""
        mask, rng = self.__baths[k]
        masses = self.__masses[mask]
        before = self.__v[mask]
        noise = np.zeros_like(self.__v)
        noise[mask] = rng.normal(size=before.shape)
        noise = self.__restrict(noise)[mask]
        after = c1 * before + c2 * np.sqrt(kt / masses)[:, None] * noise
        self.__v[mask] = after
        return 0.5 * float(np.einsum("i,ij,ij->", masses, after, after)
                           - np.einsum("i,ij,ij->", masses, before, before)) * _MVV2E  # fmt: skip

    def set_field(self, field: np.ndarray | None) -> None:
        """The driving field of HNEMD in the frame, or None for none."""
        self.__field = field
        self.__virials = None

    def heat_flux(self) -> np.ndarray:
        """(3,) the potential part of the heat flux times the volume in the
        frame, sum_i W_i v_i, of the virials of the last computation and the
        velocities now."""
        return np.einsum("iab,ib->a", self.__virials, self.__v)

    def set_switch(self, target: Any) -> None:
        """The other interaction of a Switch, None for none, or False for no
        switch."""
        self.__target = target
        self.__target_wide = None
        self.weight = 0.0

    def langevin(self, c1: float, c2: float, kt: float, keep: bool = False) -> None:
        """v = c1 v + c2 sqrt(kt / m) g, g Gaussian; with keep, the noise of
        each atom less an equal share of its total momentum, as the random
        forces of LAMMPS's fix langevin zero yes."""
        sigma = np.sqrt(kt / self.__masses)[:, None]
        noise = c2 * sigma * self.__restrict(self.__rng.normal(size=self.__v.shape))
        if keep:
            noise -= (self.__masses @ noise) / len(noise) / self.__masses[:, None]
        self.__v = c1 * self.__v + noise
        if self.prescribed is not None:
            self.restrict()


class _Native:
    """The state of the dynamics in MDAPY and its steps."""

    def __init__(self, system: System, calculator: Calculator, timestep: float):
        if not isinstance(calculator, Calculator):
            raise InvalidParameter("potential must be an MDAPY calculator")
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
        self.__n = n
        self.__calculator = calculator
        self.__masses = masses
        self.__table = table.drop([c for c in _MOVING if c in table.columns])
        self.__info = {
            k: v
            for k, v in system.info.items()
            if k not in ("energy", "virial", "stress")
        }
        box = system.box
        lammps_box = box.to_lammps()
        self.__swapped = bool(np.linalg.det(box.matrix) < 0)
        self.__rotation = lammps_box.rotation
        self.__pbc = box.pbc
        lengths = lammps_box.hi - lammps_box.lo
        xy, xz, yz = (float(t) for t in lammps_box.tilt)
        self.__h = np.array(
            [[lengths[0], 0.0, 0.0], [xy, lengths[1], 0.0], [xz, yz, lengths[2]]]
        )
        self.__lo = np.array(lammps_box.lo, dtype=float)
        self.__frame_pbc = box.pbc[[0, 2, 1]] if self.__swapped else box.pbc
        x = self.__to_frame(system["positions"])
        v = (
            self.__to_frame(system["velocities"])
            if "velocities" in table.columns
            else np.zeros((n, 3))
        )
        # the directions in which each atom may move, as VASP's selective dynamics
        movable = (
            table["movable"].to_numpy().astype(bool)
            if "movable" in table.columns
            else None
        )
        self.__movable = None if movable is None or movable.all() else movable
        # the components that move freely in the run, in the axes of the
        # system: those of the movable column, less those its controls set
        self.__free = self.__movable
        v = self.__restrict(v)
        self.__dof = 3 * n - 3 if self.__movable is None else int(self.__movable.sum())
        self.__moves: tuple = ()  # the names of the moved groups recorded
        self.__controlled = False  # whether the run has controls
        self.__deform: Any = None  # the Deform of the run
        self.__adaptive: Any = None  # the AdaptiveTimestep of the run
        self.__switch: Any = None  # the Switch of the run
        self.__hugoniostat = False  # whether the run is nphug
        self.__msst: _Msst | None = None  # the shock of a run of msst
        self.__indenters: tuple = ()  # the Indenter controls of the run
        self.__walls: tuple = ()  # (normal, offset, speed) of each wall
        self.__heating = False  # whether a heat field drives the run
        self.__keep = False  # whether the Langevin noise keeps the momentum
        self.__baths: tuple = ()  # (control, seed, damping) of each bath
        self.__prescribed = False  # whether controls set velocities
        self.__heat: list = []  # the heat each gave since the run began, eV
        self.timestep = float(timestep)
        self.__step = 0
        self.__time = 0.0
        self.__closed = False
        self.__properties = tuple(
            p for p in ("energy", "forces", "virial") if p in calculator.implemented
        )
        if not {"energy", "forces"} <= set(self.__properties):
            raise InvalidParameter(
                f"{type(calculator).__name__} does not compute energies and forces"
            )
        if getattr(calculator, "device", "cpu") == "cuda":
            self.__atoms: Any = _device.DeviceAtoms(
                calculator, system, x, v, masses, _SKIN, self.__movable, self.__rotation
            )
        else:
            self.__atoms = _HostAtoms(calculator, x, v, masses, self.__restrict)
        self.__computed = False
        self.__pstat = False
        self.__energy = math.nan
        self.__virial = np.full((3, 3), np.nan)
        self.__run: dict[str, Any] = {}

    def describe(self) -> str:
        """What runs the dynamics, for the log."""
        calculator = self.__calculator
        if getattr(calculator, "device", "cpu") == "cuda":
            name = _gpu.cuda().device_name()
            return f"MDAPY on the GPU, {name}, {calculator.precision} precision"
        return f"MDAPY on the processor, {get_num_threads()} threads"

    writes = True  # one process

    # -- frames ------------------------------------------------------------

    def __to_frame(self, vectors: Any) -> np.ndarray:
        return np.einsum("ij,jk->ik", np.asarray(vectors, dtype=float), self.__rotation)

    def __from_frame(self, vectors: np.ndarray) -> np.ndarray:
        return np.einsum("ij,kj->ik", vectors, self.__rotation)

    def __restrict(self, vectors: np.ndarray) -> np.ndarray:
        """Vectors of the atoms with the components along which they may not
        move removed, in the axes of the system."""
        if self.__free is None:
            return vectors
        return self.__to_frame(self.__from_frame(vectors) * self.__free)

    def __box(self) -> Box:
        return Box(self.__h, origin=self.__lo, pbc=self.__frame_pbc)

    # -- reading the state -------------------------------------------------

    @property
    def closed(self) -> bool:
        return self.__closed

    def close(self) -> None:
        self.__closed = True

    @property
    def step(self) -> int:
        return self.__step

    @property
    def system(self) -> System:
        r = self.__rotation
        matrix = np.einsum("ij,kj->ik", self.__h, r)
        if self.__swapped:
            matrix = matrix[[0, 2, 1]]
        box = Box(matrix, origin=self.__lo @ r.T, pbc=self.__pbc)
        wrapped, image = box.wrap(self.__from_frame(self.__atoms.positions))
        table = self.__table.with_columns(
            pl.Series("positions", wrapped, dtype=pl.Array(pl.Float64, 3)),
            pl.Series(
                "velocities",
                self.__from_frame(self.__atoms.velocities),
                dtype=pl.Array(pl.Float64, 3),
            ),
            pl.Series("image", image.astype(np.int32), dtype=pl.Array(pl.Int32, 3)),
        )
        info = self.__info | {"timestep": self.__step, "time": self.__time}
        return System(table=table, box=box, info=info)

    def temperature(self) -> float:
        """Of the components that move freely."""
        if self.__dof < 1:
            return math.nan
        # without velocities set by controls the components held are still
        twice = (
            self.__atoms.twice_kinetic_free()
            if self.__prescribed
            else self.__twice_kinetic()
        )
        return twice / (self.__dof * _KB)

    def __twice_kinetic(self) -> float:
        return self.__atoms.twice_kinetic()

    def __mvv(self) -> np.ndarray:
        return self.__atoms.mvv()

    def __volume(self) -> float:
        return float(self.__h[0, 0] * self.__h[1, 1] * self.__h[2, 2])

    def __pressure_tensor(self) -> np.ndarray:
        """The pressure tensor in eV/Å³, in the frame of LAMMPS."""
        return (self.__mvv() + self.__virial) / self.__volume()

    def record(self) -> dict[str, Any]:
        """The thermodynamic state now, as the LAMMPS dynamics records it."""
        if not self.__computed:
            self.__compute()
        kinetic = 0.5 * self.__twice_kinetic()
        tensor = self.__pressure_tensor()
        r = self.__rotation
        stress = -(r @ tensor @ r.T) * _EV_PER_A3_TO_GPA
        return {
            "step": self.__step,
            "time": self.__time,
            "temperature": self.temperature(),
            "potential_energy": self.__energy,
            "kinetic_energy": kinetic,
            "total_energy": self.__energy + kinetic,
            "pressure": float(np.trace(tensor)) / 3.0 * _EV_PER_A3_TO_GPA,
            "volume": self.__volume(),
            "stress": [float(stress[a, b]) for a, b in
                       ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))],
        } | {
            f"{name}_force": self.__from_frame(total[None, :])[0].tolist()
            for name, total in zip(self.__moves, getattr(self.__atoms, "sums", ()))
            if name is not None
        } | {
            f"{bath.name}_energy": heat
            for (bath, _, _), heat in zip(self.__baths, self.__heat)
            if bath.name is not None
        } | self.__switch_record() | self.__field_record() | self.__hugoniot_record() | self.__msst_record() | self.__indenter_record()  # fmt: skip

    def __field_record(self) -> dict[str, Any]:
        """The potential part of the heat flux times the volume of a run of
        HNEMD, in the axes of the system."""
        if not self.__heating:
            return {}
        flux = self.__from_frame(self.__atoms.heat_flux()[None, :])[0]
        return {"heat_flux": [float(x) for x in flux]}

    def __set_field(self, control: Any) -> None:
        """The driving field of HNEMD in the frame of the dynamics."""
        if control is None:
            return
        calculator = self.__calculator
        if not getattr(calculator, "_heat_flux", False):
            raise InvalidParameter(
                "the heat flux needs the centroid stress of each atom, which "
                f"{type(calculator).__name__} does not give; EAM, the pair "
                "potentials, NEP, Stillinger-Weber and Tersoff give it"
            )
        self.__atoms.set_field(self.__to_frame(control.field[None, :])[0])
        self.__heating = True

    def __set_indenters(self, indenters: tuple) -> None:
        """The virtual indenters of a run, in a box of the form of LAMMPS: a
        plane across an axis along which the box is not periodic."""
        self.__indenters = ()
        self.__atoms.indenters = ()
        if not indenters:
            return
        if not (np.array_equal(self.__rotation, np.eye(3)) and not self.__swapped):
            raise InvalidParameter(
                "an indenter needs a box whose first vector is along x and second "
                "in the xy plane"
            )
        specs = []
        for indenter in indenters:
            axis = "xyz".index(indenter.axis) if indenter.axis else 0
            if indenter.shape == "plane":
                if self.__frame_pbc[axis]:
                    raise InvalidParameter(
                        f"a plane indenter across {indenter.axis} needs a box not "
                        f"periodic along {indenter.axis}"
                    )
                sign = 1.0 if indenter.side == "high" else -1.0
            else:
                sign = 1.0 if indenter.side == "out" else -1.0
            kind = ("sphere", "cylinder", "plane").index(indenter.shape)
            specs.append((kind, axis, sign, indenter.stiffness, indenter.radius or 0.0))
        self.__indenters = indenters
        self.__atoms.indenters = tuple(specs)

    def __indenter_record(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for indenter, (energy, on) in zip(self.__indenters, self.__atoms.indented):
            if indenter.name is not None:
                out[f"{indenter.name}_force"] = [float(v) for v in on]
                out[f"{indenter.name}_energy"] = float(energy)
        return out

    def __set_walls(self, walls: tuple) -> None:
        """The plane of each wall in the frame of the dynamics at the start of
        the run, its normal into the atoms, and its speed along the normal."""
        self.__walls = ()
        if not walls:
            return
        r = self.__rotation
        matrix, origin = self.__h @ r.T, self.__lo @ r.T  # in the axes of the system
        self.__walls = tuple(
            (
                self.__to_frame(wall._normal[None, :])[0],
                wall._sign * wall._start(matrix, origin, self.__frame_pbc),
                wall._sign * wall.velocity,
            )
            for wall in walls
        )

    def __switch_record(self) -> dict[str, float]:
        """λ of a named Switch and the energy of its target minus that of the
        potential, both unscaled."""
        switch = self.__switch
        if switch is None or switch.name is None:
            return {}
        potential, target = self.__atoms.parts
        return {
            f"{switch.name}_lambda": float(self.__atoms.weight),
            f"{switch.name}_difference": float(target - potential),
        }

    def __set_switch(self, switch: Any, barostat: bool) -> None:
        """The other interaction of a Switch in the frame of the dynamics,
        and λ at the start of the run."""
        self.__switch = None
        if switch is None:
            return
        target = switch.target
        if barostat and target is not None and "virial" not in target.implemented:
            raise InvalidParameter(
                f"a barostat needs the virial, which {type(target).__name__} does "
                "not compute"
            )
        if isinstance(target, Springs) and not np.array_equal(
            self.__rotation, np.eye(3)
        ):
            # the sites in the frame of the dynamics, as the atoms
            target = Springs(self.__to_frame(target.sites), target.constants)
        self.__atoms.set_switch(target)
        self.__atoms.weight = switch.weight(0.0)
        self.__switch = switch

    # -- velocities -------------------------------------------------------------

    def velocities(self, temperature: float, seed: int) -> None:
        """Gaussian velocities of the temperature, with no total momentum
        when every atom may move."""
        rng = np.random.default_rng(int(seed))
        v = rng.normal(size=(self.__n, 3)) / np.sqrt(self.__masses)[:, None]
        if self.__movable is None:
            v -= (self.__masses @ v) / self.__masses.sum()
        else:
            v *= self.__movable
        self.__atoms.velocities = self.__to_frame(v)
        now = self.temperature()
        if temperature == 0.0 or not now > 0.0:
            self.__atoms.velocities = np.zeros((self.__n, 3))
        else:
            self.__atoms.scale(math.sqrt(temperature / now))

    # -- forces ---------------------------------------------------------------

    def __compute(self, full: bool = True) -> None:
        """The forces; the energy and the virial too when full or with a
        barostat (on the GPU; the processor always computes all three)."""
        pressure = full or self.__pstat or self.__msst is not None
        if self.__indenters:  # where they are at this step
            elapsed = (self.__step - self.__first) * self.__dt
            self.__atoms.indenter_at = tuple(c._at(elapsed) for c in self.__indenters)
        energy, virial = self.__atoms.compute(
            self.__table, self.__box(), full, pressure
        )
        if full:
            self.__energy = energy
        if "virial" in self.__properties and pressure:
            self.__virial = virial
        self.__computed = True

    # -- a run ---------------------------------------------------------------------

    def start(self, run: dict[str, Any]) -> None:
        """Sets up a run: the forces, and the variables and masses of the
        thermostat and barostat, as the fixes of LAMMPS do at its start."""
        ensemble = run["ensemble"]
        self.__run = run
        hugoniostat = ensemble == "nphug"
        thermostat = hugoniostat or (
            ensemble in ("nvt", "npt") and run["thermostat"] == "nose-hoover"
        )
        langevin = ensemble in ("nvt", "npt") and run["thermostat"] == "langevin"
        barostat = ensemble in ("npt", "nph", "nphug")
        if barostat and "virial" not in self.__properties:
            raise InvalidParameter(
                f"a barostat needs the virial, which "
                f"{type(self.__calculator).__name__} does not compute"
            )
        controls = run.get("controls", ())
        self.__set_deform(
            next((c for c in controls if isinstance(c, _controls.Deform)), None),
            barostat,
            run["barostat"],
        )
        self.__set_controls(controls, barostat)
        if (thermostat or langevin) and self.__dof < 1:
            raise InvalidParameter("a thermostat needs atoms that move")
        dt = self.timestep
        self.__dt = dt
        self.__first = self.__step
        # the time of a step, from the step where the time step last changed
        self.__atime, self.__atimestep = self.__time, self.__step
        self.__steps = run["steps"]
        self.__set_walls(tuple(c for c in controls if isinstance(c, _controls.Wall)))
        self.__set_indenters(
            tuple(c for c in controls if isinstance(c, _controls.Indenter))
        )
        self.__set_switch(
            next((c for c in controls if isinstance(c, _controls.Switch)), None),
            barostat,
        )
        self.__set_field(
            next((c for c in controls if isinstance(c, _controls._HeatField)), None)
        )
        self.__keep = bool(run.get("keep_momentum", False))
        if self.__keep:
            if self.__free is not None:
                raise InvalidParameter(
                    "keep_momentum with atoms held or moved: there is no momentum "
                    "to keep"
                )
        self.__compute()
        self.__adaptive = next(
            (c for c in controls if isinstance(c, _controls.AdaptiveTimestep)), None
        )
        if self.__adaptive is not None:
            self.__reset_dt()  # at the start of the run, as fix dt/reset
        self.__nh = thermostat
        self.__langevin = langevin
        self.__pstat = barostat
        t0, _ = run["temperatures"] if run["temperatures"] else (None, None)
        if hugoniostat:
            # the masses of the chains and of the barostat from the
            # temperature at the start, as fix nphug sets them
            t0 = self.temperature()
            if not t0 >= 1e-6:
                raise InvalidParameter(
                    "nphug needs atoms with a temperature, for the masses of its "
                    "thermostat and barostat"
                )
        # the target temperature of the chains, as the fixes of LAMMPS
        # update it: at the start of each step, after the barostat's chain
        self.__target = t0
        if thermostat:
            self.__t_freq = 1.0 / run["thermostat_damping"]
            t_target = t0
            self.__eta_dot = np.zeros(_CHAIN + 1)
            self.__eta_mass = np.full(_CHAIN, _KB * t_target / self.__t_freq**2)
            self.__eta_mass[0] *= self.__dof
            self.__eta_dotdot = np.zeros(_CHAIN)
            self.__eta_dotdot[1:] = -_KB * t_target / self.__eta_mass[1:]
        if langevin:
            self.__atoms.seed(run["seed"])
            self.__gamma = 1.0 / run["thermostat_damping"]
        if barostat:
            if run["barostat"] in ("x", "y", "z") and not (
                np.array_equal(self.__rotation, np.eye(3)) and not self.__swapped
            ):
                raise InvalidParameter(
                    "a barostat along one axis needs a box whose first vector is "
                    "along x and second in the xy plane"
                )
            if thermostat or langevin:
                # the temperature of the barostat: the target of the
                # thermostat, as fix npt; with Langevin dynamics, its value
                # at the start, as fix nph with ptemp
                self.__p_temperature = t0
            else:
                now = self.temperature()
                if not now >= 1e-6:
                    raise InvalidParameter(
                        "nph needs atoms with a temperature, for the mass of "
                        "the barostat"
                    )
                self.__p_temperature = now
            if not thermostat:
                self.__target = self.__p_temperature
            self.__setup_barostat(run)
        self.__hugoniostat = hugoniostat
        if hugoniostat:
            self.__set_hugoniot(run["hugoniot"])
        if ensemble == "msst":
            self.__set_msst(run)

    def __setup_barostat(self, run: dict[str, Any]) -> None:
        h = self.__h
        b = _barostat(
            run["barostat"],
            self.__frame_pbc,
            self.__swapped,
            h,
            None if self.__deform is None else self.__deform.rates[:3] != 0.0,
        )
        if run["ensemble"] == "nphug":  # fix nphug leaves out the deviatoric stress
            b = b._replace(deviatoric=False)
        self.__b = b
        p0, p1 = (p * _GPA_TO_EV_PER_A3 for p in run["pressures"])
        self.__p_start = np.where(b.flags, p0, 0.0)
        self.__p_stop = np.where(b.flags, p1, 0.0)
        # off-diagonal targets are zero
        self.__p_start[3:] = 0.0
        self.__p_stop[3:] = 0.0
        self.__p_target = np.zeros(6)
        self.__pdim = int(b.flags[:3].sum())
        p_freq = np.where(b.flags, 1.0 / run["barostat_damping"], 0.0)
        self.__p_freq_max = float(p_freq.max())
        kt = _KB * self.__target
        self.__omega_dot = np.zeros(6)
        with np.errstate(divide="ignore"):
            self.__omega_mass = np.where(b.flags, (self.__n + 1) * kt / p_freq**2, 0.0)
        self.__etap_dot = np.zeros(_CHAIN + 1)
        self.__etap_mass = np.full(_CHAIN, kt / self.__p_freq_max**2)
        self.__etap_dotdot = np.zeros(_CHAIN)
        self.__etap_dotdot[1:] = -kt / self.__etap_mass[1:]
        # the reference box of the deviatoric stress and the fixed point of
        # the scaling, the middle of the box
        self.__vol0 = self.__volume()
        self.__h0_inv = np.linalg.inv(_upper(h))
        self.__fixedpoint = self.__lo + 0.5 * np.diag(h)
        self.__mtk_term2 = 0.0
        self.__press_target(0)
        self.__p_current = self.__couple()

    def __set_hugoniot(self, reference: Any) -> None:
        """The reference state of the Hugoniot of nphug, given or that of the
        start: the total energy, the volume and the pressure, along the axis
        of a barostat along one axis or the mean of the diagonal; and the
        density, as fix nphug sets them; the target temperature of the first
        step's barostat chain, 0.01 K, as there."""
        flags = self.__b.flags[:3]
        self.__hug_axis = int(np.flatnonzero(flags)[0]) if flags.sum() == 1 else None
        if reference is None:
            p = self.__p_current
            e0 = self.__energy + 0.5 * self.__twice_kinetic()
            v0 = self.__volume()
            k = self.__hug_axis
            p0 = float(p[k]) if k is not None else float(p[:3].sum() / 3.0)
        else:
            e0, v0, p0 = reference
            p0 *= _GPA_TO_EV_PER_A3
        self.__hug_reference = (float(e0), float(v0), float(p0))
        self.__rho0 = _MVV2E * float(self.__masses.sum()) / float(v0)
        self.__target = 0.01

    def __hugoniot(self) -> tuple[float, float, float]:
        """The distance from the Hugoniot in K, the shock velocity and the
        particle velocity in Å/ps, of the state now, as fix nphug gives them."""
        e0, v0, p0 = self.__hug_reference
        tensor = self.__pressure_tensor()
        k = self.__hug_axis
        p = float(tensor[k, k]) if k is not None else float(np.trace(tensor)) / 3.0
        v = self.__volume()
        e = self.__energy + 0.5 * self.__twice_kinetic()
        distance = (0.5 * (p + p0) * (v0 - v) + e0 - e) / (self.__dof * _KB)
        strain = 1.0 - v / v0
        shock = (
            0.0
            if strain < 1e-10 or p < p0
            else math.sqrt((p - p0) / (self.__rho0 * strain))
        )
        return distance, shock, shock * strain

    def __set_msst(self, run: dict[str, Any]) -> None:
        """The shock of a run of msst, as fix msst sets it up: the reference
        state, given or that of the start; and the share tscale of the kinetic
        energy given to the rate of the volume, against the axis."""
        if "virial" not in self.__properties:
            raise InvalidParameter(
                f"msst needs the virial, which {type(self.__calculator).__name__} "
                "does not compute"
            )
        if not (np.array_equal(self.__rotation, np.eye(3)) and not self.__swapped):
            raise InvalidParameter(
                "msst needs a box whose first vector is along x and second in the "
                "xy plane"
            )
        if self.__free is not None:
            raise InvalidParameter("msst with atoms held is not supported")
        axis = "xyz".index(run["barostat"])
        settings = run["msst"]
        reference = run["hugoniot"]
        if reference is None:
            e0 = self.__energy + 0.5 * self.__twice_kinetic()
            v0 = self.__volume()
            p0 = float(self.__pressure_tensor()[axis, axis])
        else:
            e0, v0, p0 = reference
            p0 *= _GPA_TO_EV_PER_A3
        msst = _Msst(
            axis,
            float(run["shock_velocity"]),
            settings,
            float(self.__masses.sum()),
            (float(e0), float(v0), float(p0)),
        )
        twice = self.__twice_kinetic()
        if twice > 0.0 and msst.tscale > 0.0:
            msst.omega = -math.sqrt(msst.tscale * msst.mass / msst.q * twice / _MVV2E)
            self.__atoms.scale(math.sqrt(1.0 - msst.tscale))
        self.__msst = msst

    def __msst_record(self) -> dict[str, float]:
        msst = self.__msst
        if msst is None:
            return {}
        k = msst.axis
        energy = self.__energy + 0.5 * self.__twice_kinetic()
        p = float(self.__pressure_tensor()[k, k])
        return msst.records(energy, p, self.__volume(), self.__dof)

    def __msst_remap(self, dilation: float) -> None:
        """The box scaled along the axis about its middle, its tilts kept, the
        atoms carried with it and their velocities along it scaled, as the
        remap of fix msst."""
        k = self.__msst.axis
        old, old_lo = self.__h.copy(), self.__lo.copy()
        hi = old_lo[k] + old[k, k]
        middle = 0.5 * (old_lo[k] + hi)
        new, lo = old.copy(), old_lo.copy()
        lo[k] = (old_lo[k] - middle) * dilation + middle
        new[k, k] = (hi - middle) * dilation + middle - lo[k]
        self.__atoms.remap(old_lo, old, lo, new)
        factor = np.ones(3)
        factor[k] = dilation
        self.__atoms.scale(factor)
        self.__h, self.__lo = new, lo

    def __msst_step(self, dt: float, full: bool) -> None:
        """A step of fix msst: the rate of the volume, the velocities, the box
        and the atoms over the first half, the forces, then the velocities and
        the rate of the volume over the second."""
        msst, atoms = self.__msst, self.__atoms
        k, half = msst.axis, 0.5 * dt
        volume = self.__volume()
        msst.cell(float(self.__pressure_tensor()[k, k]), volume, half)
        # the sum of the squares of the velocities half a step on, by trial
        base, axial = msst.damping(volume, atoms.speed_squares())
        msst.vsum = atoms.msst_kick(half, base, axial, k, True, _FTM2V)
        base, axial = msst.damping(volume, msst.vsum)
        atoms.msst_kick(half, base, axial, k, False, _FTM2V)
        first = volume + msst.omega * half
        self.__msst_remap(first / volume)
        atoms.drift(dt)
        second = first + msst.omega * half
        self.__msst_remap(second / first)
        self.__compute(full)
        volume = self.__volume()
        base, axial = msst.damping(volume, msst.vsum)
        atoms.msst_kick(half, base, axial, k, False, _FTM2V)
        volume = self.__volume()
        msst.cell(float(self.__pressure_tensor()[k, k]), volume, half)
        msst.position -= msst.speed * volume / msst.v0 * dt

    def __hugoniot_record(self) -> dict[str, float]:
        if not self.__hugoniostat:
            return {}
        distance, shock, particle = self.__hugoniot()
        return {
            "hugoniot": distance,
            "shock_velocity": shock,
            "particle_velocity": particle,
        }

    def __set_deform(self, deform: Any, barostat: bool, style: str) -> None:
        """The box at the start of a deformation, as fix deform takes it."""
        self.__deform = deform
        if deform is None:
            return
        if not (np.array_equal(self.__rotation, np.eye(3)) and not self.__swapped):
            raise InvalidParameter(
                "a deformation needs a box whose first vector is along x and second "
                "in the xy plane"
            )
        if deform.kind == "true" and deform.rates[3:].any():
            raise InvalidParameter("tilts deform at engineering rates only")
        deform._periodic(self.__pbc)
        if barostat:
            if style not in ("iso", "aniso") or deform.rates[3:].any():
                raise InvalidParameter(
                    'a barostat with a deformation needs "iso" or "aniso" and no shear'
                )
            if (deform.rates[:3] != 0.0).all():
                raise InvalidParameter("the deformation leaves the barostat no length")
            sheet = bool(self.__pbc[:2].all() and not self.__pbc[2])
            if sheet:
                raise InvalidParameter(
                    "a deformed sheet with a barostat is not supported"
                )
        h = self.__h
        self.__deform_lo = self.__lo.copy()
        self.__deform_hi = self.__lo + np.diag(h)
        self.__deform_tilt = np.array([h[2, 1], h[2, 0], h[1, 0]])

    def __apply_deform(self, step: int) -> None:
        """The box of the deformation at the end of a step, and the atoms
        carried with it: the targets of fix deform, linear in the steps of
        the run for its rates of length and of tilt, exponential for true
        rates of length."""
        d = self.__deform
        done = step - self.__first
        delta = done / self.__steps if done else 0.0
        span = self.__steps * self.__dt  # the time of the whole run
        old, old_lo = self.__h.copy(), self.__lo.copy()
        lo, hi = old_lo.copy(), old_lo + np.diag(old)
        lo0, hi0 = self.__deform_lo, self.__deform_hi
        for i in range(3):
            rate = d.rates[i]
            if rate == 0.0:
                continue
            if d.kind == "true":
                shift = 0.5 * ((hi0[i] - lo0[i]) * math.exp(rate * done * self.__dt))
                lo[i] = 0.5 * (lo0[i] + hi0[i]) - shift
                hi[i] = 0.5 * (lo0[i] + hi0[i]) + shift
            else:
                shift = 0.5 * span * rate * (hi0[i] - lo0[i])
                lo[i] = lo0[i] + delta * ((lo0[i] - shift) - lo0[i])
                hi[i] = hi0[i] + delta * ((hi0[i] + shift) - hi0[i])
        # the tilts yz, xz and xy; the lengths across which they grow
        tilt = np.array([old[2, 1], old[2, 0], old[1, 0]])
        across = np.array([hi0[2] - lo0[2], hi0[2] - lo0[2], hi0[1] - lo0[1]])
        for k in range(3):
            rate = d.rates[3 + k]
            if rate != 0.0:
                start = self.__deform_tilt[k]
                tilt[k] = start + delta * ((start + span * rate * across[k]) - start)
        lengths = hi - lo
        new = np.array(
            [
                [lengths[0], 0.0, 0.0],
                [tilt[2], lengths[1], 0.0],
                [tilt[1], tilt[0], lengths[2]],
            ]
        )
        self.__atoms.remap(old_lo, old, lo, new)
        self.__h, self.__lo = new, lo

    def __set_controls(self, controls: tuple, barostat: bool) -> None:
        """The free components, prescribed velocities, added forces and
        watched groups of the controls of a run; the velocities of the
        components they set, set now."""
        n = self.__n
        if not controls:
            self.__reset_controls()
            return
        free = (
            np.ones((n, 3), dtype=bool)
            if self.__movable is None
            else self.__movable.copy()
        )
        prescribed = np.zeros((n, 3))
        added = np.zeros((n, 3))
        moves, watched, baths = [], [], []
        constrained = adds = False
        for control in controls:
            if isinstance(control, _controls.Hold):
                free[control.mask] &= ~control.axes
                constrained = constrained or bool(control.mask.any())
            elif isinstance(control, _controls.Langevin):
                damping = (
                    control.damping
                    if control.damping is not None
                    else 100 * self.timestep
                )
                seed = (
                    control.seed if control.seed is not None else secrets.randbits(63)
                )
                baths.append((control, seed, damping))
            elif isinstance(
                control,
                (
                    _controls.Deform,
                    _controls.AdaptiveTimestep,
                    _controls.Switch,
                    _controls.Wall,
                    _controls.Indenter,
                    _controls._HeatField,
                ),
            ):
                continue
            elif isinstance(control, _controls.Move):
                free[control.mask] = False
                prescribed[control.mask] = control.velocity
                constrained = constrained or bool(control.mask.any())
                moves.append(control.name)
                watched.append(control.mask)
            else:
                added[control.mask] += control.force
                adds = True
        if barostat and constrained:
            raise InvalidParameter(
                "a barostat with held or moved atoms is not supported"
            )
        self.__free = None if free.all() else free
        self.__dof = (
            3 * n - 3
            if self.__free is None and self.__movable is None
            else int(free.sum())
        )
        atoms = self.__atoms
        atoms.set_controls(
            self.__free,
            self.__to_frame(prescribed) if prescribed.any() else None,
            self.__to_frame(added) if adds else None,
            tuple(watched),
            tuple((control.mask, seed) for control, seed, _ in baths),
        )
        self.__prescribed = bool(prescribed.any())
        self.__moves = tuple(moves)
        self.__baths = tuple(baths)
        self.__heat = [0.0] * len(baths)
        atoms.restrict()
        self.__computed = False
        self.__controlled = True

    def __reset_controls(self) -> None:
        self.__controlled = False
        self.__free = self.__movable
        n = self.__n
        self.__dof = 3 * n - 3 if self.__movable is None else int(self.__movable.sum())
        self.__moves = ()
        self.__baths = ()
        self.__heat = []
        self.__prescribed = False
        self.__atoms.clear_controls()

    def finish(self) -> None:
        self.__run = {}
        self.__deform = None
        self.__adaptive = None
        self.__walls = ()
        self.__hugoniostat = False
        self.__msst = None
        self.__indenters = ()
        self.__atoms.indenters = ()
        if self.__switch is not None:
            self.__atoms.set_switch(False)
            self.__switch = None
        if self.__heating:
            self.__atoms.set_field(None)
            self.__heating = False
        if self.__controlled:
            self.__reset_controls()

    def advance(self, count: int) -> None:
        # the energy and the virial at the last step, for a record after it
        for k in range(count):
            self.__one_step(k == count - 1)

    # -- one step ------------------------------------------------------------------

    def __fraction(self, step: int) -> float:
        done = step - self.__first
        return done / self.__steps if done else 0.0

    def __t_target(self, step: int) -> float:
        t0, t1 = self.__run["temperatures"]
        return t0 + self.__fraction(step) * (t1 - t0)

    def __one_step(self, full: bool = True) -> None:
        self.__step += 1
        step = self.__step
        dt = self.__dt
        if self.__msst is not None:  # its own integrator, as fix msst
            self.__msst_step(dt, full)
            self.__time = self.__atime + (step - self.__atimestep) * dt
            return
        if self.__switch is not None:  # at the start of the step, as fix ti/spring
            self.__atoms.weight = self.__switch.weight(
                (step - self.__first) / self.__steps
            )
        if self.__langevin:
            self.__ornstein_uhlenbeck(step)
        if self.__baths:
            self.__bathe(step)
        if self.__pstat:
            self.__nhc_press()
        if self.__nh:
            now = self.temperature()
            if self.__hugoniostat:  # the temperature that brings it to the Hugoniot
                self.__target = now + self.__hugoniot()[0]
            else:
                self.__target = self.__t_target(step)
            self.__t_current = now
            self.__nhc_temp()
        if self.__pstat:
            self.__p_current = self.__couple()
            self.__press_target(step)
            self.__omega_dot_update()
            self.__v_press()
        if self.__pstat:
            self.__atoms.kick(0.5 * dt * _FTM2V)
            self.__remap()
            self.__atoms.drift(dt)
            self.__remap()
        else:
            self.__atoms.kick_drift(0.5 * dt * _FTM2V, dt)
        if self.__walls:  # after the atoms moved, as fix wall/reflect
            elapsed = (step - self.__first) * dt
            for normal, offset, speed in self.__walls:
                self.__atoms.reflect(normal, offset + speed * elapsed, speed)
        # nphug needs the energy of every step
        self.__compute(full or self.__hugoniostat)
        self.__atoms.kick(0.5 * dt * _FTM2V)
        if self.__pstat:
            self.__v_press()
            self.__p_current = self.__couple()
            self.__omega_dot_update()
        if self.__nh:
            self.__t_current = self.temperature()
            self.__nhc_temp()
        if self.__pstat:
            self.__nhc_press()
        if self.__langevin:
            self.__ornstein_uhlenbeck(step)
        if self.__baths:
            self.__bathe(step)
        if self.__deform is not None:
            self.__apply_deform(step)  # at the end of the step, as fix deform 1
        self.__time = self.__atime + (step - self.__atimestep) * dt
        if self.__adaptive is not None and step % self.__adaptive.every == 0:
            self.__reset_dt()

    def __reset_dt(self) -> None:
        """The time step of the fastest atom within the bounds, as fix
        dt/reset at the end of a step; the time goes on from this step."""
        a = self.__adaptive
        dt = _step_limit(
            self.__atoms.velocities, self.__atoms.forces, self.__masses, a.max_distance
        )
        if a.min_timestep is not None:
            dt = max(dt, a.min_timestep)
        dt = min(dt, self.timestep if a.max_timestep is None else a.max_timestep)
        if dt != self.__dt:
            self.__atime, self.__atimestep = self.__time, self.__step
            self.__dt = dt

    def __ornstein_uhlenbeck(self, step: int) -> None:
        """Friction and noise for half a step."""
        target = self.__t_target(step)
        c1 = math.exp(-0.5 * self.__gamma * self.__dt)
        c2, kt = math.sqrt(1.0 - c1 * c1), _KB * target * _FTM2V
        if self.__keep:
            self.__atoms.langevin(c1, c2, kt, True)
        else:
            self.__atoms.langevin(c1, c2, kt)

    def __bathe(self, step: int) -> None:
        """Friction and noise of each bath for half a step, the heat it
        gives counted."""
        fraction = self.__fraction(step)
        for k, (control, _, damping) in enumerate(self.__baths):
            t0, t1 = control.temperature
            kt = _KB * (t0 + fraction * (t1 - t0)) * _FTM2V
            c1 = math.exp(-0.5 * self.__dt / damping)
            self.__heat[k] += self.__atoms.bath(k, c1, math.sqrt(1.0 - c1 * c1), kt)

    def __nhc_temp(self) -> None:
        """Half a step of the chain of thermostats on the atoms."""
        dt = self.__dt
        dt4, dt8 = 0.25 * dt, 0.125 * dt
        kt = _KB * self.__target
        eta_dot, eta_dotdot, mass = self.__eta_dot, self.__eta_dotdot, self.__eta_mass
        if not self.__hugoniostat:  # fix nphug keeps the masses of its start
            mass[0] = self.__dof * kt / self.__t_freq**2
            mass[1:] = kt / self.__t_freq**2
        ke_target = self.__dof * kt
        kecurrent = self.__dof * _KB * self.__t_current
        eta_dotdot[0] = (kecurrent - ke_target) / mass[0]
        for i in range(_CHAIN - 1, 0, -1):
            expfac = math.exp(-dt8 * eta_dot[i + 1])
            eta_dot[i] = (eta_dot[i] * expfac + eta_dotdot[i] * dt4) * expfac
        expfac = math.exp(-dt8 * eta_dot[1])
        eta_dot[0] = (eta_dot[0] * expfac + eta_dotdot[0] * dt4) * expfac
        factor = math.exp(-0.5 * dt * eta_dot[0])
        self.__atoms.scale(factor)
        if self.__prescribed:
            self.__atoms.restrict()  # the moved atoms keep their velocities
        self.__t_current *= factor * factor
        eta_dotdot[0] = (self.__dof * _KB * self.__t_current - ke_target) / mass[0]
        eta_dot[0] = (eta_dot[0] * expfac + eta_dotdot[0] * dt4) * expfac
        for i in range(1, _CHAIN):
            expfac = math.exp(-dt8 * eta_dot[i + 1])
            eta_dotdot[i] = (mass[i - 1] * eta_dot[i - 1] ** 2 - kt) / mass[i]
            eta_dot[i] = (eta_dot[i] * expfac + eta_dotdot[i] * dt4) * expfac

    def __nhc_press(self) -> None:
        """Half a step of the chain of thermostats on the barostat."""
        dt = self.__dt
        dt4, dt8 = 0.25 * dt, 0.125 * dt
        b = self.__b
        kt = _KB * self.__target
        omega_dot, omega_mass = self.__omega_dot, self.__omega_mass
        etap_dot, etap_dotdot, mass = (
            self.__etap_dot,
            self.__etap_dotdot,
            self.__etap_mass,
        )
        kecurrent = float(np.sum(omega_mass[b.flags] * omega_dot[b.flags] ** 2))
        lkt = kt if b.style == "iso" else int(b.flags.sum()) * kt
        etap_dotdot[0] = (kecurrent - lkt) / mass[0]
        for i in range(_CHAIN - 1, 0, -1):
            expfac = math.exp(-dt8 * etap_dot[i + 1])
            etap_dot[i] = (etap_dot[i] * expfac + etap_dotdot[i] * dt4) * expfac
        expfac = math.exp(-dt8 * etap_dot[1])
        etap_dot[0] = (etap_dot[0] * expfac + etap_dotdot[0] * dt4) * expfac
        factor = math.exp(-0.5 * dt * etap_dot[0])
        omega_dot[b.flags] *= factor
        kecurrent = float(np.sum(omega_mass[b.flags] * omega_dot[b.flags] ** 2))
        etap_dotdot[0] = (kecurrent - lkt) / mass[0]
        etap_dot[0] = (etap_dot[0] * expfac + etap_dotdot[0] * dt4) * expfac
        for i in range(1, _CHAIN):
            expfac = math.exp(-dt8 * etap_dot[i + 1])
            etap_dotdot[i] = (mass[i - 1] * etap_dot[i - 1] ** 2 - kt) / mass[i]
            etap_dot[i] = (etap_dot[i] * expfac + etap_dotdot[i] * dt4) * expfac

    def __couple(self) -> np.ndarray:
        """The pressures the barostat compares with its targets, as the
        components xx, yy, zz, yz, xz, xy."""
        b = self.__b
        tensor = self.__pressure_tensor()
        diagonal = np.diag(tensor).copy()
        if b.style == "iso" or b.couple == "xyz":
            diagonal[:] = diagonal.mean()
        elif b.couple == "xy":
            diagonal[:2] = diagonal[:2].mean()
        elif b.couple == "xz":
            diagonal[[0, 2]] = diagonal[[0, 2]].mean()
        elif b.couple == "yz":
            diagonal[1:] = diagonal[1:].mean()
        out = np.zeros(6)
        out[:3] = diagonal
        out[3:] = [tensor[1, 2], tensor[0, 2], tensor[0, 1]]
        if not np.isfinite(out).all():
            raise CalculationFailed("the pressure is not finite: the run is unstable")
        return out

    def __press_target(self, step: int) -> None:
        b = self.__b
        fraction = self.__fraction(step)
        target = self.__p_start + fraction * (self.__p_stop - self.__p_start)
        self.__p_target[:3] = np.where(b.flags[:3], target[:3], self.__p_target[:3])
        self.__p_hydro = (
            float(self.__p_target[:3][b.flags[:3]].sum()) / self.__pdim
            if self.__pdim
            else 0.0
        )
        if b.style == "tri":
            self.__p_target[3:] = target[3:]
        if b.deviatoric:
            # sigma = V0 h0^-1 (P_target - P_hydro) h0^-T
            m = _symmetric(self.__p_target) - self.__p_hydro * np.eye(3)
            a = self.__h0_inv
            self.__sigma = self.__vol0 * (a @ m @ a.T)

    def __omega_dot_update(self) -> None:
        """Half a step of the velocities of the box."""
        b = self.__b
        n = self.__n
        volume = self.__volume()
        omega_dot, mass = self.__omega_dot, self.__omega_mass
        fdev = np.zeros(6)
        if b.deviatoric:
            hu = _upper(self.__h)
            fdev = _voigt(hu @ self.__sigma @ hu.T)
        if b.style == "iso":
            mtk_term1 = self.__dof * _KB * self.temperature() / (self.__pdim * n)
        else:
            mvv = np.diag(self.__mvv())
            mtk_term1 = float(mvv[b.flags[:3]].sum()) / (self.__pdim * n)
        half = 0.5 * self.__dt
        for i in range(3):
            if b.flags[i]:
                force = (self.__p_current[i] - self.__p_hydro) * volume / mass[i]
                force += mtk_term1 / mass[i] - fdev[i] / mass[i]
                omega_dot[i] += force * half
        self.__mtk_term2 = (
            float(omega_dot[:3][b.flags[:3]].sum()) / (self.__pdim * n)
            if self.__pdim
            else 0.0
        )
        if b.style == "tri":
            for i in range(3, 6):
                if b.flags[i]:
                    force = self.__p_current[i] * volume / mass[i] - fdev[i] / mass[i]
                    omega_dot[i] += force * half

    def __v_press(self) -> None:
        """Half a step of the scaling of the velocities by the barostat."""
        w = self.__omega_dot
        factor = np.exp(-0.25 * self.__dt * (w[:3] + self.__mtk_term2))
        self.__atoms.scale(factor)
        if self.__b.style == "tri":
            self.__atoms.shear(0.5 * self.__dt, w)
        self.__atoms.scale(factor)
        if self.__movable is not None:
            self.__atoms.restrict()

    def __remap(self) -> None:
        """Half a step of the box, the atoms scaled with it."""
        b = self.__b
        w = self.__omega_dot
        dto = 0.5 * self.__dt
        old = self.__h.copy()
        old_lo = self.__lo.copy()
        # the components of LAMMPS: h0 h1 h2 the lengths, h3 yz, h4 xz, h5 xy
        hv = [old[0, 0], old[1, 1], old[2, 2], old[2, 1], old[2, 0], old[1, 0]]
        lo = old_lo.copy()
        hi = old_lo + np.diag(old)

        def tilts() -> None:
            if b.flags[4]:
                e = math.exp(dto / 8 * w[0])
                hv[4] = (hv[4] * e + dto / 4 * (w[5] * hv[3] + w[4] * hv[2])) * e
            if b.flags[3]:
                e = math.exp(dto / 4 * w[1])
                hv[3] = (hv[3] * e + dto / 2 * (w[3] * hv[2])) * e
            if b.flags[5]:
                e = math.exp(dto / 4 * w[0])
                hv[5] = (hv[5] * e + dto / 2 * (w[5] * hv[1])) * e
            if b.flags[4]:
                e = math.exp(dto / 8 * w[0])
                hv[4] = (hv[4] * e + dto / 4 * (w[5] * hv[3] + w[4] * hv[2])) * e

        if b.style == "tri":
            tilts()
        for i in range(3):
            if b.flags[i]:
                e = math.exp(dto * w[i])
                f = self.__fixedpoint[i]
                lo[i] = (lo[i] - f) * e + f
                hi[i] = (hi[i] - f) * e + f
                if i == 1 and b.scale[2]:
                    hv[5] *= e
                if i == 2:
                    if b.scale[1]:
                        hv[4] *= e
                    if b.scale[0]:
                        hv[3] *= e
        if b.style == "tri":
            tilts()
        lengths = hi - lo
        new = np.array(
            [
                [lengths[0], 0.0, 0.0],
                [hv[5], lengths[1], 0.0],
                [hv[4], hv[3], lengths[2]],
            ]
        )
        self.__atoms.remap(old_lo, old, lo, new)
        self.__h = new
        self.__lo = lo
