# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The atoms of MDAPY's dynamics on an NVIDIA GPU: positions, velocities and
forces held on the device, where mdapy._cuda does the steps of the
integrator on every atom; each step copies back only the few numbers the
thermostats and the barostat need, and the atoms come back when asked."""

import math
import secrets
from typing import Any

import numpy as np
import polars as pl

from mdapy import units
from mdapy.calculators import _gpu
from mdapy.calculators._base import Calculator
from mdapy.calculators._springs import Springs
from mdapy.core.box import Box
from mdapy.core.system import System
from mdapy.errors import CalculationFailed, InvalidParameter

_MVV2E = units.AMU_A2_PER_PS2_TO_EV
_NOT_FINITE = "the energy or the forces are not finite: atoms may overlap"


class DeviceAtoms:
    """The interface of _HostAtoms of simulation/_integrator.py, on the GPU."""

    def __init__(
        self,
        calculator: Calculator,
        system: System,
        x: np.ndarray,
        v: np.ndarray,
        masses: np.ndarray,
        skin: float,
        movable: np.ndarray | None = None,
        rotation: np.ndarray | None = None,
    ):
        """movable, (N, 3) bool in the axes of the system, holds atoms along
        the directions where it is False; the vectors of the system are those
        of x and v times rotation transposed."""
        on = getattr(calculator, "_on_device", None)
        if on is None:
            raise InvalidParameter(
                f"{type(calculator).__name__} has no molecular dynamics on the GPU"
            )
        module = _gpu.cuda()
        self.__module = module
        self.__atoms = module.Atoms(len(system))
        self.__system = system
        self.__species = np.ascontiguousarray(calculator._species(system))
        self.__atoms.set_species(self.__species)
        self.__atoms.set_masses(np.ascontiguousarray(masses, dtype=np.float64))
        self.__atoms.set_positions(np.ascontiguousarray(x, dtype=np.float64))
        self.__atoms.set_velocities(np.ascontiguousarray(v, dtype=np.float64))
        self.__rotation = np.ascontiguousarray(
            np.eye(3) if rotation is None else rotation, dtype=np.float64
        )
        self.__movable = movable  # the atoms the system itself holds
        self.__held = False
        self.__set_free(movable)
        self.__watched = 0  # groups whose forces are summed: bits 0 to this
        self.__baths: list = []  # (bit, seed, [counter]) of each bath
        self.sums: list = []
        self.__cutoff, self.__skin = on.cutoff, skin
        self.__neighbors = module.Neighbors(on.cutoff, skin)
        self.__potential = on.potential
        self.__h = np.zeros((3, 3))  # the box, for the images of the wrapped atoms
        self.__seed = 0
        self.__counter = 0
        # a Switch: the other interaction (False for none, None for the ideal
        # gas), its potential on the device and its species, and λ
        self.__target: Any = False
        self.__other: Any = None
        self.__other_species: np.ndarray | None = None
        self.weight = 0.0
        self.parts = (math.nan, math.nan)
        self.__field: np.ndarray | None = None  # of HNEMD, in the frame

    @property
    def positions(self) -> np.ndarray:
        """Unwrapped: the GPU wraps the atoms into the box when it builds a
        neighbor list and counts the boxes it moved them by."""
        return self.__atoms.positions() + self.__atoms.images() @ self.__h

    @property
    def velocities(self) -> np.ndarray:
        return self.__atoms.velocities()

    @velocities.setter
    def velocities(self, v: np.ndarray) -> None:
        self.__atoms.set_velocities(np.ascontiguousarray(v, dtype=np.float64))

    @property
    def forces(self) -> np.ndarray:
        return self.__atoms.forces()

    def twice_kinetic(self) -> float:
        return float(np.trace(self.__atoms.kinetic())) * _MVV2E

    def mvv(self) -> np.ndarray:
        return self.__atoms.kinetic() * _MVV2E

    def compute(
        self, table: pl.DataFrame, box: Box, energy: bool, virial: bool
    ) -> tuple[float, np.ndarray]:
        """The forces, and the energy and the virial when asked; the
        neighbor list built again when the atoms have moved enough."""
        h, lo, pbc = _gpu.box_arguments(box)
        self.__h = h
        try:
            self.__neighbors.update(self.__atoms, h, lo, pbc)
        except ValueError as error:
            raise InvalidParameter(str(error)) from None
        except self.__module.NotFinite:
            raise CalculationFailed(_NOT_FINITE) from None
        driven = self.__field is not None
        if driven:  # the energies and virials of the heat flux
            self.__potential.compute_heat(self.__atoms, self.__neighbors, h, lo, pbc)
        else:
            self.__potential.compute(
                self.__atoms, self.__neighbors, h, lo, pbc, energy, virial, False
            )
        if driven:
            self.__atoms.drive(self.__field)
        if self.__target is not False:
            e, w = self.__switched(h, lo, pbc, energy, virial)
        if self.__watched:
            self.sums = list(self.__atoms.group_forces(self.__watched))
        self.indented = []
        for (kind, axis, sign, k, radius), at in zip(self.indenters, self.indenter_at):
            where = np.zeros(3)
            where[:] = at if np.ndim(at) else (float(at), 0.0, 0.0)
            out = self.__atoms.indent(
                kind, axis, sign, k, radius, where, np.ascontiguousarray(h), pbc
            )
            self.indented.append((float(out[0]), np.asarray(out[1:], dtype=float)))
        self.__atoms.add_forces()
        if self.__held:
            self.__atoms.restrict_forces()
        if self.__target is False:
            e = self.__atoms.total_energy() if energy else math.nan
            w = self.__atoms.total_virial() if virial else np.full((3, 3), np.nan)
        if energy and not math.isfinite(e):
            raise CalculationFailed(_NOT_FINITE)
        return e, w

    def __switched(
        self, h: np.ndarray, lo: np.ndarray, pbc: Any, energy: bool, virial: bool
    ) -> tuple[float, np.ndarray]:
        """(1 - λ) times the potential plus λ times the other interaction of a
        Switch: the energy and the virial, the forces mixed on the device; the
        energies of the two kept."""
        atoms, weight = self.__atoms, self.weight
        first = atoms.total_energy() if energy else math.nan
        work = atoms.total_virial() if virial else np.full((3, 3), np.nan)
        other, push = 0.0, np.zeros((3, 3))
        if self.__target is None:  # the ideal gas
            atoms.mix_forces(weight, False)
        elif isinstance(self.__target, Springs):
            atoms.stash_forces()
            other = atoms.springs(np.ascontiguousarray(h))
            atoms.mix_forces(weight, True)
        else:
            atoms.stash_forces()
            if self.__other_species is not None:
                atoms.set_species(self.__other_species)
            with_virial = virial and "virial" in self.__target.implemented
            self.__other.potential.compute(
                atoms, self.__neighbors, h, lo, pbc, energy, with_virial, False
            )
            other = atoms.total_energy() if energy else math.nan
            if with_virial:
                push = atoms.total_virial()
            if self.__other_species is not None:
                atoms.set_species(self.__species)
            atoms.mix_forces(weight, True)
        self.parts = (first, other)
        return (1.0 - weight) * first + weight * other, (
            1.0 - weight
        ) * work + weight * push

    def set_switch(self, target: Any) -> None:
        """The other interaction of a Switch, None for none, or False for no
        switch; a calculator on the GPU, or Springs."""
        if isinstance(self.__target, Springs):
            self.__atoms.clear_springs()
        if self.__other is not None and self.__other.cutoff > self.__cutoff:
            self.__neighbors = self.__module.Neighbors(self.__cutoff, self.__skin)
        self.__target, self.__other, self.__other_species = target, None, None
        self.weight = 0.0
        self.parts = (math.nan, math.nan)
        if target is False or target is None:
            return
        if isinstance(target, Springs):
            self.__atoms.set_springs(
                np.ascontiguousarray(target.sites, dtype=np.float64),
                np.ascontiguousarray(target.constants, dtype=np.float64),
            )
            return
        on = getattr(target, "_on_device", None)
        if on is None:
            self.__target = False
            raise InvalidParameter(
                "the target of a Switch on the GPU must compute on the GPU too: "
                f'{type(target).__name__} with device="cuda"'
            )
        self.__other = on
        species = np.ascontiguousarray(target._species(self.__system))
        if not np.array_equal(species, self.__species):
            self.__other_species = species
        if on.cutoff > self.__cutoff:  # one list for both, as each sorts the atoms
            self.__neighbors = self.__module.Neighbors(on.cutoff, self.__skin)

    def set_field(self, field: np.ndarray | None) -> None:
        """The driving field of HNEMD in the frame, or None for none."""
        self.__field = (
            None if field is None else np.ascontiguousarray(field, dtype=np.float64)
        )

    def heat_flux(self) -> np.ndarray:
        """(3,) the potential part of the heat flux times the volume in the
        frame, of the virials of the last computation and the velocities now."""
        return self.__atoms.heat_flux()

    def kick(self, c: float) -> None:
        self.__atoms.kick(c)

    def drift(self, dt: float) -> None:
        self.__atoms.drift(dt)

    def kick_drift(self, c: float, dt: float) -> None:
        self.__atoms.kick_drift(c, dt)

    indenters: tuple = ()  # (kind, axis, sign, k, radius) of each indenter
    indenter_at: tuple = ()  # their centers or planes at the step
    indented: list = []  # (energy, force on it) of each, of the last computation

    def speed_squares(self) -> float:
        """The sum of the squares of the velocities."""
        return float(self.__atoms.speed_squares())

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
        return float(
            self.__atoms.msst_kick(
                float(half), float(base), float(axial), int(axis), bool(trial), ftm2v
            )
        )

    def reflect(self, normal: np.ndarray, offset: float, speed: float) -> None:
        """A flat wall: each atom with x . normal < offset put back by as
        much, its velocity along the normal made 2 speed - v . normal."""
        self.__atoms.reflect(
            np.ascontiguousarray(normal, dtype=np.float64), float(offset), float(speed)
        )

    def scale(self, factor: Any) -> None:
        f = np.broadcast_to(np.asarray(factor, dtype=np.float64), (3,))
        self.__atoms.scale(float(f[0]), float(f[1]), float(f[2]))

    def shear(self, half: float, w: np.ndarray) -> None:
        self.__atoms.shear(half, float(w[3]), float(w[4]), float(w[5]))

    def restrict(self) -> None:
        self.__atoms.restrict_velocities()

    def __set_free(self, free: np.ndarray | None) -> None:
        self.__held = free is not None
        if self.__held:
            self.__atoms.set_movable(
                np.ascontiguousarray(free, dtype=bool), self.__rotation
            )
        else:
            self.__atoms.clear_movable()

    def set_controls(
        self,
        free: np.ndarray | None,
        prescribed: np.ndarray | None,
        added: np.ndarray | None,
        watched: tuple,
        baths: tuple,
    ) -> None:
        """The controls of a run, as _HostAtoms.set_controls takes them: the
        groups whose forces are summed and the baths as bits of a word of
        each atom."""
        atoms = self.__atoms
        self.__set_free(free)
        if prescribed is None:
            atoms.clear_prescribed()
        else:
            atoms.set_prescribed(np.ascontiguousarray(prescribed, dtype=np.float64))
        if added is None:
            atoms.clear_added()
        else:
            atoms.set_added(np.ascontiguousarray(added, dtype=np.float64))
        masks = [*watched, *(mask for mask, _ in baths)]
        if len(masks) > 31:
            raise InvalidParameter("the GPU takes at most 31 groups moved or bathed")
        if masks:
            bits = np.zeros(len(masks[0]), dtype=np.int32)
            for k, mask in enumerate(masks):
                bits[mask] |= np.int32(1 << k)
            atoms.set_groups(bits)
        else:
            atoms.clear_groups()
        self.__watched = len(watched)
        self.__baths = [
            (len(watched) + k, int(seed) % (1 << 63), [0])
            for k, (_, seed) in enumerate(baths)
        ]
        self.sums = []

    def clear_controls(self) -> None:
        atoms = self.__atoms
        self.__set_free(self.__movable)
        atoms.clear_prescribed()
        atoms.clear_added()
        atoms.clear_groups()
        self.__watched = 0
        self.__baths = []
        self.sums = []

    def twice_kinetic_free(self) -> float:
        return float(self.__atoms.kinetic_free()) * _MVV2E

    def bath(self, k: int, c1: float, c2: float, kt: float) -> float:
        """Bath k for half a step: the kinetic energy it added, in eV."""
        bit, seed, counter = self.__baths[k]
        heat = self.__atoms.bath(bit, c1, c2, kt, seed, counter[0])
        counter[0] += 1
        return float(heat) * _MVV2E

    def remap(
        self, old_lo: np.ndarray, old: np.ndarray, lo: np.ndarray, new: np.ndarray
    ) -> None:
        self.__atoms.remap(
            np.ascontiguousarray(old_lo, dtype=np.float64),
            np.ascontiguousarray(np.linalg.inv(old)),
            np.ascontiguousarray(lo, dtype=np.float64),
            np.ascontiguousarray(new, dtype=np.float64),
        )
        self.__h = np.array(new, dtype=np.float64)

    def seed(self, seed: int | None) -> None:
        self.__seed = secrets.randbits(63) if seed is None else int(seed)
        self.__counter = 0

    def langevin(self, c1: float, c2: float, kt: float, keep: bool = False) -> None:
        """v = c1 v + c2 sqrt(kt / m) g, g Gaussian of the seed and a counter,
        with the noise of held directions removed; with keep, less an equal
        share of the momentum of the noise."""
        self.__atoms.langevin(c1, c2, kt, self.__seed, self.__counter, keep)
        self.__counter += 1
        if self.__held:
            self.__atoms.restrict_velocities()
