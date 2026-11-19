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
        self.__atoms.set_species(np.ascontiguousarray(calculator._species(system)))
        self.__atoms.set_masses(np.ascontiguousarray(masses, dtype=np.float64))
        self.__atoms.set_positions(np.ascontiguousarray(x, dtype=np.float64))
        self.__atoms.set_velocities(np.ascontiguousarray(v, dtype=np.float64))
        self.__held = movable is not None
        if self.__held:
            self.__atoms.set_movable(
                np.ascontiguousarray(movable, dtype=bool),
                np.ascontiguousarray(rotation, dtype=np.float64),
            )
        self.__neighbors = module.Neighbors(on.cutoff, skin)
        self.__potential = on.potential
        self.__h = np.zeros((3, 3))  # the box, for the images of the wrapped atoms
        self.__seed = 0
        self.__counter = 0

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
        self.__potential.compute(
            self.__atoms, self.__neighbors, h, lo, pbc, energy, virial, False
        )
        if self.__held:
            self.__atoms.restrict_forces()
        e = math.nan
        if energy:
            e = self.__atoms.total_energy()
            if not math.isfinite(e):
                raise CalculationFailed(_NOT_FINITE)
        w = self.__atoms.total_virial() if virial else np.full((3, 3), np.nan)
        return e, w

    def kick(self, c: float) -> None:
        self.__atoms.kick(c)

    def drift(self, dt: float) -> None:
        self.__atoms.drift(dt)

    def scale(self, factor: Any) -> None:
        f = np.broadcast_to(np.asarray(factor, dtype=np.float64), (3,))
        self.__atoms.scale(float(f[0]), float(f[1]), float(f[2]))

    def shear(self, half: float, w: np.ndarray) -> None:
        self.__atoms.shear(half, float(w[3]), float(w[4]), float(w[5]))

    def restrict(self) -> None:
        self.__atoms.restrict_velocities()

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

    def langevin(self, c1: float, c2: float, kt: float) -> None:
        """v = c1 v + c2 sqrt(kt / m) g, g Gaussian of the seed and a counter,
        with the noise of held directions removed."""
        self.__atoms.langevin(c1, c2, kt, self.__seed, self.__counter)
        self.__counter += 1
        if self.__held:
            self.__atoms.restrict_velocities()
