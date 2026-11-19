# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The thermal conductivity by homogeneous nonequilibrium molecular dynamics
(HNEMD): the heat that flows along a small driving field."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from mdapy.calculators._base import Calculator
from mdapy.calculators._lammps import LAMMPS
from mdapy.core.system import System
from mdapy.core.threads import CancelToken
from mdapy.errors import InvalidParameter
from mdapy.lattice_dynamics._green_kubo import _W_PER_M_K
from mdapy.simulation._controls import _HeatField
from mdapy.simulation._dynamics import MolecularDynamics


@dataclass(frozen=True)
class HNEMD:
    """The thermal conductivity of homogeneous nonequilibrium molecular
    dynamics.

    Attributes
    ----------
    temperature : float64
        The mean temperature of the driven run in K.
    volume : float64
        The volume in Å³.
    field : numpy.ndarray
        (3,) float64 the driving field in 1/Å.
    times : numpy.ndarray
        (S,) float64 the time of each sample in ps, from the start of the
        driven run.
    running : numpy.ndarray
        (S, 3) float64 the conductivity in W/(m K) of the mean heat flux up to
        each time: the components of the conductivity tensor times the unit
        vector of the field.
    kappa : numpy.ndarray
        (3,) float64 that of the whole run in W/(m K); along the field, the
        conductivity in its direction.
    errors : numpy.ndarray
        (3,) float64 the standard error of kappa in W/(m K), from blocks of
        the run.
    flux : numpy.ndarray
        (S, 3) float64 the potential part of the heat flux times the volume in
        eV Å/ps, at the samples.
    """

    temperature: float
    volume: float
    field: np.ndarray
    times: np.ndarray
    running: np.ndarray
    kappa: np.ndarray
    errors: np.ndarray
    flux: np.ndarray

    def __post_init__(self):
        for array in (
            self.field,
            self.times,
            self.running,
            self.kappa,
            self.errors,
            self.flux,
        ):
            array.flags.writeable = False

    @property
    def along(self) -> float:
        """The conductivity along the field in W/(m K)."""
        unit = self.field / np.linalg.norm(self.field)
        return float(self.kappa @ unit)

    def __repr__(self) -> str:
        unit = self.field / np.linalg.norm(self.field)
        error = float(np.sqrt((self.errors**2) @ unit**2))
        return (
            f"HNEMD({self.along:.4g} ± {error:.2g} W/(m K) along the field at "
            f"{self.temperature:.6g} K)"
        )


def hnemd(
    system: System,
    potential: Calculator,
    temperature: float,
    *,
    field: ArrayLike,
    timestep: float = 0.001,
    equilibration: int = 10000,
    production: int = 1000000,
    every: int = 10,
    blocks: int = 5,
    thermostat_damping: float | None = None,
    seed: int = 1,
    progress: Callable[[int, int], None] | None = None,
    cancel: CancelToken | None = None,
) -> HNEMD:
    """The thermal conductivity of a crystal at a temperature by homogeneous
    nonequilibrium molecular dynamics (HNEMD) in MDAPY: each atom pushed by
    a small driving field through its virial, the heat that flows along it
    measured, at the temperature of a Nosé-Hoover thermostat.

    Parameters
    ----------
    system : System
        The atoms in a box periodic in every direction, of its size at the
        temperature, with an ``element`` column.
    potential : Calculator
        The interaction, whose virials of each atom give the heat flux: EAM,
        a pair potential or NEP, on the processor or the GPU.
    temperature : float64
        Temperature in K, positive.
    field : array_like
        (3,) float64 the driving field in 1/Å: large enough for the flux to
        stand out of its noise, small enough for the linear response; check
        that half of it gives the same conductivity.
    timestep : float64, optional
        Time step in ps. Default 0.001.
    equilibration : int32, optional
        Steps at the temperature without the field first. Default 10000.
    production : int32, optional
        Steps with the field, whose heat flux is averaged. Default 1000000.
    every : int32, optional
        Steps between the samples of the flux. Default 10.
    blocks : int32, optional
        Blocks of the run for the standard error, at least 2. Default 5.
    thermostat_damping : float64, optional
        Damping time of the thermostat in ps. Default 100 time steps.
    seed : int32, optional
        Seed of the starting velocities, positive. Default 1.
    progress : callable, optional
        Called as ``progress(done, total)`` with the steps run.
    cancel : CancelToken, optional
        Stops the calculation, which raises ``mdapy.errors.Cancelled``.

    Returns
    -------
    HNEMD
        The conductivity of the mean heat flux, and its running value and
        error.

    Raises
    ------
    InvalidParameter
        For a box that is not periodic, a temperature, field or numbers of
        steps that are not valid, a potential whose virials of each atom do
        not give the heat flux, a LAMMPS one, or the errors of
        ``MolecularDynamics``.
    Cancelled
        If cancelled.
    """
    if not isinstance(potential, Calculator) or isinstance(potential, LAMMPS):
        raise InvalidParameter("potential must be a mdapy calculator other than LAMMPS")
    if not system.box.pbc.all():
        raise InvalidParameter("HNEMD needs a box periodic in every direction")
    if not temperature > 0:
        raise InvalidParameter(f"temperature must be positive, not {temperature!r}")
    drive = _HeatField(field)
    if min(int(equilibration), int(production), int(every)) < 1:
        raise InvalidParameter("equilibration, production and every must be positive")
    if int(blocks) < 2 or int(production) // int(every) < int(blocks):
        raise InvalidParameter("blocks must be at least 2, with a sample each")
    total = int(equilibration) + int(production)
    done = 0

    def counted(steps_done: int, _: int) -> None:
        if progress is not None:
            progress(done + steps_done, total)

    nvt = {
        "ensemble": "nvt",
        "temperature": float(temperature),
        "thermostat_damping": thermostat_damping,
        "progress": counted,
        "cancel": cancel,
    }
    with MolecularDynamics(system, potential, timestep=timestep) as md:
        md.velocities(2 * temperature, seed=int(seed))
        md.run(int(equilibration), thermo=int(equilibration), **nvt)
        done = int(equilibration)
        start = md.step
        md.run(int(production), thermo=int(every), controls=drive, **nvt)
        rows = md.thermo.filter(md.thermo["step"] > start)
    flux = np.stack(rows["heat_flux"].to_numpy())
    volume = float(rows["volume"].mean())
    mean = float(rows["temperature"].mean())
    strength = float(np.linalg.norm(drive.field))
    scale = _W_PER_M_K / (mean * volume * strength)
    running = np.cumsum(flux, axis=0) / np.arange(1, len(flux) + 1)[:, None] * scale
    parts = np.array([b.mean(axis=0) for b in np.array_split(flux, int(blocks))])
    errors = parts.std(axis=0, ddof=1) / np.sqrt(int(blocks)) * scale
    return HNEMD(
        temperature=mean,
        volume=volume,
        field=drive.field,
        times=(rows["step"].to_numpy() - start) * float(timestep),
        running=running,
        kappa=running[-1].copy(),
        errors=errors,
        flux=flux,
    )
