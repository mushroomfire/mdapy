# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The thermal conductivity from the fluctuations of the heat flux in
equilibrium molecular dynamics, by the Green-Kubo relation."""

import os
import tempfile
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from mdapy import units
from mdapy.calculators._lammps import LAMMPS
from mdapy.core.system import System
from mdapy.errors import InvalidParameter
from mdapy.simulation._dynamics import MolecularDynamics

# W/(m K) of 1 eV/(Å ps K)
_W_PER_M_K = units.ELEMENTARY_CHARGE / (1e-10 * 1e-12)


@dataclass(frozen=True, eq=False, repr=False)
class GreenKubo:
    """The thermal conductivity of the fluctuations of the heat flux.

    Attributes
    ----------
    temperature : float64
        The temperature in K.
    volume : float64
        The volume in Å³.
    times : numpy.ndarray
        (C,) float64 the lags of the correlation in ps, from 0.
    running : numpy.ndarray
        (C, 3, 3) float64 the conductivity in W/(m K) of the integral of the
        correlation up to each lag.
    kappa : numpy.ndarray
        (3, 3) float64 the conductivity in W/(m K), that of the last lag.
    errors : numpy.ndarray
        (3, 3) float64 the standard error of kappa in W/(m K), from blocks of
        the flux.
    flux : numpy.ndarray
        (S, 3) float64 the heat flux times the volume in eV Å/ps, at the
        samples.
    """

    temperature: float
    volume: float
    times: np.ndarray
    running: np.ndarray
    kappa: np.ndarray
    errors: np.ndarray
    flux: np.ndarray

    def __post_init__(self):
        for array in (self.times, self.running, self.kappa, self.errors, self.flux):
            array.flags.writeable = False

    def __repr__(self) -> str:
        k = np.trace(self.kappa) / 3
        e = np.sqrt(np.trace(self.errors**2) / 3) / np.sqrt(3)
        return f"GreenKubo({k:.4g} ± {e:.2g} W/(m K) at {self.temperature:.6g} K)"


def _running(flux: np.ndarray, lags: int) -> np.ndarray:
    """(lags, 3, 3) the cumulative trapezoid integrals, over lags of one
    sample, of the correlations of the flux, each lag the mean over all its
    time origins, symmetric in the two axes."""
    count = len(flux)
    size = 1 << int(np.ceil(np.log2(2 * count)))
    spectrum = np.fft.rfft(flux, n=size, axis=0)
    correlation = np.empty((lags, 3, 3))
    for a in range(3):
        for b in range(3):
            full = np.fft.irfft(spectrum[:, a].conj() * spectrum[:, b], n=size)
            correlation[:, a, b] = full[:lags] / (count - np.arange(lags))
    correlation = 0.5 * (correlation + correlation.transpose(0, 2, 1))
    out = np.zeros_like(correlation)
    out[1:] = np.cumsum(0.5 * (correlation[1:] + correlation[:-1]), axis=0)
    return out


def green_kubo_from_flux(
    flux: ArrayLike,
    *,
    interval: float,
    volume: float,
    temperature: float,
    correlation: float,
    blocks: int = 5,
) -> GreenKubo:
    """The thermal conductivity from a heat flux of equilibrium molecular
    dynamics of any kind, by the Green-Kubo relation.

    Parameters
    ----------
    flux : array_like of float64
        (S, 3) the heat flux times the volume in eV Å/ps, sampled at equal
        intervals, as LAMMPS's compute heat/flux gives it in metal units.
    interval : float64
        The time between samples in ps.
    volume : float64
        The volume in Å³.
    temperature : float64
        The temperature in K.
    correlation : float64
        The longest lag of the correlation in ps, over which it is
        integrated.
    blocks : int32, optional
        The number of blocks of the flux for the standard error, each at
        least as long as the correlation. Default 5.

    Returns
    -------
    GreenKubo
        The conductivity, its integral up to each lag and its error.

    Raises
    ------
    InvalidParameter
        If the flux is not (S, 3) and finite, the interval, volume,
        temperature or correlation is not positive, blocks is less than 2,
        or a block is shorter than the correlation.
    """
    flux = np.array(flux, dtype=float)
    if flux.ndim != 2 or flux.shape[1] != 3 or not np.isfinite(flux).all():
        raise InvalidParameter("flux must be (S, 3) finite values")
    for name, value in (
        ("interval", interval),
        ("volume", volume),
        ("temperature", temperature),
        ("correlation", correlation),
    ):
        if not value > 0:
            raise InvalidParameter(f"{name} must be positive, not {value!r}")
    if (
        isinstance(blocks, bool)
        or not isinstance(blocks, (int, np.integer))
        or blocks < 2
    ):
        raise InvalidParameter(
            f"blocks must be an integer of 2 or more, not {blocks!r}"
        )
    lags = int(round(correlation / interval)) + 1
    if len(flux) // int(blocks) < lags:
        raise InvalidParameter(
            f"each of {blocks} blocks of {len(flux)} samples is shorter than the "
            f"correlation of {lags} samples: a longer run, or fewer blocks"
        )
    scale = interval / (units.BOLTZMANN * temperature**2 * volume) * _W_PER_M_K
    running = _running(flux, lags) * scale
    parts = np.array(
        [_running(part, lags)[-1] * scale for part in np.array_split(flux, int(blocks))]
    )
    errors = parts.std(axis=0, ddof=1) / np.sqrt(int(blocks))
    return GreenKubo(
        temperature=float(temperature),
        volume=float(volume),
        times=np.arange(lags) * float(interval),
        running=running,
        kappa=running[-1].copy(),
        errors=errors,
        flux=flux,
    )


def green_kubo(
    system: System,
    potential: LAMMPS,
    temperature: float,
    *,
    timestep: float = 0.001,
    equilibration: int = 10000,
    production: int = 100000,
    every: int = 5,
    correlation: float = 10.0,
    blocks: int = 5,
    seed: int = 1,
) -> GreenKubo:
    """The thermal conductivity of a crystal or liquid at a temperature, from
    the fluctuations of the heat flux in equilibrium molecular dynamics in
    LAMMPS, by the Green-Kubo relation.

    Parameters
    ----------
    system : System
        The atoms in a box periodic in every direction, of its size at the
        temperature, with an ``element`` column.
    potential : LAMMPS
        The interaction, as a LAMMPS calculator; a many-body one must give
        LAMMPS the centroid stress of each atom.
    temperature : float64
        Temperature in K, positive.
    timestep : float64, optional
        Time step in ps. Default 0.001.
    equilibration : int32, optional
        Steps at constant temperature first. Default 10000.
    production : int32, optional
        Steps at constant energy whose heat flux is recorded. Default 100000.
    every : int32, optional
        Steps between the samples of the flux. Default 5.
    correlation : float64, optional
        The longest lag of the correlation in ps. Default 10.
    blocks : int32, optional
        The number of blocks of the flux for the standard error. Default 5.
    seed : int32, optional
        Seed of the starting velocities. Default 1.

    Returns
    -------
    GreenKubo
        The conductivity, at the mean temperature of the run at constant
        energy, with its integral up to each lag and its error.

    Raises
    ------
    InvalidParameter
        For a box that is not periodic, a temperature or numbers of steps
        that are not positive, a potential without the centroid stress of
        each atom, the errors of ``green_kubo_from_flux``, or those of
        ``MolecularDynamics``.
    """
    if not system.box.pbc.all():
        raise InvalidParameter(
            "the Green-Kubo conductivity needs a box periodic in every direction"
        )
    if not temperature > 0:
        raise InvalidParameter(f"temperature must be positive, not {temperature!r}")
    if min(int(equilibration), int(production), int(every)) < 1:
        raise InvalidParameter("equilibration, production and every must be positive")
    every = int(every)
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "flux.txt")
        with MolecularDynamics(system, potential, timestep=timestep) as md:
            md.velocities(2 * temperature, seed=seed)
            md.run(int(equilibration), ensemble="nvt", temperature=temperature)
            md.command(
                "compute mdapy_ke all ke/atom\n"
                "compute mdapy_pe all pe/atom\n"
                "compute mdapy_stress all centroid/stress/atom NULL virial\n"
                "compute mdapy_flux all heat/flux mdapy_ke mdapy_pe mdapy_stress\n"
                f"fix mdapy_flux all ave/time {every} 1 {every} "
                f"c_mdapy_flux[1] c_mdapy_flux[2] c_mdapy_flux[3] file {path} "
                'format " %.17g"'
            )
            start = md.step
            try:
                md.run(int(production), ensemble="nve", thermo=every)
            except InvalidParameter as error:
                if "centroid" not in str(error):
                    raise
                raise InvalidParameter(
                    "the heat flux needs the centroid stress of each atom, which "
                    f"this potential does not give LAMMPS: {error}"
                ) from None
            md.command("unfix mdapy_flux")
            thermo = md.thermo.filter(md.thermo["step"] > start)
            volume = float(np.mean(thermo["volume"].to_numpy()))
            mean = float(np.mean(thermo["temperature"].to_numpy()))
        flux = np.loadtxt(path, comments="#")[:, 1:]
    return green_kubo_from_flux(
        flux,
        interval=every * float(timestep),
        volume=volume,
        temperature=mean,
        correlation=correlation,
        blocks=blocks,
    )
