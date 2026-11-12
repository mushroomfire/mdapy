# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Elastic constants at finite temperature from molecular dynamics."""

import numpy as np

from mdapy.calculators._lammps import LAMMPS
from mdapy.core.box import Box
from mdapy.core.system import System
from mdapy.errors import InvalidParameter
from mdapy.lattice_dynamics._elastic import (
    _N_PER_M,
    _PAIRS,
    _SHEET_PAIRS,
    ElasticResult,
    SheetElasticResult,
    _strained,
)
from mdapy.lattice_dynamics._sheets import _is_sheet
from mdapy.simulation._dynamics import MolecularDynamics


def _averaged(stresses: np.ndarray, blocks: int) -> tuple[np.ndarray, np.ndarray]:
    """The mean of a series of stresses and its standard error, from the
    means of blocks of it."""
    means = np.array([b.mean(axis=0) for b in np.array_split(stresses, blocks)])
    return stresses.mean(axis=0), means.std(axis=0, ddof=1) / np.sqrt(blocks)


def md_elastic_constants(
    system: System,
    potential: LAMMPS,
    temperature: float,
    *,
    pressure: float = 0.0,
    strain: float = 0.01,
    timestep: float = 0.001,
    equilibration: int = 10000,
    production: int = 20000,
    every: int = 10,
    blocks: int = 5,
    barostat: str = "tri",
    seed: int = 1,
) -> ElasticResult | SheetElasticResult:
    """Isothermal elastic constants of a crystal at a temperature, from the
    mean stress of molecular dynamics in LAMMPS under small strains.

    Parameters
    ----------
    system : System
        The crystal, periodic in every direction, or a sheet: periodic along
        two vectors in the x-y plane and not along a third along z; with an
        ``element`` column.
    potential : LAMMPS
        The interaction, as a LAMMPS calculator.
    temperature : float64
        Temperature in K, positive.
    pressure : float64, optional
        Pressure in GPa; of a sheet, over the height of its box. Default 0.
    strain : float64, optional
        Magnitude of the strains, positive and negative; shears are
        engineering strains. Default 0.01.
    timestep : float64, optional
        Time step in ps. Default 0.001.
    equilibration : int32, optional
        Steps before each average. Default 10000.
    production : int32, optional
        Steps of each average. Default 20000.
    every : int32, optional
        Steps between the stresses and boxes averaged. Default 10.
    blocks : int32, optional
        Number of blocks of each average for its standard error. Default 5.
    barostat : {"iso", "aniso", "tri"}, optional
        How the box changes while it finds its mean shape under pressure; a
        sheet changes in its plane only. Default "tri".
    seed : int32, optional
        Seed of the starting velocities and of the thermostat of the
        strained runs. Default 1.

    Returns
    -------
    ElasticResult or SheetElasticResult
        The (6, 6) elastic constants in GPa and their standard errors; of a
        sheet, the (3, 3) elastic constants per unit of its area in N/m and
        their standard errors.

    Raises
    ------
    InvalidParameter
        For a box that is neither periodic nor that of a sheet, a
        temperature, strain or numbers of steps that are not positive, fewer
        than two blocks, or the errors of ``MolecularDynamics``.
    """
    sheet = _is_sheet(system.box, "elastic constants")
    if not temperature > 0:
        raise InvalidParameter(f"temperature must be positive, not {temperature!r}")
    if not strain > 0:
        raise InvalidParameter(f"strain must be positive, not {strain!r}")
    if min(int(equilibration), int(production), int(every)) < 1:
        raise InvalidParameter("equilibration, production and every must be positive")
    if int(blocks) < 2 or int(production) // int(every) < int(blocks):
        raise InvalidParameter("blocks must be at least 2, with a sample each")
    run = {"thermo": int(every)}
    # the mean box under pressure, and a configuration in it
    with MolecularDynamics(system, potential, timestep=timestep) as md:
        md.velocities(2 * temperature, seed=seed)
        thermal = {
            "temperature": temperature,
            "pressure": pressure,
            "barostat": barostat,
        }
        md.run(int(equilibration), ensemble="npt", **thermal, **run)
        md.run(int(production), ensemble="npt", save_every=int(every), **thermal, **run)
        boxes = np.array([f.box.matrix for f in md.frames[1:]])
        last = md.system
    mean = Box(boxes.mean(axis=0), origin=last.box.origin, pbc=last.box.pbc)
    reference = last.with_box(mean).with_columns(
        positions=mean.to_cartesian(last.box.to_fractional(last["positions"]))
    )
    pairs = _SHEET_PAIRS if sheet else _PAIRS
    columns = [_PAIRS.index(pair) for pair in pairs]  # of the stress, in Voigt order
    constants, errors = np.zeros((len(pairs),) * 2), np.zeros((len(pairs),) * 2)
    for k, (a, b) in enumerate(pairs):
        means, spreads = [], []
        for sign in (1.0, -1.0):
            tensor = np.zeros((3, 3))
            tensor[a, b] = tensor[b, a] = sign * strain * (1 if a == b else 0.5)
            deformed = _strained(reference, tensor)
            with MolecularDynamics(deformed, potential, timestep=timestep) as md:
                # a Langevin thermostat, as a Nose-Hoover one, being
                # deterministic, does not bring all the vibrations of a crystal
                # to the temperature after the strain; the same noise for both
                # signs, which the difference of their stresses then cancels
                damping = 100 * md.timestep
                md.command(
                    f"fix mdapy_langevin all langevin {float(temperature)!r} "
                    f"{float(temperature)!r} {damping!r} {int(seed) + k + 1} zero yes"
                )
                md.run(int(equilibration), ensemble="nve", **run)
                start = md.step
                md.run(int(production), ensemble="nve", **run)
                table = md.thermo.filter(md.thermo["step"] > start)
            stresses = np.stack(table["stress"].to_numpy())[:, columns]
            if sheet:  # the tension, N/m: the stress times the height of the box
                matrix = deformed.box.matrix
                area = float(np.linalg.norm(np.cross(matrix[0], matrix[1])))
                volume = table["volume"].to_numpy()[:, None]
                stresses = stresses * volume / area * _N_PER_M
            mean_stress, error = _averaged(stresses, int(blocks))
            means.append(mean_stress)
            spreads.append(error)
        constants[:, k] = (means[0] - means[1]) / (2 * strain)
        errors[:, k] = np.hypot(spreads[0], spreads[1]) / (2 * strain)
    # the constants are symmetric: the mean of the two estimates of each
    # entry off the diagonal, from independent runs
    constants = (constants + constants.T) / 2
    diagonal = np.diag(errors).copy()
    errors = np.hypot(errors, errors.T) / 2
    np.fill_diagonal(errors, diagonal)
    if sheet:
        return SheetElasticResult(constants, errors)
    return ElasticResult(constants, errors)
