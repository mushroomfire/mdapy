# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Elastic constants at finite temperature from molecular dynamics."""

import numpy as np

from mdapy.calculators._lammps import LAMMPS
from mdapy.core.box import Box
from mdapy.core.system import System
from mdapy.errors import InvalidParameter
from mdapy.lattice_dynamics._elastic import _PAIRS, ElasticResult, _strained
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
) -> ElasticResult:
    """Isothermal elastic constants of a crystal at a temperature, from the
    mean stress of molecular dynamics in LAMMPS under small strains.

    Parameters
    ----------
    system : System
        The crystal, periodic in every direction, with an ``element``
        column.
    potential : LAMMPS
        The interaction, as a LAMMPS calculator.
    temperature : float
        Temperature in K, positive.
    pressure : float, optional
        Pressure in GPa. Default 0.
    strain : float, optional
        Magnitude of the strains, positive and negative; shears are
        engineering strains. Default 0.01.
    timestep : float, optional
        Time step in ps. Default 0.001.
    equilibration : int, optional
        Steps before each average. Default 10000.
    production : int, optional
        Steps of each average. Default 20000.
    every : int, optional
        Steps between the stresses and boxes averaged. Default 10.
    blocks : int, optional
        Number of blocks of each average for its standard error. Default 5.
    barostat : {"iso", "aniso", "tri"}, optional
        How the box changes while it finds its mean shape under pressure.
        Default "tri".
    seed : int, optional
        Seed of the starting velocities. Default 1.

    Returns
    -------
    ElasticResult
        The (6, 6) elastic constants in GPa and their standard errors.

    Raises
    ------
    InvalidParameter
        For a box that is not periodic, a temperature, strain or numbers of
        steps that are not positive, fewer than two blocks, or the errors of
        :class:`MolecularDynamics`.
    """
    if not system.box.pbc.all():
        raise InvalidParameter(
            "elastic constants need a box periodic in every direction"
        )
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
        md.run(int(production), ensemble="npt", frames=int(every), **thermal, **run)
        boxes = np.array([f.box.matrix for f in md.frames[1:]])
        last = md.system
    mean = Box(boxes.mean(axis=0), origin=last.box.origin, pbc=last.box.pbc)
    reference = last.with_box(mean).with_columns(
        positions=mean.to_cartesian(last.box.to_fractional(last["positions"]))
    )
    constants, errors = np.zeros((6, 6)), np.zeros((6, 6))
    for k, (a, b) in enumerate(_PAIRS):
        means, spreads = [], []
        for sign in (1.0, -1.0):
            tensor = np.zeros((3, 3))
            tensor[a, b] = tensor[b, a] = sign * strain * (1 if a == b else 0.5)
            with MolecularDynamics(
                _strained(reference, tensor), potential, timestep=timestep
            ) as md:
                md.run(
                    int(equilibration), ensemble="nvt", temperature=temperature, **run
                )
                start = md.step
                md.run(int(production), ensemble="nvt", temperature=temperature, **run)
                table = md.thermo.filter(md.thermo["step"] > start)
            stresses = np.stack(table["stress"].to_numpy())
            mean_stress, error = _averaged(stresses, int(blocks))
            means.append(mean_stress)
            spreads.append(error)
        constants[:, k] = (means[0] - means[1]) / (2 * strain)
        errors[:, k] = np.hypot(spreads[0], spreads[1]) / (2 * strain)
    return ElasticResult(constants, errors)
