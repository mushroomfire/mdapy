# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Elastic constants at finite temperature from molecular dynamics: from the
mean stresses under small strains, or from the fluctuations of the stress."""

import tempfile
from pathlib import Path

import numpy as np

from mdapy import units
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


# the 21 values of LAMMPS's compute born/matrix: their entries of the (6, 6)
# matrix in Voigt order
_BORN = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5),
         (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
         (1, 2), (1, 3), (1, 4), (1, 5),
         (2, 3), (2, 4), (2, 5),
         (3, 4), (3, 5),
         (4, 5)]  # fmt: skip
# the step of the strains of the numerical second derivatives of the Born term:
# right for the potentials tried, from EAM to NEP, which 1e-6 is not
_DELTA = 1e-5
# the Born term once in this many records of the stress
_BORN_EVERY = 10


def _tensor(voigt: np.ndarray) -> np.ndarray:
    """(..., 6, 6) constants in Voigt order as (..., 3, 3, 3, 3) tensors."""
    full = np.zeros(voigt.shape[:-2] + (3, 3, 3, 3))
    for a, (i, j) in enumerate(_PAIRS):
        for b, (k, m) in enumerate(_PAIRS):
            for p, q in {(i, j), (j, i)}:
                for r, t in {(k, m), (m, k)}:
                    full[..., p, q, r, t] = voigt[..., a, b]
    return full


def _voigt(tensor: np.ndarray) -> np.ndarray:
    return np.array([[tensor[i, j, k, m] for k, m in _PAIRS] for i, j in _PAIRS])


def _fluctuations(
    reference: System,
    potential: LAMMPS,
    temperature: float,
    *,
    timestep: float,
    equilibration: int,
    production: int,
    every: int,
    blocks: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """The (6, 6) elastic constants of the reference box in GPa and their
    standard errors, from one run at constant volume: the Born term, the
    fluctuations of the virial stress and the kinetic term, as stress-strain
    coefficients under the mean stress."""
    rotation = reference.box.to_lammps().rotation  # into the frame of LAMMPS
    with tempfile.TemporaryDirectory() as folder:
        stress_file = Path(folder) / "stress.txt"
        born_file = Path(folder) / "born.txt"
        with MolecularDynamics(reference, potential, timestep=timestep) as md:
            damping = 100 * md.timestep
            md.command(
                f"fix mdapy_langevin all langevin {float(temperature)!r} "
                f"{float(temperature)!r} {damping!r} {int(seed)} zero yes"
            )
            md.run(equilibration, ensemble="nve", thermo=every)
            start = md.step
            md.command("compute mdapy_virial all pressure NULL virial")
            md.command(
                f"fix mdapy_stress all ave/time {every} 1 {every} "
                + " ".join(f"c_mdapy_virial[{k}]" for k in range(1, 7))
                + f" file {stress_file}"
            )
            md.command(
                f"compute mdapy_born all born/matrix numdiff {_DELTA!r} mdapy_virial"
            )
            sparse = every * _BORN_EVERY
            md.command(
                f"fix mdapy_born all ave/time {sparse} 1 {sparse} c_mdapy_born[*] "
                f"file {born_file}"
            )
            md.run(production, ensemble="nve", thermo=every)
            md.command("unfix mdapy_stress")
            md.command("unfix mdapy_born")
            table = md.thermo.filter(md.thermo["step"] > start)
        # pxx pyy pzz pxy pxz pyz in bar, as stresses in GPa in the frame of
        # the system
        p = np.atleast_2d(np.loadtxt(stress_file, comments="#"))[:, 1:]
        tensors = np.zeros((len(p), 3, 3))
        for column, (a, b) in enumerate(
            ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
        ):
            tensors[:, a, b] = tensors[:, b, a] = -p[:, column] * units.BAR_TO_GPA
        tensors = rotation @ tensors @ rotation.T
        virial = np.stack([tensors[:, a, b] for a, b in _PAIRS], axis=1)
        values = np.atleast_2d(np.loadtxt(born_file, comments="#"))[:, 1:]
    volume = float(table["volume"].mean())
    born = np.zeros((len(values), 6, 6))
    for m, (i, j) in enumerate(_BORN):
        born[:, i, j] = born[:, j, i] = values[:, m]
    # eV per box to GPa, in the frame of the system
    born = _tensor(born) / volume * units.EV_PER_A3_TO_GPA
    born = np.einsum(
        "ia,jb,kc,ld,nabcd->nijkl", rotation, rotation, rotation, rotation, born
    )
    born = np.array([_voigt(b) for b in born])
    kt = units.BOLTZMANN * float(temperature)
    kinetic = np.zeros((6, 6))
    kinetic[[0, 1, 2], [0, 1, 2]] = 2.0
    kinetic[[3, 4, 5], [3, 4, 5]] = 1.0
    # the momenta of all atoms but the center of mass, whose momentum the
    # thermostat keeps at zero, as the temperature of LAMMPS counts them
    kinetic *= (len(reference) - 1) * kt / volume * units.EV_PER_A3_TO_GPA
    factor = volume / kt / units.EV_PER_A3_TO_GPA  # 1/GPa
    stress = np.stack(table["stress"].to_numpy())
    eye = np.eye(3)

    def under(voigt: np.ndarray) -> np.ndarray:
        """The term that makes the constants stress-strain coefficients under
        a mean stress (Wallace)."""
        mean = np.zeros((3, 3))
        for k, (a, b) in enumerate(_PAIRS):
            mean[a, b] = mean[b, a] = voigt[k]
        return _voigt(
            0.5
            * (
                np.einsum("ik,jl->ijkl", mean, eye)
                + np.einsum("jk,il->ijkl", mean, eye)
                + np.einsum("il,jk->ijkl", mean, eye)
                + np.einsum("jl,ik->ijkl", mean, eye)
            )
            - np.einsum("ij,kl->ijkl", mean, eye)
        )

    def constants(virials, borns, stresses):
        spread = np.cov(virials.T, bias=True)
        return (
            borns.mean(axis=0)
            - factor * spread
            + kinetic
            + under(stresses.mean(axis=0))
        )

    total = constants(virial, born, stress)
    parts = np.array(
        [
            constants(*part)
            for part in zip(
                np.array_split(virial, blocks),
                np.array_split(born, blocks),
                np.array_split(stress, blocks),
            )
        ]
    )
    errors = parts.std(axis=0, ddof=1) / np.sqrt(blocks)
    return (total + total.T) / 2, (errors + errors.T) / 2


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
    method: str = "strain",
) -> ElasticResult | SheetElasticResult:
    """Isothermal elastic constants of a crystal at a temperature, from
    molecular dynamics in LAMMPS: from the mean stress under small strains,
    or from the fluctuations of the stress at the mean box.

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
    method : {"strain", "fluctuation"}, optional
        From the mean stress of a run under each strain, or from one run at
        the mean box: the Born term, the fluctuations of the virial stress
        and the kinetic term; ``strain`` is then unused. Default "strain".

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
    if method not in ("strain", "fluctuation"):
        raise InvalidParameter(
            f'method must be "strain" or "fluctuation", not {method!r}'
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
        md.run(int(production), ensemble="npt", save_every=int(every), **thermal, **run)
        boxes = np.array([f.box.matrix for f in md.frames[1:]])
        last = md.system
    mean = Box(boxes.mean(axis=0), origin=last.box.origin, pbc=last.box.pbc)
    reference = last.with_box(mean).with_columns(
        positions=mean.to_cartesian(last.box.to_fractional(last["positions"]))
    )
    pairs = _SHEET_PAIRS if sheet else _PAIRS
    if method == "fluctuation":
        constants, errors = _fluctuations(
            reference,
            potential,
            temperature,
            timestep=timestep,
            equilibration=int(equilibration),
            production=int(production),
            every=int(every),
            blocks=int(blocks),
            seed=int(seed),
        )
        if sheet:  # the components in the plane, times the height of the box
            keep = [_PAIRS.index(pair) for pair in pairs]
            height = reference.box.volume / float(
                np.linalg.norm(np.cross(*reference.box.matrix[:2]))
            )
            scale = height * _N_PER_M
            return SheetElasticResult(
                constants[np.ix_(keep, keep)] * scale,
                errors[np.ix_(keep, keep)] * scale,
            )
        return ElasticResult(constants, errors)
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
