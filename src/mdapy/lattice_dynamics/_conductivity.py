# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The lattice thermal conductivity of a crystal in the relaxation-time
approximation, from its second and third order force constants."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from mdapy import units
from mdapy.analysis._symmetry import _rotations
from mdapy.errors import InvalidParameter
from mdapy.lattice_dynamics._force_constants import ForceConstants
from mdapy.lattice_dynamics._phonons import (
    _SAME_VELOCITY,
    _masses,
    _repeat,
    _supercell,
)
from mdapy.lattice_dynamics._sheets import _is_sheet

_HBAR = units.PLANCK_SI / (2 * np.pi)  # J s
_EV = units.ELEMENTARY_CHARGE  # J
_AMU = units.ATOMIC_MASS_UNIT_SI  # kg
_KB = units.BOLTZMANN_SI  # J/K
# the wave vectors of a chunk of the mesh taken at once
_CHUNK = 256


@dataclass(frozen=True, eq=False, repr=False)
class ThermalConductivity:
    """The lattice thermal conductivity of a crystal in the relaxation-time
    approximation, with the phonons of the irreducible wave vectors of the
    mesh.

    Attributes
    ----------
    temperatures : numpy.ndarray
        (T,) float64 temperatures in K.
    kappa : numpy.ndarray
        (T, 3, 3) float64 thermal conductivity in W/(m K).
    q : numpy.ndarray
        (Q, 3) float64 the irreducible wave vectors of the mesh, in units of
        the reciprocal lattice vectors of the cell.
    weights : numpy.ndarray
        (Q,) int64 the number of wave vectors of the mesh that each stands
        for.
    frequencies : numpy.ndarray
        (Q, 3n) float64 frequencies in THz.
    group_velocities : numpy.ndarray
        (Q, 3n, 3) float64 group velocities in Å/ps.
    gamma : numpy.ndarray
        (T, Q, 3n) float64 the imaginary part of the self energy of each
        mode by three-phonon scattering, half its linewidth, in THz.
    lifetimes : numpy.ndarray
        (T, Q, 3n) float64 lifetimes in ps, 1/(4π gamma); infinite for modes
        that do not scatter.
    heat_capacities : numpy.ndarray
        (T, Q, 3n) float64 heat capacity of each mode in eV/K.
    """

    temperatures: np.ndarray
    kappa: np.ndarray
    q: np.ndarray
    weights: np.ndarray
    frequencies: np.ndarray
    group_velocities: np.ndarray
    gamma: np.ndarray
    lifetimes: np.ndarray
    heat_capacities: np.ndarray

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            getattr(self, name).flags.writeable = False

    def __repr__(self) -> str:
        return (
            f"ThermalConductivity({len(self.temperatures)} temperatures, "
            f"{len(self.q)} irreducible wave vectors)"
        )


def _irreducible(
    counts: np.ndarray, reciprocal: np.ndarray, rotations: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The wave vectors of a mesh through Γ, in the order of its indices, the
    third fastest; the index of the irreducible one of each, under the
    rotations that map the mesh onto itself and time reversal; and the
    rotations used."""
    axes = [np.arange(c) for c in counts]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    inverse = np.linalg.inv(reciprocal)
    # the rotations and their opposites, as time reversal gives -q the
    # phonons of q
    rotations = np.concatenate([rotations, -rotations])
    rotations = rotations[
        np.unique(np.round(rotations, 8), axis=0, return_index=True)[1]
    ]
    kept, images = [], []
    for rotation in rotations:
        # the rotation on wave vectors in units of the reciprocal vectors
        turn = reciprocal @ rotation.T @ inverse
        moved = (grid / counts) @ turn * counts
        if np.abs(moved - np.round(moved)).max() > 1e-6:
            continue  # it does not map the mesh onto itself
        moved = np.round(moved).astype(np.int64) % counts
        kept.append(rotation)
        images.append((moved[:, 0] * counts[1] + moved[:, 1]) * counts[2] + moved[:, 2])
    first = np.min(images, axis=0)  # the smallest index of each orbit
    return grid / counts, first, np.array(kept)


def thermal_conductivity(
    constants: ForceConstants,
    temperatures: ArrayLike,
    *,
    mesh: int | ArrayLike,
    sigma: float,
    symmetry: bool | None = None,
) -> ThermalConductivity:
    """The lattice thermal conductivity of a crystal in the relaxation-time
    approximation, from three-phonon scattering.

    Parameters
    ----------
    constants : ForceConstants
        Second and third order force constants of a crystal periodic in every
        direction, each atom with an element or a mass, such as those of
        ``force_constants`` with ``cutoff_third``.
    temperatures : array_like of float64
        Temperatures in K, positive.
    mesh : int32 or array_like
        Number of wave vectors along each reciprocal lattice vector, of a
        mesh through Γ.
    sigma : float64
        Width in THz of the Gaussian that stands for the conservation of
        energy in each scattering.
    symmetry : bool, optional
        Compute the irreducible wave vectors only, and average the group
        velocities, with the rotations of the crystal found by spglib: True,
        False, or by default whenever spglib is installed.

    Returns
    -------
    ThermalConductivity
        The conductivity at each temperature, with the frequencies, group
        velocities, lifetimes and heat capacities of the modes of the
        irreducible wave vectors.

    Raises
    ------
    InvalidParameter
        If the force constants have no third order, the box is not periodic
        in every direction, an atom has no mass, a temperature is not
        positive, the mesh is not positive integers, sigma is not positive,
        or symmetry is True without spglib.
    """
    if constants.third is None:
        raise InvalidParameter(
            "thermal conductivity needs third order force constants: "
            "force_constants(..., cutoff_third=...)"
        )
    cell = constants.cell
    if _is_sheet(cell.box, "thermal conductivity"):
        raise InvalidParameter(
            "thermal conductivity needs a box periodic in every direction"
        )
    temperatures = np.atleast_1d(np.asarray(temperatures, dtype=float))
    if temperatures.ndim != 1 or not (temperatures > 0).all():
        raise InvalidParameter("temperatures must be positive")
    if not sigma > 0:
        raise InvalidParameter(f"sigma must be positive, not {sigma!r}")
    counts = _repeat(mesh, "mesh")
    phonons = constants.phonons()
    masses = _masses(cell) * _AMU
    n = len(cell)
    modes = 3 * n
    reciprocal = 2 * np.pi * np.linalg.inv(cell.box.matrix).T
    rotations = _rotations(cell, symmetry)
    q, first, rotations = _irreducible(counts, reciprocal, rotations)
    irreducible, weights = np.unique(first, return_counts=True)
    total = len(q)

    # the third order force constants of each triplet of atoms, at the
    # shortest images of its pairs, grouped by the atoms of the cell
    big = _supercell(cell, constants.supercell)
    third = constants.third
    a, j, k = np.nonzero(np.abs(third).sum(axis=(3, 4, 5)) > 0)
    positions = big["positions"]
    first_vector = big.box.minimum_image(positions[j] - positions[a])
    second_vector = big.box.minimum_image(positions[k] - positions[a])
    values = third[a, j, k].reshape(-1, 27) * _EV * 1e30  # J/m³
    groups = []
    for x in range(n):
        for y in range(n):
            for z in range(n):
                members = np.flatnonzero((a == x) & (j % n == y) & (k % n == z))
                if len(members):
                    groups.append((x, y, z, members))

    # the phonons of the mesh
    frequencies, vectors = phonons.modes(q)  # THz, (Q, 3n, n, 3)
    weighted = vectors / np.sqrt(masses)[None, None, :, None]
    live = frequencies > _SAME_VELOCITY
    omega = 2 * np.pi * frequencies * 1e12  # rad/s
    x_ = np.where(
        live[None], _HBAR * omega[None] / (_KB * temperatures[:, None, None]), 1.0
    )
    occupation = np.where(live[None], 1.0 / np.expm1(x_), 0.0)  # (T, Q, 3n)

    def delta(apart: np.ndarray) -> np.ndarray:
        """The Gaussian of frequencies apart in THz, per rad/s."""
        return np.exp(-0.5 * (apart / sigma) ** 2) / (
            sigma * np.sqrt(2 * np.pi) * 2 * np.pi * 1e12
        )

    gamma = np.zeros((len(temperatures), len(irreducible), modes))
    grid = np.round(q * counts).astype(np.int64)
    for slot, index in enumerate(irreducible):
        for start in range(0, total, _CHUNK):
            chunk = np.arange(start, min(start + _CHUNK, total))
            # -q + q' + q'' = 0: q'' = q - q', not moved back into the mesh
            q2 = q[index] - q[chunk]
            f2, w2 = phonons.modes(q2)
            w2 = w2 / np.sqrt(masses)[None, None, :, None]
            g2 = (grid[index] - grid[chunk]) % counts
            g2 = (g2[:, 0] * counts[1] + g2[:, 1]) * counts[2] + g2[:, 2]
            phase1 = np.exp(1j * (q[chunk] @ reciprocal) @ first_vector.T)
            phase2 = np.exp(1j * (q2 @ reciprocal) @ second_vector.T)
            strength = np.zeros((len(chunk), modes, modes, modes), dtype=complex)
            for x, y, z, members in groups:
                v3 = (phase1[:, members] * phase2[:, members]) @ values[members]
                strength += np.einsum(
                    "qabc,la,qmb,qnc->qlmn",
                    v3.reshape(-1, 3, 3, 3),
                    weighted[index, :, x].conj(),
                    weighted[chunk][:, :, y],
                    w2[:, :, z],
                    optimize=True,
                )
            live2 = f2 > _SAME_VELOCITY
            good = (
                live[index][None, :, None, None]
                & live[chunk][:, None, :, None]
                & live2[:, None, None, :]
            )
            denominator = np.where(
                good,
                omega[index][None, :, None, None]
                * omega[chunk][:, None, :, None]
                * (2 * np.pi * f2 * 1e12)[:, None, None, :],
                1.0,
            )
            strength = (
                np.where(good, np.abs(strength) ** 2 / denominator, 0.0)
                * (_HBAR / 2) ** 3
                / 36
                / total
            )
            f0 = frequencies[index][None, :, None, None]
            f1 = frequencies[chunk][:, None, :, None]
            f2_ = f2[:, None, None, :]
            joined = delta(f0 - f1 - f2_)
            parted = delta(f0 + f1 - f2_) - delta(f0 - f1 + f2_)
            for t in range(len(temperatures)):
                n1 = occupation[t][chunk][:, None, :, None]
                n2 = occupation[t][g2][:, None, None, :]
                weight = (n1 + n2 + 1) * joined + (n1 - n2) * parted
                gamma[t, slot] += (
                    18 * np.pi / _HBAR**2 * (strength * weight).sum(axis=(0, 2, 3))
                )
    gamma /= 2 * np.pi * 1e12  # THz
    # modes of one frequency share their width
    reduced = frequencies[irreducible]
    for slot in range(len(irreducible)):
        apart = np.flatnonzero(np.diff(reduced[slot]) >= _SAME_VELOCITY) + 1
        for same in np.split(np.arange(modes), apart):
            gamma[:, slot, same] = gamma[:, slot, same].mean(axis=1, keepdims=True)

    velocities = phonons.group_velocities(q[irreducible], symmetry=symmetry)
    alive = live[irreducible][None] & (gamma > 0)
    lifetimes = np.where(alive, 1 / (4 * np.pi * np.where(alive, gamma, 1.0)), np.inf)
    x = x_[:, irreducible]
    heat = np.where(
        live[irreducible][None], _KB * x**2 * np.exp(x) / np.expm1(x) ** 2, 0.0
    )
    # v v over the rotations, as the wave vectors of the orbit hold them
    v = velocities * 100  # m/s
    pairs = np.einsum("rab,qmb,rcd,qmd->qmac", rotations, v, rotations, v) / len(
        rotations
    )
    seconds = np.where(alive, lifetimes, 0.0) * 1e-12
    volume = abs(np.linalg.det(cell.box.matrix)) * 1e-30
    kappa = np.einsum("q,tqm,qmac,tqm->tac", weights, heat, pairs, seconds) / (
        total * volume
    )
    return ThermalConductivity(
        temperatures=temperatures,
        kappa=kappa,
        q=q[irreducible],
        weights=weights.astype(np.int64),
        frequencies=reduced,
        group_velocities=velocities,
        gamma=gamma,
        lifetimes=lifetimes,
        heat_capacities=heat / _EV,
    )
