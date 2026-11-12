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
from mdapy._core import analysis as _scattering
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
# corner values closer than this are one value, as in phonopy
_SAME_VERTEX = 1e-10
# the elements (wave vectors times bands cubed) of a chunk of tetrahedron
# weights
_ELEMENTS = 40000


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


def _tetrahedra_around(counts: np.ndarray, reciprocal: np.ndarray) -> np.ndarray:
    """(24, 4, 3) int64 steps on the mesh to the corners of the 24 tetrahedra
    that share a mesh point, the point itself first: six in each cell of the
    mesh, around its shortest main diagonal (the first of those of one
    length, as phonopy)."""
    step = reciprocal / counts[:, None]  # rows: the vectors of a cell
    signs = np.array([[1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1]])
    lengths = ((signs @ step) ** 2).sum(axis=1)
    # the first within rounding of the shortest
    sign = signs[np.flatnonzero(lengths <= lengths.min() * (1 + 1e-10))[0]]
    start = (1 - sign) // 2
    cell = []
    for a, b in ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)):
        second = start.copy()
        second[a] += sign[a]
        third = second.copy()
        third[b] += sign[b]
        cell.append([start, second, third, start + sign])
    cell = np.array(cell)  # (6, 4, 3) corners of the tetrahedra of a cell
    around = []
    for tetrahedron in cell:
        for corner in range(4):
            rest = [k for k in range(4) if k != corner]
            around.append(tetrahedron[[corner, *rest]] - tetrahedron[corner])
    return np.array(around, dtype=np.int64)


def _degenerate_means(frequencies: np.ndarray) -> np.ndarray:
    """(Q, B, B) the matrix of each wave vector that replaces values over its
    bands by their means over each set of degenerate bands."""
    new = np.ones(frequencies.shape, dtype=bool)
    new[:, 1:] = np.diff(frequencies, axis=1) >= _SAME_VELOCITY
    group = np.cumsum(new, axis=1)
    same = group[:, :, None] == group[:, None, :]
    return same / same.sum(axis=2, keepdims=True)


def thermal_conductivity(
    constants: ForceConstants,
    temperatures: ArrayLike,
    *,
    mesh: int | ArrayLike,
    sigma: float | None = None,
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
    sigma : float64, optional
        Width in THz of a Gaussian that stands for the conservation of
        energy in each scattering; by default the linear tetrahedron method,
        with the tetrahedra of every rotation of the crystal and the weights
        averaged over modes of one frequency.
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
    if sigma is not None and not sigma > 0:
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

    def flat(values: np.ndarray) -> np.ndarray:
        """Complex values as float64 pairs, for the kernel."""
        return np.ascontiguousarray(values, dtype=complex).view(np.float64)

    def index_of(address: np.ndarray) -> np.ndarray:
        """The index in the mesh of grid addresses, moved into it."""
        address = address % counts
        return (address[..., 0] * counts[1] + address[..., 1]) * counts[2] + address[
            ..., 2
        ]

    chunk_size = _CHUNK
    if sigma is None:
        # the tetrahedra around each point, turned by every rotation, those
        # that differ once each, so that the weights keep the symmetry
        around = _tetrahedra_around(counts, reciprocal)
        inverse = np.linalg.inv(reciprocal)
        found = {}
        for rotation in rotations:
            turn = reciprocal @ rotation.T @ inverse
            turned = np.round((around / counts) @ turn * counts).astype(np.int64)
            key = frozenset(frozenset(map(tuple, t)) for t in turned.tolist())
            found.setdefault(key, turned)
        around = np.concatenate(list(found.values()))
        clipped = np.maximum(frequencies, 0.0)
        means = _degenerate_means(frequencies)
        chunk_size = max(1, _ELEMENTS // modes**3)

    rates = np.zeros((len(temperatures), len(irreducible), modes))
    grid = np.round(q * counts).astype(np.int64)
    angular = 2 * np.pi * 1e12  # rad/s per THz
    for slot, index in enumerate(irreducible):
        first = flat(weighted[index].reshape(modes, modes))
        for start in range(0, total, chunk_size):
            chunk = np.arange(start, min(start + chunk_size, total))
            # -q + q' + q'' = 0: q'' = q - q', not moved back into the mesh
            q2 = q[index] - q[chunk]
            f2, w2 = phonons.modes(q2)
            w2 = w2 / np.sqrt(masses)[None, None, :, None]
            g2 = (grid[index] - grid[chunk]) % counts
            g2 = (g2[:, 0] * counts[1] + g2[:, 1]) * counts[2] + g2[:, 2]
            phase1 = np.exp(1j * (q[chunk] @ reciprocal) @ first_vector.T)
            phase2 = np.exp(1j * (q2 @ reciprocal) @ second_vector.T)
            # the third order force constants of q' and q'', by the atoms
            # and axes of q, q' and q''
            tensor = np.zeros((len(chunk), n, 3, n, 3, n, 3), dtype=complex)
            for x, y, z, members in groups:
                v3 = (phase1[:, members] * phase2[:, members]) @ values[members]
                tensor[:, x, :, y, :, z, :] += v3.reshape(-1, 3, 3, 3)
            joined = parted = None
            if sigma is None:
                # q' steps to a corner as q'' steps back; the weights of
                # modes of one frequency are their mean
                joined, parted = _scattering.tetrahedron_triplet_weights(
                    np.ascontiguousarray(
                        clipped[index_of(grid[chunk][:, None, None] + around)]
                    ),
                    np.ascontiguousarray(
                        clipped[
                            index_of(
                                (grid[index] - grid[chunk])[:, None, None] - around
                            )
                        ]
                    ),
                    np.ascontiguousarray(frequencies[index]),
                    _SAME_VERTEX,
                )
                shape = (len(chunk), modes, modes, modes)
                joined, parted = (
                    np.ascontiguousarray(
                        np.einsum(
                            "qmx,qlxy,qyn->qlmn",
                            means[chunk],
                            w.reshape(shape),
                            means[g2],
                        )
                    )
                    / angular
                    for w in (joined, parted)
                )
            rates[:, slot] += _scattering.three_phonon_rates(
                first,
                flat(weighted[chunk].reshape(len(chunk), modes, modes)),
                flat(w2.reshape(len(chunk), modes, modes)),
                flat(tensor),
                np.ascontiguousarray(omega[index]),
                np.ascontiguousarray(omega[chunk]),
                np.ascontiguousarray(angular * f2),
                np.ascontiguousarray(occupation[:, chunk]),
                np.ascontiguousarray(occupation[:, g2]),
                joined,
                parted,
                0.0 if sigma is None else angular * sigma,
                angular * _SAME_VELOCITY,
            ).reshape(len(temperatures), modes)
    gamma = rates * (18 * np.pi / _HBAR**2) * (_HBAR / 2) ** 3 / 36 / total
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
