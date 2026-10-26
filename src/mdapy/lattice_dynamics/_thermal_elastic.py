# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Elastic constants at finite temperature in the quasi-harmonic
approximation."""

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from mdapy import units
from mdapy.analysis._symmetry import _rotations
from mdapy.calculators._base import Calculator
from mdapy.core.box import Box
from mdapy.core.system import System
from mdapy.errors import InvalidParameter
from mdapy.lattice_dynamics._phonons import (
    _default_supercell,
    _phonons,
    _read_only,
    _supercell_matrix,
)
from mdapy.lattice_dynamics._quasi_harmonic import QuasiHarmonicResult
from mdapy.simulation._minimize import minimize

# Voigt order xx, yy, zz, yz, xz, xy
_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
_METHODS = ("quasi-harmonic", "quasi-static")


@dataclass(frozen=True)
class ThermalElasticResult:
    """Elastic constants of a crystal against temperature.

    Attributes
    ----------
    temperatures : numpy.ndarray
        (T,) temperatures in K.
    isothermal : numpy.ndarray
        (T, 6, 6) isothermal elastic constants in GPa, in Voigt order xx,
        yy, zz, yz, xz, xy.
    adiabatic : numpy.ndarray
        (T, 6, 6) adiabatic elastic constants in GPa.
    method : str
        "quasi-harmonic" or "quasi-static".
    """

    temperatures: np.ndarray
    isothermal: np.ndarray
    adiabatic: np.ndarray
    method: str

    def __post_init__(self):
        _read_only(self.temperatures, self.isothermal, self.adiabatic)


def _strain(voigt: np.ndarray) -> np.ndarray:
    """The Lagrangian strain tensor of a Voigt vector, shears engineering."""
    eta = np.zeros((3, 3))
    for k, (a, b) in enumerate(_PAIRS):
        eta[a, b] = eta[b, a] = voigt[k] if a == b else voigt[k] / 2
    return eta


def _deformed(cell: System, voigt: np.ndarray) -> System:
    """The cell under the deformation sqrt(1 + 2 eta) of a Lagrangian strain."""
    values, vectors = np.linalg.eigh(np.eye(3) + 2 * _strain(voigt))
    deformation = vectors @ np.diag(np.sqrt(values)) @ vectors.T
    box = cell.box
    origin = box.origin
    return cell.with_box(
        Box(box.matrix @ deformation.T, origin=origin, pbc=box.pbc)
    ).with_columns(positions=origin + (cell["positions"] - origin) @ deformation.T)


def _stencil(h: float) -> list[np.ndarray]:
    """The strains of central differences of second derivatives."""
    points = [np.zeros(6)]
    for i in range(6):
        for sign in (1, -1):
            points.append(sign * h * np.eye(6)[i])
    for i, j in combinations(range(6), 2):
        for si in (1, -1):
            for sj in (1, -1):
                points.append(h * (si * np.eye(6)[i] + sj * np.eye(6)[j]))
    return points


def _key(voigt: np.ndarray, h: float) -> bytes:
    """The key of a strain of the stencil of step h."""
    return np.round(voigt / h).astype(int).tobytes()


def _hessian(values: dict[bytes, np.ndarray], h: float) -> np.ndarray:
    """(6, 6, ...) second derivatives from the values at the stencil."""

    def at(*terms):
        v = np.zeros(6)
        for k, s in terms:
            v[k] += s * h
        return values[_key(v, h)]

    zero = at()
    hessian = np.zeros((6, 6) + np.shape(zero))
    for i in range(6):
        hessian[i, i] = (at((i, 1)) - 2 * zero + at((i, -1))) / h**2
    for i, j in combinations(range(6), 2):
        hessian[i, j] = hessian[j, i] = (
            at((i, 1), (j, 1))
            - at((i, 1), (j, -1))
            - at((i, -1), (j, 1))
            + at((i, -1), (j, -1))
        ) / (4 * h**2)
    return hessian


def _birch(constants: np.ndarray, pressure: float) -> np.ndarray:
    """The stress-strain coefficients of a crystal under hydrostatic
    pressure from its elastic constants, both in Voigt order."""
    stress = -pressure * np.eye(3)
    delta = np.eye(3)
    out = constants.copy()
    for p, (i, j) in enumerate(_PAIRS):
        for q, (k, m) in enumerate(_PAIRS):
            out[..., p, q] += 0.5 * (
                stress[i, k] * delta[j, m]
                + stress[j, k] * delta[i, m]
                + stress[i, m] * delta[j, k]
                + stress[j, m] * delta[i, k]
                - 2 * stress[i, j] * delta[k, m]
            )
    return out


def thermal_elastic_constants(
    qha: QuasiHarmonicResult,
    calculator: Calculator,
    *,
    method: str = "quasi-harmonic",
    strain: float = 5e-3,
    supercell=None,
    mesh=20,
    displacement: float = 0.01,
    max_force: float = 1e-6,
    max_steps: int = 5000,
    symmetry: bool | None = None,
) -> ThermalElasticResult:
    """Isothermal and adiabatic elastic constants of a crystal at the
    temperatures of a quasi-harmonic calculation.

    Parameters
    ----------
    qha : QuasiHarmonicResult
        The quasi-harmonic calculation of the crystal, with its cells; the
        cells around the volumes of its temperatures are strained.
    calculator : Calculator
        Computes energies and forces, as in the quasi-harmonic calculation.
    method : {"quasi-harmonic", "quasi-static"}, optional
        Second derivatives with respect to strain of the static energy and
        the free energy of the vibrations, from phonons of strained cells,
        or of the static energy alone, at the volume of each temperature.
        Default "quasi-harmonic".
    strain : float, optional
        Magnitude of the Lagrangian strains of the finite differences.
        Default 5e-3.
    supercell, mesh, displacement : optional
        Of the phonons of the strained cells, as in :func:`phonons` and
        :meth:`Phonons.thermal`.
    max_force : float, optional
        Convergence of the relaxation of the positions in each strained
        cell in eV/Å. Default 1e-6.
    max_steps : int, optional
        Largest number of steps of each relaxation. Default 5000.
    symmetry : bool, optional
        Compute one of each set of strains that the symmetry operations of
        the cells, found by spglib, make from each other, and use symmetry
        in the phonons too: True, False, or by default whenever spglib is
        installed.

    Returns
    -------
    ThermalElasticResult
        The isothermal and adiabatic elastic constants at each temperature.

    Raises
    ------
    InvalidParameter
        For an unknown method, a strain that is not positive, a result
        without cells, a relaxation that does not converge, imaginary
        frequencies in a strained cell, or symmetry True without spglib.
    """
    if method not in _METHODS:
        raise InvalidParameter(f"method must be one of {_METHODS}, not {method!r}")
    if not strain > 0:
        raise InvalidParameter(f"strain must be positive, not {strain!r}")
    if not qha.cells:
        raise InvalidParameter("the quasi-harmonic result has no cells")
    temperatures = qha.temperatures
    n = len(qha.cells[0])
    # the cells of the grid around the volumes of the temperatures: two more
    # on either side, and four at least
    first = max(int(np.searchsorted(qha.volumes, qha.volume.min())) - 3, 0)
    last = min(
        int(np.searchsorted(qha.volumes, qha.volume.max())) + 2, len(qha.volumes)
    )
    while last - first < min(4, len(qha.volumes)):
        first, last = max(first - 1, 0), min(last + 1, len(qha.volumes))
    cells = qha.cells[first:last]
    volumes = qha.volumes[first:last]  # per atom
    stencil = _stencil(strain)
    curvatures = []  # per cell: (6, 6, T) in eV/Å³
    keys = {_key(voigt, strain) for voigt in stencil}
    for cell in cells:
        # the strained cells keep the supercell and the periodic images of
        # the cell, so that their free energies change smoothly with strain
        matrix = supercell
        if matrix is None:
            matrix = _default_supercell(cell.box.matrix, 2 * calculator.cutoff)
        # strains related by the rotations that keep the supercell and the
        # mesh of the phonons have the same free energy
        lattices = []
        if method == "quasi-harmonic":
            lattices = [_supercell_matrix(matrix), np.diag(np.broadcast_to(mesh, 3))]
        rotations = _rotations(cell, symmetry, *lattices)
        values = {}
        for voigt in stencil:
            if _key(voigt, strain) in values:
                continue
            relaxed = minimize(
                _deformed(cell, voigt),
                calculator,
                max_force=max_force,
                max_steps=max_steps,
            )
            if not relaxed.converged:
                raise InvalidParameter(
                    f"a strained cell did not relax within {max_steps} steps"
                )
            free = np.full(len(temperatures), relaxed.energy)
            if method == "quasi-harmonic":
                vibrations = _phonons(
                    relaxed.system,
                    calculator,
                    matrix,
                    displacement,
                    symmetry,
                    reference=cell,
                ).thermal(temperatures, mesh)
                free = free + n * vibrations.free_energy
            # the strains that a rotation of the crystal makes from this one
            for rotation in rotations:
                eta = rotation @ _strain(voigt) @ rotation.T
                image = np.array([eta[a, b] * (1 if a == b else 2) for a, b in _PAIRS])
                key = _key(image, strain)
                if key in keys and np.allclose(
                    image / strain, np.round(image / strain), atol=1e-6
                ):
                    values[key] = free
        curvatures.append(_hessian(values, strain) / cell.box.volume)
    curvatures = np.array(curvatures)  # (V, 6, 6, T)
    degree = min(3, len(volumes) - 1)
    isothermal = np.zeros((len(temperatures), 6, 6))
    for k, volume in enumerate(qha.volume):
        for i in range(6):
            for j in range(6):
                c = np.polynomial.polynomial.polyfit(
                    volumes, curvatures[:, i, j, k], degree
                )
                isothermal[k, i, j] = np.polynomial.polynomial.polyval(volume, c)
    pressure = qha.pressure * units.GPA_TO_EV_PER_A3
    isothermal = _birch(isothermal, pressure)
    # the thermal expansion tensor, from the cells against volume
    matrices = np.array([c.box.matrix for c in cells])
    fits = [
        np.polynomial.polynomial.polyfit(volumes, matrices[:, a, b], degree)
        for a in range(3)
        for b in range(3)
    ]
    adiabatic = isothermal.copy()
    for k, (t, volume) in enumerate(zip(temperatures, qha.volume)):
        if t == 0:
            continue
        h = np.array(
            [np.polynomial.polynomial.polyval(volume, f) for f in fits]
        ).reshape(3, 3)
        dh = np.array(
            [
                np.polynomial.polynomial.polyval(
                    volume, np.polynomial.polynomial.polyder(f)
                )
                for f in fits
            ]
        ).reshape(3, 3)
        rate = dh.T @ np.linalg.inv(h.T)  # d(strain)/dV
        rate = 0.5 * (rate + rate.T) * qha.thermal_expansion[k] * volume  # per K
        alpha = np.array([rate[a, b] * (1 if a == b else 2) for a, b in _PAIRS])
        bulk = qha.bulk_modulus[k] * units.GPA_TO_EV_PER_A3
        capacity = (
            qha.heat_capacity[k] - t * volume * bulk * qha.thermal_expansion[k] ** 2
        )
        coupling = isothermal[k] @ alpha
        adiabatic[k] += t * volume * np.outer(coupling, coupling) / capacity
    return ThermalElasticResult(
        temperatures=temperatures,
        isothermal=isothermal * units.EV_PER_A3_TO_GPA,
        adiabatic=adiabatic * units.EV_PER_A3_TO_GPA,
        method=method,
    )
