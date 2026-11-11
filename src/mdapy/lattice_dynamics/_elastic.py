# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Elastic constants from the stress under small strains."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from mdapy.analysis._symmetry import _rotations
from mdapy.calculators._base import Calculator
from mdapy.core.box import Box
from mdapy.core.result import Result
from mdapy.core.system import System
from mdapy.errors import InvalidParameter
from mdapy.lattice_dynamics._sheets import _in_periodic_box, _is_sheet
from mdapy.simulation._minimize import minimize

# Voigt order xx, yy, zz, yz, xz, xy
_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
# the Voigt index of each pair of Cartesian indices
_VOIGT = np.array([[0, 5, 4], [5, 1, 3], [4, 3, 2]])
# a shear component of the compliance in Voigt order holds its tensor entry
# twice, as an engineering strain is twice the tensor one
_HALF_SHEAR = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
# the same order in the plane of a sheet
_SHEET_PAIRS = ((0, 0), (1, 1), (0, 1))
# N/m per GPa Å
_N_PER_M = 0.1


def _components(tensor: np.ndarray, pairs: tuple) -> np.ndarray:
    return np.array([tensor[a, b] for a, b in pairs])


def _voigt(tensor: np.ndarray) -> np.ndarray:
    return _components(tensor, _PAIRS)


def _directions(direction: ArrayLike) -> np.ndarray:
    """Unit vectors along one direction (3,) or along each of (N, 3)."""
    n = np.asarray(direction, dtype=np.float64)
    if n.ndim not in (1, 2) or n.shape[-1] != 3:
        raise InvalidParameter(
            f"direction must be a vector (3,) or vectors (N, 3), not of shape {n.shape}"
        )
    length = np.linalg.norm(n, axis=-1, keepdims=True)
    if not np.all(np.isfinite(n)) or np.any(length == 0):
        raise InvalidParameter("a direction must be finite and not zero")
    return n / length


@dataclass(frozen=True)
class ElasticResult(Result):
    """Elastic constants of a crystal.

    Attributes
    ----------
    constants : numpy.ndarray
        (6, 6) float64 elastic constants in GPa, in Voigt order xx, yy, zz, yz, xz,
        xy; column k is the stress per unit of strain component k.
    errors : numpy.ndarray or None
        (6, 6) float64 standard errors of the constants in GPa, from molecular
        dynamics; None otherwise.
    """

    constants: np.ndarray
    errors: np.ndarray | None = None

    def __compliance(self) -> np.ndarray:
        if np.linalg.cond(self.constants) > 1e12:
            raise InvalidParameter(
                "the elastic constants cannot be inverted: for a sheet, make its "
                "box not periodic along z and read its moduli in N/m"
            )
        return np.linalg.inv(self.constants)

    def bulk_modulus(self, average: str = "hill") -> float:
        """The bulk modulus in GPa: the "voigt", "reuss" or "hill" average."""
        c, s = self.constants, self.__compliance()
        voigt = (c[0, 0] + c[1, 1] + c[2, 2] + 2 * (c[0, 1] + c[0, 2] + c[1, 2])) / 9
        reuss = 1 / (s[0, 0] + s[1, 1] + s[2, 2] + 2 * (s[0, 1] + s[0, 2] + s[1, 2]))
        return float(self.__pick(average, voigt, reuss))

    def shear_modulus(self, average: str = "hill") -> float:
        """The shear modulus in GPa: the "voigt", "reuss" or "hill" average."""
        c, s = self.constants, self.__compliance()
        voigt = (
            c[0, 0] + c[1, 1] + c[2, 2] - (c[0, 1] + c[0, 2] + c[1, 2])
            + 3 * (c[3, 3] + c[4, 4] + c[5, 5])
        ) / 15  # fmt: skip
        reuss = 15 / (
            4 * (s[0, 0] + s[1, 1] + s[2, 2]) - 4 * (s[0, 1] + s[0, 2] + s[1, 2])
            + 3 * (s[3, 3] + s[4, 4] + s[5, 5])
        )  # fmt: skip
        return float(self.__pick(average, voigt, reuss))

    def __compliance_tensor(self) -> np.ndarray:
        """The compliance as a tensor (3, 3, 3, 3) in 1/GPa."""
        s = self.__compliance() * _HALF_SHEAR[:, None] * _HALF_SHEAR[None, :]
        return s[_VOIGT[:, :, None, None], _VOIGT[None, None, :, :]]

    def young_modulus(self, direction: ArrayLike) -> float | np.ndarray:
        """The Young's modulus of the crystal in GPa along a direction.

        Parameters
        ----------
        direction : array_like
            A direction (3,) float64 in the axes of the box, of any length, or
            directions (N, 3) float64.

        Returns
        -------
        float64 or numpy.ndarray
            The modulus along the direction, or (N,) float64 along each.

        Raises
        ------
        InvalidParameter
            For a direction of another shape, zero or not finite.
        """
        n = _directions(direction)
        modulus = 1 / np.einsum(
            "...i,...j,...k,...l,ijkl->...", n, n, n, n, self.__compliance_tensor()
        )
        return float(modulus) if n.ndim == 1 else modulus

    def linear_compressibility(self, direction: ArrayLike) -> float | np.ndarray:
        """The linear compressibility of the crystal in 1/GPa along a direction.

        Parameters
        ----------
        direction : array_like
            A direction (3,) float64 in the axes of the box, of any length, or
            directions (N, 3) float64.

        Returns
        -------
        float64 or numpy.ndarray
            The relative shortening along the direction per unit of hydrostatic
            pressure, or (N,) float64 along each.

        Raises
        ------
        InvalidParameter
            For a direction of another shape, zero or not finite.
        """
        n = _directions(direction)
        compressibility = np.einsum(
            "...i,...j,ijkk->...", n, n, self.__compliance_tensor()
        )
        return float(compressibility) if n.ndim == 1 else compressibility

    def universal_anisotropy(self) -> float:
        """The universal anisotropy index, zero for an isotropic crystal."""
        shear = self.shear_modulus("voigt") / self.shear_modulus("reuss")
        bulk = self.bulk_modulus("voigt") / self.bulk_modulus("reuss")
        return float(5 * shear + bulk - 6)

    @staticmethod
    def __pick(average: str, voigt: float, reuss: float) -> float:
        if average == "voigt":
            return voigt
        if average == "reuss":
            return reuss
        if average == "hill":
            return 0.5 * (voigt + reuss)
        raise InvalidParameter(
            f'average must be "voigt", "reuss" or "hill", not {average!r}'
        )


@dataclass(frozen=True)
class SheetElasticResult(Result):
    """Elastic constants of a sheet, per unit of its area.

    Attributes
    ----------
    constants : numpy.ndarray
        (3, 3) float64 elastic constants in N/m, in the order xx, yy, xy; column k
        is the tension per unit of strain component k.
    """

    constants: np.ndarray

    def __compliance(self) -> np.ndarray:
        if np.linalg.cond(self.constants) > 1e12:
            raise InvalidParameter(
                "the elastic constants cannot be inverted: the sheet has no "
                "stiffness in some direction"
            )
        return np.linalg.inv(self.constants)

    @staticmethod
    def __index(axis: str) -> int:
        if axis not in ("x", "y"):
            raise InvalidParameter(f'axis must be "x" or "y", not {axis!r}')
        return "xy".index(axis)

    def young_modulus(self, axis: str = "x") -> float:
        """The Young's modulus in N/m along "x" or "y"."""
        k = self.__index(axis)
        return float(1 / self.__compliance()[k, k])

    def poisson_ratio(self, axis: str = "x") -> float:
        """The contraction across a stretch along "x" or "y", over the stretch."""
        k = self.__index(axis)
        s = self.__compliance()
        return float(-s[0, 1] / s[k, k])

    def shear_modulus(self) -> float:
        """The shear modulus in N/m."""
        return float(1 / self.__compliance()[2, 2])

    def area_modulus(self) -> float:
        """The area modulus in N/m: the tension per unit of relative change
        of the area."""
        s = self.__compliance()
        return float(1 / (s[0, 0] + s[1, 1] + 2 * s[0, 1]))


def _strained(system: System, strain: np.ndarray) -> System:
    """The system and its box under a homogeneous strain."""
    deformation = np.eye(3) + strain
    origin = system.box.origin
    box = Box(system.box.matrix @ deformation.T, origin=origin, pbc=system.box.pbc)
    return system.with_box(box).with_columns(
        positions=origin + (system["positions"] - origin) @ deformation.T
    )


def _image(
    tensor: np.ndarray,
    known: list[tuple[np.ndarray, np.ndarray]],
    rotations: np.ndarray,
    strain: float,
) -> np.ndarray | None:
    """The stress under a strain that a rotation makes from a known one, or
    None."""
    for done, stress in known:
        for rotation in rotations:
            if np.allclose(
                rotation @ done @ rotation.T, tensor, rtol=0, atol=1e-9 * strain
            ):
                return rotation @ stress @ rotation.T
    return None


def elastic_constants(
    system: System,
    calculator: Calculator,
    *,
    strain: float = 1e-3,
    relax: bool = True,
    max_force: float = 1e-4,
    max_steps: int = 5000,
    symmetry: bool | None = None,
) -> ElasticResult | SheetElasticResult:
    """Elastic constants from the stress under small strains.

    Parameters
    ----------
    system : System
        The crystal, periodic in every direction, or a sheet: periodic along
        two vectors in the x-y plane and not along a third along z. Best at
        zero stress (relax its box first with ``minimize(..., cell="full")``).
    calculator : Calculator
        Computes the virial, and energies and forces to relax the positions.
    strain : float64, optional
        Magnitude of the strains applied in each direction, positive and
        negative; shear strains are engineering strains. Default 1e-3.
    relax : bool, optional
        Relax the positions under each strain. Default True.
    max_force : float64, optional
        Convergence of each relaxation in eV/Å. Default 1e-4.
    max_steps : int32, optional
        Largest number of steps of each relaxation. Default 5000.
    symmetry : bool, optional
        Take the stress under a strain that a symmetry operation of the
        crystal, found by spglib, makes from another one as the image of its
        stress: True, False, or by default whenever spglib is installed.

    Returns
    -------
    ElasticResult or SheetElasticResult
        The (6, 6) elastic constants in GPa; of a sheet, the (3, 3) elastic
        constants in N/m.

    Raises
    ------
    InvalidParameter
        If the box is neither periodic nor that of a sheet, the strain is not
        positive, the calculator has no virial, a relaxation does not
        converge, or symmetry is True without spglib.
    """
    sheet = _is_sheet(system.box, "elastic constants")
    if not strain > 0:
        raise InvalidParameter(f"strain must be positive, not {strain!r}")
    needed = {"virial"} | ({"energy", "forces"} if relax else set())
    if not needed <= calculator.implemented:
        raise InvalidParameter(
            f"{type(calculator).__name__} does not compute {sorted(needed)}"
        )
    rotations = _rotations(_in_periodic_box(system) if sheet else system, symmetry)
    known: list[tuple[np.ndarray, np.ndarray]] = []  # strains and their stresses
    pairs = _SHEET_PAIRS if sheet else _PAIRS
    constants = np.zeros((len(pairs), len(pairs)))
    for k, (a, b) in enumerate(pairs):
        stresses = []
        for sign in (1.0, -1.0):
            tensor = np.zeros((3, 3))
            if a == b:
                tensor[a, a] = sign * strain
            else:  # engineering shear strain: half on each off-diagonal entry
                tensor[a, b] = tensor[b, a] = 0.5 * sign * strain
            image = _image(tensor, known, rotations, strain)
            if image is not None:
                stresses.append(_components(image, pairs))
                continue
            strained = _strained(system, tensor)
            if relax:
                outcome = minimize(
                    strained, calculator, max_force=max_force, max_steps=max_steps
                )
                if not outcome.converged:
                    raise InvalidParameter(
                        f"the relaxation under strain {k} did not converge in {max_steps} steps"
                    )
                stress = outcome.stress
            else:
                stress = calculator.compute(strained, ("virial",)).stress
            known.append((tensor, stress))
            stresses.append(_components(stress, pairs))
        constants[:, k] = (stresses[0] - stresses[1]) / (2 * strain)
    if sheet:
        return SheetElasticResult(constants * system.box.matrix[2, 2] * _N_PER_M)
    return ElasticResult(constants)
