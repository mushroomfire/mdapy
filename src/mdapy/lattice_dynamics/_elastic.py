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
# the forces left by a relaxation under a strain, as a part of those the
# strain makes
_RELATIVE = 1e-4


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


def _pair(
    first: ArrayLike | None, second: ArrayLike | None, a: str, b: str
) -> tuple[np.ndarray, np.ndarray]:
    """Unit vectors of two perpendicular directions, or of pairs of them."""
    if first is None or second is None:
        raise InvalidParameter(f"give both {a} and {b}, or neither")
    n, m = _directions(first), _directions(second)
    if n.shape != m.shape:
        raise InvalidParameter(f"{a} and {b} must have the same shape")
    if np.any(np.abs((n * m).sum(axis=-1)) > 1e-6):
        raise InvalidParameter(f"{a} and {b} must be perpendicular")
    return n, m


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

    def __bulk(self) -> tuple[float, float]:
        """The Voigt and Reuss bounds of the bulk modulus."""
        c, s = self.constants, self.__compliance()
        voigt = (c[0, 0] + c[1, 1] + c[2, 2] + 2 * (c[0, 1] + c[0, 2] + c[1, 2])) / 9
        reuss = 1 / (s[0, 0] + s[1, 1] + s[2, 2] + 2 * (s[0, 1] + s[0, 2] + s[1, 2]))
        return voigt, reuss

    def __shear(self) -> tuple[float, float]:
        """The Voigt and Reuss bounds of the shear modulus."""
        c, s = self.constants, self.__compliance()
        voigt = (
            c[0, 0] + c[1, 1] + c[2, 2] - (c[0, 1] + c[0, 2] + c[1, 2])
            + 3 * (c[3, 3] + c[4, 4] + c[5, 5])
        ) / 15  # fmt: skip
        reuss = 15 / (
            4 * (s[0, 0] + s[1, 1] + s[2, 2]) - 4 * (s[0, 1] + s[0, 2] + s[1, 2])
            + 3 * (s[3, 3] + s[4, 4] + s[5, 5])
        )  # fmt: skip
        return voigt, reuss

    def __compliance_tensor(self) -> np.ndarray:
        """The compliance as a tensor (3, 3, 3, 3) in 1/GPa."""
        s = self.__compliance() * _HALF_SHEAR[:, None] * _HALF_SHEAR[None, :]
        return s[_VOIGT[:, :, None, None], _VOIGT[None, None, :, :]]

    @staticmethod
    def __alone(average: str) -> None:
        """A modulus along directions takes no average."""
        if average != "hill":
            raise InvalidParameter(
                "average is that of a polycrystal: give it without directions"
            )

    def bulk_modulus(self, average: str = "hill") -> float:
        """The bulk modulus in GPa: the "voigt", "reuss" or "hill" average."""
        return float(self.__pick(average, *self.__bulk()))

    def shear_modulus(
        self,
        average: str = "hill",
        *,
        normal: ArrayLike | None = None,
        direction: ArrayLike | None = None,
    ) -> float | np.ndarray:
        """The shear modulus in GPa: of a polycrystal, or of the crystal sheared
        on a plane along a direction.

        Parameters
        ----------
        average : str, optional
            Of a polycrystal, the "voigt", "reuss" or "hill" average. Default
            "hill".
        normal : array_like, optional
            The normal (3,) float64 of the plane sheared, in the axes of the box
            and of any length, or normals (N, 3) float64.
        direction : array_like, optional
            The direction (3,) or directions (N, 3) float64 of the shear,
            perpendicular to the normal. Give both normal and direction, or
            neither.

        Returns
        -------
        float64 or numpy.ndarray
            The modulus, or (N,) float64 for each normal and direction.

        Raises
        ------
        InvalidParameter
            For an unknown average, an average with directions, a normal
            without a direction or the other way round, vectors of another
            shape, zero, not finite or not perpendicular.
        """
        if normal is None and direction is None:
            return float(self.__pick(average, *self.__shear()))
        self.__alone(average)
        n, m = _pair(normal, direction, "normal", "direction")
        modulus = 1 / (
            4
            * np.einsum(
                "...i,...j,...k,...l,ijkl->...", n, m, n, m, self.__compliance_tensor()
            )
        )
        return float(modulus) if n.ndim == 1 else modulus

    def young_modulus(
        self, direction: ArrayLike | None = None, *, average: str = "hill"
    ) -> float | np.ndarray:
        """The Young's modulus in GPa: of the crystal along a direction, or of a
        polycrystal.

        Parameters
        ----------
        direction : array_like, optional
            A direction (3,) float64 in the axes of the box, of any length, or
            directions (N, 3) float64. Without one, the modulus of a
            polycrystal.
        average : str, optional
            Of a polycrystal, the Young's modulus of the "voigt", "reuss" or
            "hill" average of the bulk and shear moduli. Default "hill".

        Returns
        -------
        float64 or numpy.ndarray
            The modulus, or (N,) float64 along each direction.

        Raises
        ------
        InvalidParameter
            For an unknown average, an average with a direction, or a
            direction of another shape, zero or not finite.
        """
        if direction is None:
            bulk = self.__pick(average, *self.__bulk())
            shear = self.__pick(average, *self.__shear())
            return float(9 * bulk * shear / (3 * bulk + shear))
        self.__alone(average)
        n = _directions(direction)
        modulus = 1 / np.einsum(
            "...i,...j,...k,...l,ijkl->...", n, n, n, n, self.__compliance_tensor()
        )
        return float(modulus) if n.ndim == 1 else modulus

    def poisson_ratio(
        self,
        direction: ArrayLike | None = None,
        transverse: ArrayLike | None = None,
        *,
        average: str = "hill",
    ) -> float | np.ndarray:
        """The Poisson ratio: of the crystal stretched along a direction, the
        contraction along a transverse direction over the stretch; or of a
        polycrystal.

        Parameters
        ----------
        direction : array_like, optional
            The direction (3,) float64 of the stretch, in the axes of the box
            and of any length, or directions (N, 3) float64.
        transverse : array_like, optional
            The direction (3,) or directions (N, 3) float64 of the contraction,
            perpendicular to the stretch. Give both directions, or neither for
            a polycrystal.
        average : str, optional
            Of a polycrystal, the Poisson ratio of the "voigt", "reuss" or
            "hill" average of the bulk and shear moduli. Default "hill".

        Returns
        -------
        float64 or numpy.ndarray
            The ratio, or (N,) float64 for each pair of directions.

        Raises
        ------
        InvalidParameter
            For an unknown average, an average with directions, one direction
            without the other, or directions of another shape, zero, not finite
            or not perpendicular.
        """
        if direction is None and transverse is None:
            bulk = self.__pick(average, *self.__bulk())
            shear = self.__pick(average, *self.__shear())
            return float((3 * bulk - 2 * shear) / (2 * (3 * bulk + shear)))
        self.__alone(average)
        n, m = _pair(direction, transverse, "direction", "transverse")
        s = self.__compliance_tensor()
        across = np.einsum("...i,...j,...k,...l,ijkl->...", n, n, m, m, s)
        along = np.einsum("...i,...j,...k,...l,ijkl->...", n, n, n, n, s)
        ratio = -across / along
        return float(ratio) if n.ndim == 1 else ratio

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
        shear_voigt, shear_reuss = self.__shear()
        bulk_voigt, bulk_reuss = self.__bulk()
        return float(5 * shear_voigt / shear_reuss + bulk_voigt / bulk_reuss - 6)

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
    errors : numpy.ndarray or None
        (3, 3) float64 standard errors of the constants in N/m, from molecular
        dynamics; None from a static calculation.
    """

    constants: np.ndarray
    errors: np.ndarray | None = None

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
        Convergence of each relaxation in eV/Å: the largest force left, or
        1e-4 of the largest force the strain makes, whichever is smaller,
        but no less than 1e-3 of max_force. Default 1e-4.
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
                # the stress follows the relaxation as the forces left over
                # those the strain makes: a fixed tolerance near those forces,
                # as at small strains, leaves the atoms half relaxed
                made = np.abs(calculator.compute(strained, ("forces",)).forces).max()
                tolerance = min(max_force, max(_RELATIVE * made, 1e-3 * max_force))
                outcome = minimize(
                    strained, calculator, max_force=tolerance, max_steps=max_steps
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
