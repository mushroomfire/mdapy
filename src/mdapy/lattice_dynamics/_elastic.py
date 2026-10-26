# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Elastic constants from the stress under small strains."""

from dataclasses import dataclass

import numpy as np

from mdapy.calculators._base import Calculator
from mdapy.core.box import Box
from mdapy.core.system import System
from mdapy.errors import InvalidParameter
from mdapy.simulation._minimize import minimize

# Voigt order xx, yy, zz, yz, xz, xy
_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def _voigt(tensor: np.ndarray) -> np.ndarray:
    return np.array([tensor[a, b] for a, b in _PAIRS])


@dataclass(frozen=True)
class ElasticResult:
    """Elastic constants of a crystal.

    Attributes
    ----------
    constants : numpy.ndarray
        (6, 6) elastic constants in GPa, in Voigt order xx, yy, zz, yz, xz,
        xy; column k is the stress per unit of strain component k.
    errors : numpy.ndarray or None
        (6, 6) standard errors of the constants in GPa, from molecular
        dynamics; None otherwise.
    """

    constants: np.ndarray
    errors: np.ndarray | None = None

    def __post_init__(self):
        self.constants.flags.writeable = False
        if self.errors is not None:
            self.errors.flags.writeable = False

    def bulk_modulus(self, average: str = "hill") -> float:
        """The bulk modulus in GPa: the "voigt", "reuss" or "hill" average."""
        c, s = self.constants, np.linalg.inv(self.constants)
        voigt = (c[0, 0] + c[1, 1] + c[2, 2] + 2 * (c[0, 1] + c[0, 2] + c[1, 2])) / 9
        reuss = 1 / (s[0, 0] + s[1, 1] + s[2, 2] + 2 * (s[0, 1] + s[0, 2] + s[1, 2]))
        return float(self.__pick(average, voigt, reuss))

    def shear_modulus(self, average: str = "hill") -> float:
        """The shear modulus in GPa: the "voigt", "reuss" or "hill" average."""
        c, s = self.constants, np.linalg.inv(self.constants)
        voigt = (
            c[0, 0] + c[1, 1] + c[2, 2] - (c[0, 1] + c[0, 2] + c[1, 2])
            + 3 * (c[3, 3] + c[4, 4] + c[5, 5])
        ) / 15  # fmt: skip
        reuss = 15 / (
            4 * (s[0, 0] + s[1, 1] + s[2, 2]) - 4 * (s[0, 1] + s[0, 2] + s[1, 2])
            + 3 * (s[3, 3] + s[4, 4] + s[5, 5])
        )  # fmt: skip
        return float(self.__pick(average, voigt, reuss))

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


def _strained(system: System, strain: np.ndarray) -> System:
    """The system and its box under a homogeneous strain."""
    deformation = np.eye(3) + strain
    origin = system.box.origin
    box = Box(system.box.matrix @ deformation.T, origin=origin, pbc=system.box.pbc)
    return system.with_box(box).with_columns(
        positions=origin + (system["positions"] - origin) @ deformation.T
    )


def elastic_constants(
    system: System,
    calculator: Calculator,
    *,
    strain: float = 1e-3,
    relax: bool = True,
    max_force: float = 1e-4,
    max_steps: int = 5000,
) -> ElasticResult:
    """Elastic constants from the stress under small strains.

    Parameters
    ----------
    system : System
        The crystal, periodic in every direction, best at zero stress (relax
        its box first with ``minimize(..., cell="full")``).
    calculator : Calculator
        Computes the virial, and energies and forces to relax the positions.
    strain : float, optional
        Magnitude of the strains applied in each direction, positive and
        negative; shear strains are engineering strains. Default 1e-3.
    relax : bool, optional
        Relax the positions under each strain. Default True.
    max_force : float, optional
        Convergence of each relaxation in eV/Å. Default 1e-4.
    max_steps : int, optional
        Largest number of steps of each relaxation. Default 5000.

    Returns
    -------
    ElasticResult
        The (6, 6) elastic constants in GPa.

    Raises
    ------
    InvalidParameter
        If the box is not periodic, the strain is not positive, the
        calculator has no virial, or a relaxation does not converge.
    """
    if not system.box.pbc.all():
        raise InvalidParameter(
            "elastic constants need a box periodic in every direction"
        )
    if not strain > 0:
        raise InvalidParameter(f"strain must be positive, not {strain!r}")
    needed = {"virial"} | ({"energy", "forces"} if relax else set())
    if not needed <= calculator.implemented:
        raise InvalidParameter(
            f"{type(calculator).__name__} does not compute {sorted(needed)}"
        )
    constants = np.zeros((6, 6))
    for k, (a, b) in enumerate(_PAIRS):
        stresses = []
        for sign in (1.0, -1.0):
            tensor = np.zeros((3, 3))
            if a == b:
                tensor[a, a] = sign * strain
            else:  # engineering shear strain: half on each off-diagonal entry
                tensor[a, b] = tensor[b, a] = 0.5 * sign * strain
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
                stress = calculator.compute(strained, ("virial",)).stress(strained)
            stresses.append(_voigt(stress))
        constants[:, k] = (stresses[0] - stresses[1]) / (2 * strain)
    return ElasticResult(constants)
