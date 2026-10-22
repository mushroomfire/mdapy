# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Energies of defects: vacancies, surfaces and generalized stacking faults."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from mdapy import units
from mdapy.build._defects import _axis, add_vacancies, add_vacuum, slip
from mdapy.calculators import Calculator
from mdapy.core.result import Result
from mdapy.core.system import System
from mdapy.errors import InvalidParameter
from mdapy.simulation._minimize import minimize


@dataclass(frozen=True)
class VacancyResult(Result):
    """The formation energy of a vacancy.

    Attributes
    ----------
    energy : float
        The formation energy in eV.
    perfect : float
        The energy of the perfect crystal in eV.
    defect : float
        The energy of the crystal with the vacancy in eV.
    system : System
        The crystal with the vacancy, relaxed.
    """

    energy: float
    perfect: float
    defect: float
    system: System


@dataclass(frozen=True)
class SurfaceResult(Result):
    """The energy of a surface.

    Attributes
    ----------
    energy : float
        The surface energy in J/m².
    bulk : float
        The energy of the periodic crystal in eV.
    slab : float
        The energy of the slab in eV.
    area : float
        The area of one surface in Å².
    system : System
        The slab, relaxed.
    """

    energy: float
    bulk: float
    slab: float
    area: float
    system: System


@dataclass(frozen=True)
class StackingFaultResult(Result):
    """A generalized stacking fault energy curve.

    Attributes
    ----------
    fraction : numpy.ndarray
        (P,) fraction of the slip vector of each point.
    displacement : numpy.ndarray
        (P, 3) displacement in Å of the upper half of each point.
    energy : numpy.ndarray
        (P,) energy per area of the fault in J/m², 0 for no slip.
    area : float
        The area of the fault in Å².
    """

    fraction: np.ndarray
    displacement: np.ndarray
    energy: np.ndarray
    area: float


def _relaxed(
    system: System, calculator: Calculator, options: dict[str, Any]
) -> tuple[float, System]:
    result = minimize(system, calculator, **options)
    if not result.converged:
        raise InvalidParameter(
            f"the minimization did not converge in {result.steps} steps "
            f"(largest force {result.max_force:.3g} eV/Å); raise max_steps"
        )
    return result.energy, result.system


def _options(
    max_force: float, max_steps: int, cell: str | None = None
) -> dict[str, Any]:
    options: dict[str, Any] = {"max_force": max_force, "max_steps": max_steps}
    if cell is not None:
        options["cell"] = cell
    return options


def vacancy_formation_energy(
    crystal: System,
    calculator: Calculator,
    *,
    atom: int = 0,
    cell: str | None = None,
    max_force: float = 1e-4,
    max_steps: int = 10000,
) -> VacancyResult:
    """The energy to form a vacancy in a perfect crystal.

    Parameters
    ----------
    crystal : System
        The perfect crystal, periodic, large enough that the vacancy does not
        meet its images.
    calculator : Calculator
        The potential.
    atom : int, optional
        The atom removed. Default 0.
    cell : {None, "isotropic", "axes", "full"}, optional
        How the box relaxes, for both crystals, as in
        :func:`mdapy.simulation.minimize`. Default None, a fixed box.
    max_force : float, optional
        Convergence of the minimizations in eV/Å. Default 1e-4.
    max_steps : int, optional
        Largest number of steps of each minimization. Default 10000.

    Returns
    -------
    VacancyResult
        The formation energy, the two energies and the relaxed defect.
    """
    n = len(crystal)
    if n < 2:
        raise InvalidParameter("the crystal needs two atoms or more")
    options = _options(max_force, max_steps, cell)
    perfect, _ = _relaxed(crystal, calculator, options)
    defect, system = _relaxed(add_vacancies(crystal, atoms=[atom]), calculator, options)
    return VacancyResult(
        energy=defect - (n - 1) / n * perfect,
        perfect=perfect,
        defect=defect,
        system=system,
    )


def surface_energy(
    crystal: System,
    calculator: Calculator,
    *,
    axis: int | str = 2,
    vacuum: float = 15.0,
    max_force: float = 1e-4,
    max_steps: int = 10000,
) -> SurfaceResult:
    """The energy of the surface of a crystal normal to a box vector.

    Parameters
    ----------
    crystal : System
        The periodic crystal; the slab is this crystal between two surfaces
        normal to the box vector of axis, thick enough not to feel each
        other.
    calculator : Calculator
        The potential.
    axis : int or str, optional
        The box vector across the slab, 0, 1, 2 or x, y, z. Default z.
    vacuum : float, optional
        Thickness in Å of the vacuum between the slab and its image, beyond
        the cutoff of the potential. Default 15.
    max_force : float, optional
        Convergence of the minimizations in eV/Å. Default 1e-4.
    max_steps : int, optional
        Largest number of steps of each minimization. Default 10000.

    Returns
    -------
    SurfaceResult
        The surface energy, the two energies, the area and the relaxed slab.
    """
    k = _axis(axis)
    if not crystal.box.pbc.all():
        raise InvalidParameter("the crystal must be periodic in every direction")
    if not vacuum > 0:
        raise InvalidParameter(f"vacuum must be positive, not {vacuum!r}")
    options = _options(max_force, max_steps)
    bulk, _ = _relaxed(crystal, calculator, options)
    slab, system = _relaxed(add_vacuum(crystal, vacuum, axis=k), calculator, options)
    others = [i for i in range(3) if i != k]
    area = float(
        np.linalg.norm(
            np.cross(crystal.box.matrix[others[0]], crystal.box.matrix[others[1]])
        )
    )
    energy = (slab - bulk) / (2 * area) * units.EV_PER_A2_TO_J_PER_M2
    return SurfaceResult(energy=energy, bulk=bulk, slab=slab, area=area, system=system)


def stacking_fault_energy(
    crystal: System,
    calculator: Calculator,
    direction: ArrayLike,
    *,
    points: int = 21,
    height: float | None = None,
    axis: int | str = 2,
    relax: bool = True,
    max_force: float = 1e-4,
    max_steps: int = 10000,
) -> StackingFaultResult:
    """The generalized stacking fault energy along a slip vector.

    Parameters
    ----------
    crystal : System
        The crystal, with the plane of the fault normal to the Cartesian axis
        of axis, such as fcc with z along [111]; periodic along that axis for
        a single fault, else between two free surfaces.
    calculator : Calculator
        The potential.
    direction : array_like
        (3,) the slip vector in Å, in the plane, such as a/2 [1-10].
    points : int, optional
        Number of displacements from 0 to the whole slip vector, at least 2.
        Default 21.
    height : float, optional
        The fault, at this fractional coordinate along the box vector of axis.
        Default 0.5.
    axis : int or str, optional
        The Cartesian axis normal to the fault. Default z.
    relax : bool, optional
        Relax the atoms along the normal only, as is usual, or not at all.
        Default True.
    max_force : float, optional
        Convergence of the minimizations in eV/Å. Default 1e-4.
    max_steps : int, optional
        Largest number of steps of each minimization. Default 10000.

    Returns
    -------
    StackingFaultResult
        The energy per area for each fraction of the slip vector.
    """
    k = _axis(axis)
    others = [i for i in range(3) if i != k]
    normal = np.cross(crystal.box.matrix[others[0]], crystal.box.matrix[others[1]])
    if np.linalg.norm(np.delete(normal, k)) > 1e-9 * np.linalg.norm(normal):
        raise InvalidParameter(
            "the plane of the fault must be normal to the Cartesian axis"
        )
    if int(points) < 2:
        raise InvalidParameter(f"points must be at least 2, not {points!r}")
    vector = np.asarray(direction, dtype=float)
    fraction = np.linspace(0.0, 1.0, int(points))
    movable = np.zeros((len(crystal), 3), dtype=bool)
    movable[:, k] = True
    options = _options(max_force, max_steps)
    area = float(np.linalg.norm(normal))
    energies = []
    for f in fraction:
        faulted = slip(crystal, f * vector, height=height, axis=k)
        if relax:
            energy, _ = _relaxed(
                faulted.with_columns(movable=movable), calculator, options
            )
        else:
            energy = calculator.compute(faulted, ("energy",)).energy
        energies.append(energy)
    energy = (np.array(energies) - energies[0]) / area * units.EV_PER_A2_TO_J_PER_M2
    return StackingFaultResult(
        fraction=fraction,
        displacement=fraction[:, None] * vector,
        energy=energy,
        area=area,
    )
