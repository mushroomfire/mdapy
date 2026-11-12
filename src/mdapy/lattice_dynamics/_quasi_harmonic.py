# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The quasi-harmonic approximation: harmonic phonons at several volumes."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from mdapy import units
from mdapy.calculators._base import Calculator
from mdapy.core.box import Box
from mdapy.core.result import Result
from mdapy.core.system import System
from mdapy.errors import InvalidParameter
from mdapy.lattice_dynamics._phonons import (
    _crossing,
    _default_sheet_supercell,
    _default_supercell,
    phonons,
)
from mdapy.lattice_dynamics._sheets import _is_sheet
from mdapy.simulation._minimize import minimize


@dataclass(frozen=True)
class QuasiHarmonicResult(Result):
    """Thermodynamic properties of a crystal in the quasi-harmonic
    approximation, per atom.

    Attributes
    ----------
    temperatures : numpy.ndarray
        (T,) float64 temperatures in K.
    pressure : float64
        Pressure in GPa.
    volume : numpy.ndarray
        (T,) float64 equilibrium volume in Å³.
    gibbs_energy : numpy.ndarray
        (T,) float64 Gibbs energy in eV: the static energy, the free energy of the
        vibrations and the pressure times the volume.
    entropy : numpy.ndarray
        (T,) float64 entropy in eV/K.
    bulk_modulus : numpy.ndarray
        (T,) float64 isothermal bulk modulus in GPa.
    thermal_expansion : numpy.ndarray
        (T,) float64 volumetric thermal expansion coefficient in 1/K.
    heat_capacity : numpy.ndarray
        (T,) float64 heat capacity at constant pressure in eV/K.
    volumes : numpy.ndarray
        (V,) float64, the volumes in Å³ at which the phonons were computed.
    energies : numpy.ndarray
        (V,) float64 static energy at each volume in eV.
    free_energies : numpy.ndarray
        (V, T) float64 free energy of the vibrations at each volume and temperature
        in eV.
    cells : tuple of System
        The relaxed cell at each volume.
    """

    temperatures: np.ndarray
    pressure: float
    volume: np.ndarray
    gibbs_energy: np.ndarray
    entropy: np.ndarray
    bulk_modulus: np.ndarray
    thermal_expansion: np.ndarray
    heat_capacity: np.ndarray
    volumes: np.ndarray
    energies: np.ndarray
    free_energies: np.ndarray
    cells: tuple[System, ...] = ()


@dataclass(frozen=True)
class SheetQuasiHarmonicResult(Result):
    """Thermodynamic properties of a sheet in the quasi-harmonic
    approximation, per atom.

    Attributes
    ----------
    temperatures : numpy.ndarray
        (T,) float64 temperatures in K.
    pressure : float64
        Pressure in the plane in N/m, the opposite of a tension.
    area : numpy.ndarray
        (T,) float64 equilibrium area in Å².
    gibbs_energy : numpy.ndarray
        (T,) float64 Gibbs energy in eV: the static energy, the free energy of the
        vibrations and the pressure times the area.
    entropy : numpy.ndarray
        (T,) float64 entropy in eV/K.
    area_modulus : numpy.ndarray
        (T,) float64 isothermal area modulus in N/m.
    thermal_expansion : numpy.ndarray
        (T,) float64 thermal expansion coefficient of the area in 1/K.
    heat_capacity : numpy.ndarray
        (T,) float64 heat capacity at constant pressure in eV/K.
    areas : numpy.ndarray
        (A,) float64, the areas in Å² at which the phonons were computed.
    energies : numpy.ndarray
        (A,) float64 static energy at each area in eV.
    free_energies : numpy.ndarray
        (A, T) float64 free energy of the vibrations at each area and temperature
        in eV.
    cells : tuple of System
        The relaxed cell at each area.
    """

    temperatures: np.ndarray
    pressure: float
    area: np.ndarray
    gibbs_energy: np.ndarray
    entropy: np.ndarray
    area_modulus: np.ndarray
    thermal_expansion: np.ndarray
    heat_capacity: np.ndarray
    areas: np.ndarray
    energies: np.ndarray
    free_energies: np.ndarray
    cells: tuple[System, ...] = ()


class _Fit:
    """A least-squares cubic polynomial in V^p of values at the volumes: with
    p = -2/3 the third-order Birch-Murnaghan form, with p = -1 its form for
    the areas of a sheet."""

    def __init__(self, volumes: np.ndarray, values: np.ndarray, power: float = -2 / 3):
        self.p = power
        self.t = volumes**power
        self.c = np.polynomial.polynomial.polyfit(self.t, values, 3)

    def __call__(self, volume: float) -> float:
        return float(np.polynomial.polynomial.polyval(volume**self.p, self.c))

    def derivative(self, volume: float) -> float:
        """dy/dV."""
        t = volume**self.p
        dy = np.polynomial.polynomial.polyval(
            t, np.polynomial.polynomial.polyder(self.c)
        )
        return float(dy * self.p * volume ** (self.p - 1))

    def minimum(self) -> tuple[float, float] | None:
        """The volume of the minimum within the volumes, and the second
        derivative there; None when there is none."""
        slope = np.polynomial.polynomial.polyder(self.c)
        curvature = np.polynomial.polynomial.polyder(slope)
        for root in np.polynomial.polynomial.polyroots(slope):
            if abs(root.imag) > 1e-12 * abs(root.real):
                continue
            t = root.real
            if self.t.min() <= t <= self.t.max():
                second = np.polynomial.polynomial.polyval(t, curvature)
                if second > 0:
                    volume = t ** (1 / self.p)
                    # d2y/dV2 at a minimum, where dy/dt is zero
                    return volume, float(
                        second * (self.p * volume ** (self.p - 1)) ** 2
                    )
        return None


def _pressure(system: System, calculator: Calculator, sheet: bool) -> float:
    """The pressure of a system in GPa; of a sheet, of the mean of its two
    stresses in the plane."""
    stress = calculator.compute(system, ("virial",)).stress
    if sheet:
        return float(-(stress[0, 0] + stress[1, 1]) / 2)
    return float(-np.trace(stress) / 3)


def _size(system: System, sheet: bool) -> float:
    """The volume of the box in Å³; of a sheet, its area in Å²."""
    matrix = system.box.matrix
    if sheet:
        return float(np.linalg.norm(np.cross(matrix[0], matrix[1])))
    return system.box.volume


def _scaled(system: System, ratio: float, sheet: bool = False) -> System:
    """The system with its box and positions scaled to ratio times the
    volume; a sheet in its plane, to ratio times the area."""
    box = system.box
    origin = box.origin
    if sheet:
        scale = np.array([ratio**0.5, ratio**0.5, 1.0])
        matrix = box.matrix.copy()
        matrix[:2] *= ratio**0.5
    else:
        scale = np.full(3, ratio ** (1 / 3))
        matrix = box.matrix * scale[0]
    return system.with_box(Box(matrix, origin=origin, pbc=box.pbc)).with_columns(
        positions=origin + (system["positions"] - origin) * scale
    )


def _relaxed(system, calculator, pressure, max_force, max_stress, max_steps, where):
    relaxed = minimize(
        system,
        calculator,
        cell="full",
        pressure=pressure,
        max_force=max_force,
        max_stress=max_stress,
        max_steps=max_steps,
    )
    if not relaxed.converged:
        raise InvalidParameter(
            f"the crystal {where} did not relax within {max_steps} steps"
        )
    return relaxed


def _relaxed_cells(
    system, calculator, pressures, volumes, max_force, max_stress, max_steps, sheet
):
    """The relaxed cells of the grid: under each pressure, the box and the
    positions; or at each volume, the shape of the box and the positions;
    those of a sheet in its plane, with pressures in N/m and areas."""
    kind = "area" if sheet else "volume"
    if pressures is not None:
        pressures = np.sort(np.asarray(pressures, dtype=float).reshape(-1))[::-1]
        if len(pressures) < 5 or not np.isfinite(pressures).all():
            raise InvalidParameter("pressures must be five or more numbers")
        for p in pressures:
            where = f"under {p:.4g} {'N/m' if sheet else 'GPa'}"
            yield (
                where,
                _relaxed(
                    system,
                    calculator,
                    _in_gpa(float(p), system) if sheet else float(p),
                    max_force,
                    max_stress,
                    max_steps,
                    where,
                ),
            )
        return
    default = np.linspace(1.0, 1.04, 9) if sheet else np.linspace(0.97, 1.09, 9)
    ratios = (
        default
        if volumes is None
        else np.sort(np.asarray(volumes, dtype=float).reshape(-1))
    )
    if len(ratios) < 5 or not (ratios > 0).all():
        raise InvalidParameter(f"{kind}s must be five or more positive ratios")
    for ratio in ratios:
        # at a fixed volume: relaxed under the pressure of the box scaled to
        # it, scaled back to it, until it keeps its volume
        target = ratio * _size(system, sheet)
        where = f"at {kind} ratio {ratio:.4g}"
        current = _scaled(system, ratio, sheet)
        for _ in range(20):
            relaxed = _relaxed(
                current,
                calculator,
                _pressure(current, calculator, sheet),
                max_force,
                max_stress,
                max_steps,
                where,
            )
            size = _size(relaxed.system, sheet)
            if abs(size / target - 1) < 1e-6:
                break
            current = _scaled(relaxed.system, target / size, sheet)
        else:
            raise InvalidParameter(f"the crystal {where} did not keep its {kind}")
        yield where, relaxed


def _in_gpa(pressure: float, sheet: System) -> float:
    """A pressure in the plane of a sheet in N/m as a pressure in GPa over the
    height of its box."""
    return pressure / (units.GPA_A_TO_N_PER_M * sheet.box.matrix[2, 2])


def quasi_harmonic(
    system: System,
    calculator: Calculator,
    temperatures: ArrayLike,
    *,
    volumes: ArrayLike | None = None,
    pressures: ArrayLike | None = None,
    pressure: float = 0.0,
    supercell: int | ArrayLike | None = None,
    mesh: int | ArrayLike = 20,
    displacement: float = 0.01,
    max_force: float = 1e-6,
    max_stress: float = 1e-4,
    max_steps: int = 5000,
    symmetry: bool | None = None,
) -> QuasiHarmonicResult | SheetQuasiHarmonicResult:
    """Thermodynamic properties of a crystal in the quasi-harmonic
    approximation: harmonic phonons at several volumes; of a sheet, at
    several areas.

    Parameters
    ----------
    system : System
        The cell of the crystal, periodic in every direction, or of a sheet,
        periodic along two vectors in the x-y plane and not along a third
        along z; best relaxed and primitive. Each atom needs an element or a
        mass.
    calculator : Calculator
        Computes energies and forces.
    temperatures : array_like of float64
        Temperatures in K, not negative.
    volumes : array_like of float64, optional
        The volumes at which to compute the phonons, five or more, as ratios
        to the volume of system; at each the box keeps its volume while its
        shape and the positions relax. Default 9 from 0.97 to 1.09. Of a
        sheet, the areas, as ratios to its area; default 9 from 1 to 1.04, as
        a sheet compressed in its plane bends.
    pressures : array_like of float64, optional
        Instead of volumes, the pressures in GPa under which the box and the
        positions relax, five or more; of a sheet, in its plane, in N/m.
    pressure : float64, optional
        Pressure in GPa; of a sheet, in its plane in N/m, the opposite of a
        tension. Default 0.
    supercell : int32 or array_like, optional
        Repeats of the cell for the phonons, as in ``phonons``.
    mesh : int32 or array_like, optional
        Mesh of wave vectors for the thermal properties. Default 20.
    displacement : float64, optional
        Displacement of the atoms for the force constants in Å. Default 0.01.
    max_force : float64, optional
        Convergence of the relaxation at each volume in eV/Å. Default 1e-6.
    max_stress : float64, optional
        Convergence of the relaxation of the box under each pressure in
        GPa. Default 1e-4.
    max_steps : int32, optional
        Largest number of steps of each relaxation. Default 5000.
    symmetry : bool, optional
        Use the symmetry of the cells in the phonons, as in
        ``phonons``. Default whenever spglib is installed.

    Returns
    -------
    QuasiHarmonicResult or SheetQuasiHarmonicResult
        Per atom: the equilibrium volume, Gibbs energy, entropy, bulk
        modulus, thermal expansion and heat capacity at each temperature,
        and the static and vibrational energies at each volume; of a sheet,
        the area, the area modulus and the thermal expansion of the area.

    Raises
    ------
    InvalidParameter
        For a box neither periodic in every direction nor that of a sheet, a
        negative temperature, both volumes and pressures, fewer than five of
        them, a relaxation that does not converge, a pair of atoms that
        reaches the cutoff of the calculator between the volumes, imaginary
        frequencies at a volume, or an equilibrium volume outside the
        volumes.
    """
    sheet = _is_sheet(system.box, "quasi-harmonic properties")
    temperatures = np.atleast_1d(np.asarray(temperatures, dtype=float))
    if temperatures.ndim != 1 or not (temperatures >= 0).all():
        raise InvalidParameter("temperatures must not be negative")
    if pressures is not None and volumes is not None:
        raise InvalidParameter("give pressures or volumes, not both")
    n = len(system)
    if supercell is None:  # the same supercell at every volume
        default = _default_sheet_supercell if sheet else _default_supercell
        supercell = default(system.box.matrix, 2 * calculator.cutoff)
    grid = list(
        _relaxed_cells(
            system,
            calculator,
            pressures,
            volumes,
            max_force,
            max_stress,
            max_steps,
            sheet,
        )
    )
    # the force constants jump where a pair of atoms crosses the cutoff
    if crossing := _crossing(
        [r.system for _, r in grid], calculator.cutoff, displacement
    ):
        short, near, long, far = crossing
        kind = "areas" if sheet else "volumes"
        raise InvalidParameter(
            f"a pair of atoms reaches the cutoff of the calculator, "
            f"{calculator.cutoff:.4g} Å, between the {kind}, give or take the "
            f"displacement of {displacement:.4g} Å: it is {short:.4f} Å apart "
            f"{grid[near][0]} and {long:.4f} Å {grid[far][0]}, and its force "
            f"constants jump there; use a smaller range of {kind} or pressures, "
            "or another potential"
        )
    cells, energies, free, entropy, capacity = [], [], [], [], []
    for where, relaxed in grid:
        try:
            thermal = phonons(
                relaxed.system,
                calculator,
                supercell=supercell,
                displacement=displacement,
                symmetry=symmetry,
            ).thermal(temperatures, mesh=mesh)
        except InvalidParameter as error:
            hint = (
                "; a sheet compressed in its plane bends, its vibrations across "
                "the plane imaginary: use areas no smaller than that of the "
                "relaxed sheet, or a tension"
                if sheet and "imaginary" in str(error)
                else ""
            )
            raise InvalidParameter(f"{where}: {error}{hint}") from None
        cells.append(relaxed.system)
        energies.append(relaxed.energy / n)
        free.append(thermal.free_energy)
        entropy.append(thermal.entropy)
        capacity.append(thermal.heat_capacity)
    order = np.argsort([_size(c, sheet) for c in cells])
    cells = tuple(cells[k] for k in order)
    per_atom = np.array([_size(c, sheet) / n for c in cells])
    energies = np.array(energies)[order]
    free, entropy, capacity = (np.array(x)[order] for x in (free, entropy, capacity))
    if sheet:  # N/m to eV/Å², and a fit in 1/A
        work = pressure / units.EV_PER_A2_TO_J_PER_M2 * per_atom
        power, kind = -1.0, "area"
    else:
        work = pressure * units.GPA_TO_EV_PER_A3 * per_atom
        power, kind = -2 / 3, "volume"
    results = []
    for k, t in enumerate(temperatures):
        fit = _Fit(per_atom, energies + free[:, k] + work, power)
        found = fit.minimum()
        if found is None:
            raise InvalidParameter(
                f"the equilibrium {kind} at {t:g} K is outside the {kind}s; widen them"
            )
        volume, curvature = found
        bulk = volume * curvature  # eV/Å³, of a sheet eV/Å²
        entropy_fit = _Fit(per_atom, entropy[:, k], power)
        # α B = (∂S/∂V) at constant T
        expansion = entropy_fit.derivative(volume) / bulk
        results.append(
            (
                volume,
                fit(volume),
                entropy_fit(volume),
                bulk
                * (units.EV_PER_A2_TO_J_PER_M2 if sheet else units.EV_PER_A3_TO_GPA),
                expansion,
                _Fit(per_atom, capacity[:, k], power)(volume)
                + t * volume * bulk * expansion**2,
            )
        )
    volume, gibbs, s, bulk, expansion, heat = (np.array(c) for c in zip(*results))
    if sheet:
        return SheetQuasiHarmonicResult(
            temperatures=temperatures,
            pressure=float(pressure),
            area=volume,
            gibbs_energy=gibbs,
            entropy=s,
            area_modulus=bulk,
            thermal_expansion=expansion,
            heat_capacity=heat,
            areas=per_atom,
            energies=energies,
            free_energies=free,
            cells=cells,
        )
    return QuasiHarmonicResult(
        temperatures=temperatures,
        pressure=float(pressure),
        volume=volume,
        gibbs_energy=gibbs,
        entropy=s,
        bulk_modulus=bulk,
        thermal_expansion=expansion,
        heat_capacity=heat,
        volumes=per_atom,
        energies=energies,
        free_energies=free,
        cells=cells,
    )
