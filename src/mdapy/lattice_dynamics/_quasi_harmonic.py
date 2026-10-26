# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""The quasi-harmonic approximation: harmonic phonons at several volumes."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from mdapy import units
from mdapy.calculators._base import Calculator
from mdapy.core.box import Box
from mdapy.core.system import System
from mdapy.errors import InvalidParameter
from mdapy.lattice_dynamics._phonons import _default_supercell, _read_only, phonons
from mdapy.simulation._minimize import minimize


@dataclass(frozen=True)
class QuasiHarmonicResult:
    """Thermodynamic properties of a crystal in the quasi-harmonic
    approximation, per atom.

    Attributes
    ----------
    temperatures : numpy.ndarray
        (T,) temperatures in K.
    pressure : float
        Pressure in GPa.
    volume : numpy.ndarray
        (T,) equilibrium volume in Å³.
    gibbs_energy : numpy.ndarray
        (T,) Gibbs energy in eV: the static energy, the free energy of the
        vibrations and the pressure times the volume.
    entropy : numpy.ndarray
        (T,) entropy in eV/K.
    bulk_modulus : numpy.ndarray
        (T,) isothermal bulk modulus in GPa.
    thermal_expansion : numpy.ndarray
        (T,) volumetric thermal expansion coefficient in 1/K.
    heat_capacity : numpy.ndarray
        (T,) heat capacity at constant pressure in eV/K.
    volumes : numpy.ndarray
        (V,) the volumes in Å³ at which the phonons were computed.
    energies : numpy.ndarray
        (V,) static energy at each volume in eV.
    free_energies : numpy.ndarray
        (V, T) free energy of the vibrations at each volume and temperature
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

    def __post_init__(self):
        _read_only(
            self.temperatures,
            self.volume,
            self.gibbs_energy,
            self.entropy,
            self.bulk_modulus,
            self.thermal_expansion,
            self.heat_capacity,
            self.volumes,
            self.energies,
            self.free_energies,
        )


class _Fit:
    """A least-squares cubic polynomial in V^(-2/3), the third-order
    Birch-Murnaghan form, of values at the volumes."""

    def __init__(self, volumes: np.ndarray, values: np.ndarray):
        self.t = volumes ** (-2 / 3)
        self.c = np.polynomial.polynomial.polyfit(self.t, values, 3)

    def __call__(self, volume: float) -> float:
        return float(np.polynomial.polynomial.polyval(volume ** (-2 / 3), self.c))

    def derivative(self, volume: float) -> float:
        """dy/dV."""
        t = volume ** (-2 / 3)
        dy = np.polynomial.polynomial.polyval(
            t, np.polynomial.polynomial.polyder(self.c)
        )
        return float(dy * (-2 / 3) * volume ** (-5 / 3))

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
                    volume = t ** (-3 / 2)
                    return volume, float(second * (4 / 9) * volume ** (-10 / 3))
        return None


def _pressure(system: System, calculator: Calculator) -> float:
    """The pressure of a system in GPa."""
    return float(-np.trace(calculator.compute(system, ("virial",)).stress(system)) / 3)


def _scaled(system: System, ratio: float) -> System:
    """The system with its box and positions scaled to ratio times the
    volume."""
    scale = ratio ** (1 / 3)
    box = system.box
    origin = box.origin
    return system.with_box(
        Box(box.matrix * scale, origin=origin, pbc=box.pbc)
    ).with_columns(positions=origin + (system["positions"] - origin) * scale)


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
    system, calculator, pressures, volumes, max_force, max_stress, max_steps
):
    """The relaxed cells of the grid: under each pressure, the box and the
    positions; or at each volume, the shape of the box and the positions."""
    if pressures is not None:
        pressures = np.sort(np.asarray(pressures, dtype=float).reshape(-1))[::-1]
        if len(pressures) < 5 or not np.isfinite(pressures).all():
            raise InvalidParameter("pressures must be five or more numbers")
        for p in pressures:
            where = f"under {p:.4g} GPa"
            yield (
                where,
                _relaxed(
                    system,
                    calculator,
                    float(p),
                    max_force,
                    max_stress,
                    max_steps,
                    where,
                ),
            )
        return
    ratios = (
        np.linspace(0.97, 1.09, 9)
        if volumes is None
        else np.sort(np.asarray(volumes, dtype=float).reshape(-1))
    )
    if len(ratios) < 5 or not (ratios > 0).all():
        raise InvalidParameter("volumes must be five or more positive ratios")
    for ratio in ratios:
        # at a fixed volume: relaxed under the pressure of the box scaled to
        # it, scaled back to it, until it keeps its volume
        target = ratio * system.box.volume
        where = f"at volume ratio {ratio:.4g}"
        current = _scaled(system, ratio)
        for _ in range(20):
            relaxed = _relaxed(
                current,
                calculator,
                _pressure(current, calculator),
                max_force,
                max_stress,
                max_steps,
                where,
            )
            if abs(relaxed.system.box.volume / target - 1) < 1e-6:
                break
            current = _scaled(relaxed.system, target / relaxed.system.box.volume)
        else:
            raise InvalidParameter(f"the crystal {where} did not keep its volume")
        yield where, relaxed


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
) -> QuasiHarmonicResult:
    """Thermodynamic properties of a crystal in the quasi-harmonic
    approximation: harmonic phonons at several volumes.

    Parameters
    ----------
    system : System
        The cell of the crystal, periodic in every direction, best relaxed
        and primitive. Each atom needs an element or a mass.
    calculator : Calculator
        Computes energies and forces.
    temperatures : array_like
        Temperatures in K, not negative.
    volumes : array_like, optional
        The volumes at which to compute the phonons, five or more, as ratios
        to the volume of system; at each the box keeps its volume while its
        shape and the positions relax. Default 9 from 0.97 to 1.09.
    pressures : array_like, optional
        Instead of volumes, the pressures in GPa under which the box and the
        positions relax, five or more.
    pressure : float, optional
        Pressure in GPa. Default 0.
    supercell : int or array_like, optional
        Repeats of the cell for the phonons, as in :func:`phonons`.
    mesh : int or array_like, optional
        Mesh of wave vectors for the thermal properties. Default 20.
    displacement : float, optional
        Displacement of the atoms for the force constants in Å. Default 0.01.
    max_force : float, optional
        Convergence of the relaxation at each volume in eV/Å. Default 1e-6.
    max_stress : float, optional
        Convergence of the relaxation of the box under each pressure in
        GPa. Default 1e-4.
    max_steps : int, optional
        Largest number of steps of each relaxation. Default 5000.
    symmetry : bool, optional
        Use the symmetry of the cells in the phonons, as in
        :func:`phonons`. Default whenever spglib is installed.

    Returns
    -------
    QuasiHarmonicResult
        Per atom: the equilibrium volume, Gibbs energy, entropy, bulk
        modulus, thermal expansion and heat capacity at each temperature,
        and the static and vibrational energies at each volume.

    Raises
    ------
    InvalidParameter
        For a negative temperature, both volumes and pressures, fewer than
        five of them, a relaxation that does not converge, imaginary
        frequencies at a volume, or an equilibrium volume outside the
        volumes.
    """
    temperatures = np.atleast_1d(np.asarray(temperatures, dtype=float))
    if temperatures.ndim != 1 or not (temperatures >= 0).all():
        raise InvalidParameter("temperatures must not be negative")
    if pressures is not None and volumes is not None:
        raise InvalidParameter("give pressures or volumes, not both")
    n = len(system)
    if supercell is None:  # the same supercell at every volume
        supercell = _default_supercell(system.box.matrix, 2 * calculator.cutoff)
    cells, energies, free, entropy, capacity = [], [], [], [], []
    for where, relaxed in _relaxed_cells(
        system, calculator, pressures, volumes, max_force, max_stress, max_steps
    ):
        try:
            thermal = phonons(
                relaxed.system,
                calculator,
                supercell=supercell,
                displacement=displacement,
                symmetry=symmetry,
            ).thermal(temperatures, mesh)
        except InvalidParameter as error:
            raise InvalidParameter(f"{where}: {error}") from None
        cells.append(relaxed.system)
        energies.append(relaxed.energy / n)
        free.append(thermal.free_energy)
        entropy.append(thermal.entropy)
        capacity.append(thermal.heat_capacity)
    order = np.argsort([c.box.volume for c in cells])
    cells = tuple(cells[k] for k in order)
    per_atom = np.array([c.box.volume / n for c in cells])
    energies = np.array(energies)[order]
    free, entropy, capacity = (np.array(x)[order] for x in (free, entropy, capacity))
    work = pressure * units.GPA_TO_EV_PER_A3 * per_atom
    results = []
    for k, t in enumerate(temperatures):
        fit = _Fit(per_atom, energies + free[:, k] + work)
        found = fit.minimum()
        if found is None:
            raise InvalidParameter(
                f"the equilibrium volume at {t:g} K is outside the volumes; widen them"
            )
        volume, curvature = found
        bulk = volume * curvature  # eV/Å³
        entropy_fit = _Fit(per_atom, entropy[:, k])
        # α B = (∂S/∂V) at constant T
        expansion = entropy_fit.derivative(volume) / bulk
        results.append(
            (
                volume,
                fit(volume),
                entropy_fit(volume),
                bulk * units.EV_PER_A3_TO_GPA,
                expansion,
                _Fit(per_atom, capacity[:, k])(volume)
                + t * volume * bulk * expansion**2,
            )
        )
    volume, gibbs, s, bulk, expansion, heat = (np.array(c) for c in zip(*results))
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
