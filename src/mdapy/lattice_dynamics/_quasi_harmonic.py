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
from mdapy.lattice_dynamics._phonons import _read_only, phonons
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


def quasi_harmonic(
    system: System,
    calculator: Calculator,
    temperatures: ArrayLike,
    *,
    volumes: ArrayLike | None = None,
    pressure: float = 0.0,
    supercell: int | ArrayLike | None = None,
    mesh: int | ArrayLike = 20,
    displacement: float = 0.01,
    max_force: float = 1e-6,
    max_steps: int = 5000,
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
        The volumes at which to compute the phonons, as ratios to the volume
        of system; the box is scaled alike in every direction and the
        positions relax. Default 9 from 0.97 to 1.09.
    pressure : float, optional
        Pressure in GPa. Default 0.
    supercell : int or array_like, optional
        Repeats of the cell for the phonons, as in :func:`phonons`.
    mesh : int or array_like, optional
        Mesh of wave vectors for the thermal properties. Default 20.
    displacement : float, optional
        Displacement of the atoms for the force constants in Å. Default 0.01.
    max_force : float, optional
        Convergence of the relaxation of the positions at each volume in
        eV/Å. Default 1e-6.
    max_steps : int, optional
        Largest number of steps of each relaxation. Default 5000.

    Returns
    -------
    QuasiHarmonicResult
        Per atom: the equilibrium volume, Gibbs energy, entropy, bulk
        modulus, thermal expansion and heat capacity at each temperature,
        and the static and vibrational energies at each volume.

    Raises
    ------
    InvalidParameter
        For a negative temperature, fewer than five volumes, a relaxation
        that does not converge, imaginary frequencies at a volume, or an
        equilibrium volume outside the volumes.
    """
    temperatures = np.atleast_1d(np.asarray(temperatures, dtype=float))
    if temperatures.ndim != 1 or not (temperatures >= 0).all():
        raise InvalidParameter("temperatures must not be negative")
    ratios = (
        np.linspace(0.97, 1.09, 9)
        if volumes is None
        else np.sort(np.asarray(volumes, dtype=float).reshape(-1))
    )
    if len(ratios) < 5 or not (ratios > 0).all():
        raise InvalidParameter("volumes must be five or more positive ratios")
    n = len(system)
    box = system.box
    energies, free, entropy, capacity = [], [], [], []
    for ratio in ratios:
        scale = ratio ** (1 / 3)
        origin = box.origin
        scaled = system.with_box(
            Box(box.matrix * scale, origin=origin, pbc=box.pbc)
        ).with_columns(positions=origin + (system["positions"] - origin) * scale)
        relaxed = minimize(scaled, calculator, max_force=max_force, max_steps=max_steps)
        if not relaxed.converged:
            raise InvalidParameter(
                f"the positions at volume ratio {ratio:.4g} did not relax within "
                f"{max_steps} steps"
            )
        try:
            thermal = phonons(
                relaxed.system,
                calculator,
                supercell=supercell,
                displacement=displacement,
            ).thermal(temperatures, mesh)
        except InvalidParameter as error:
            raise InvalidParameter(f"at volume ratio {ratio:.4g}: {error}") from None
        energies.append(relaxed.energy / n)
        free.append(thermal.free_energy)
        entropy.append(thermal.entropy)
        capacity.append(thermal.heat_capacity)
    per_atom = ratios * box.volume / n
    energies = np.array(energies)
    free, entropy, capacity = np.array(free), np.array(entropy), np.array(capacity)
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
    )
