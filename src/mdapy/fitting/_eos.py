# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Equations of state fitted to energies at volumes."""

import numpy as np
from numpy.typing import ArrayLike

from mdapy import units
from mdapy.errors import InvalidParameter
from mdapy.fitting._least_squares import Model, _numbers, fit_curve, fit_polynomial


def _birch_murnaghan(v, e0, v0, b0, b0_prime):
    eta = (v0 / v) ** (2 / 3)
    return e0 + 9 * v0 * b0 / 16 * (
        (eta - 1) ** 3 * b0_prime + (eta - 1) ** 2 * (6 - 4 * eta)
    )


def _birch_murnaghan_pressure(v, v0, b0, b0_prime):
    eta = (v0 / v) ** (2 / 3)
    return 1.5 * b0 * (eta**3.5 - eta**2.5) * (1 + 0.75 * (b0_prime - 4) * (eta - 1))


def _vinet(v, e0, v0, b0, b0_prime):
    x = (v / v0) ** (1 / 3)
    eta = 1.5 * (b0_prime - 1)
    return e0 + 2 * b0 * v0 / (b0_prime - 1) ** 2 * (
        2 - (5 + 3 * b0_prime * (x - 1) - 3 * x) * np.exp(-eta * (x - 1))
    )


def _vinet_pressure(v, v0, b0, b0_prime):
    x = (v / v0) ** (1 / 3)
    return 3 * b0 * (1 - x) / x**2 * np.exp(1.5 * (b0_prime - 1) * (1 - x))


def _murnaghan(v, e0, v0, b0, b0_prime):
    return (
        e0
        + b0 * v / b0_prime * ((v0 / v) ** b0_prime / (b0_prime - 1) + 1)
        - v0 * b0 / (b0_prime - 1)
    )


def _murnaghan_pressure(v, v0, b0, b0_prime):
    return b0 / b0_prime * ((v0 / v) ** b0_prime - 1)


_NAMES = ("e0", "v0", "b0", "b0_prime")
_FORMS = {
    "birch_murnaghan": (
        Model("Birch-Murnaghan", _birch_murnaghan, _NAMES),
        _birch_murnaghan_pressure,
    ),
    "vinet": (Model("Vinet", _vinet, _NAMES), _vinet_pressure),
    "murnaghan": (Model("Murnaghan", _murnaghan, _NAMES), _murnaghan_pressure),
}
FORMS = tuple(_FORMS)


class EquationOfState:
    """An equation of state fitted to energies at volumes, made by
    ``fit_eos``.

    Attributes
    ----------
    form : str
        One of ``FORMS``.
    e0 : float64
        The energy at the minimum in eV.
    v0 : float64
        The volume at the minimum in Å³.
    b0 : float64
        The bulk modulus at the minimum in GPa.
    b0_prime : float64
        Its derivative with respect to the pressure.
    errors : numpy.ndarray
        (4,) float64 standard errors of e0, v0, b0 (in GPa) and b0_prime.
    residuals : numpy.ndarray
        (n,) float64 the energies less the equation of state, in eV.
    r_squared : float64
        The coefficient of determination of the energies.
    """

    def __init__(self, form, parameters, errors, residuals, r_squared):
        self.form = form
        self.e0, self.v0, b0, self.b0_prime = (float(p) for p in parameters)
        self.b0 = b0 * units.EV_PER_A3_TO_GPA
        self.errors = errors * np.array([1.0, 1.0, units.EV_PER_A3_TO_GPA, 1.0])
        self.residuals = residuals
        self.r_squared = r_squared
        self.__parameters = np.array([self.e0, self.v0, b0, self.b0_prime])
        for array in (self.errors, self.residuals):
            array.flags.writeable = False

    def __repr__(self) -> str:
        return (
            f"EquationOfState({self.form}: E0={self.e0:.6g} eV, V0={self.v0:.6g} Å³, "
            f"B0={self.b0:.6g} GPa, B0'={self.b0_prime:.4g})"
        )

    def energy(self, volume: ArrayLike) -> np.ndarray:
        """The energy at volumes.

        Parameters
        ----------
        volume : array_like
            float64 volumes in Å³.

        Returns
        -------
        numpy.ndarray
            float64 energies in eV, of the shape of volume.
        """
        model = _FORMS[self.form][0]
        return np.asarray(model(volume, *self.__parameters), dtype=np.float64)

    def pressure(self, volume: ArrayLike) -> np.ndarray:
        """The pressure at volumes, -dE/dV.

        Parameters
        ----------
        volume : array_like
            float64 volumes in Å³.

        Returns
        -------
        numpy.ndarray
            float64 pressures in GPa, of the shape of volume.
        """
        v = np.asarray(volume, dtype=np.float64)
        _, v0, b0, b0_prime = self.__parameters
        return _FORMS[self.form][1](v, v0, b0, b0_prime) * units.EV_PER_A3_TO_GPA


def fit_eos(
    volumes: ArrayLike,
    energies: ArrayLike,
    *,
    form: str = "birch_murnaghan",
) -> EquationOfState:
    """An equation of state fitted to energies at volumes by least squares:
    the energy at its minimum, the volume there, the bulk modulus and its
    derivative with respect to the pressure.

    Parameters
    ----------
    volumes : array_like
        (n,) float64 volumes in Å³, five or more, of a cell or per atom.
    energies : array_like
        (n,) float64 energies at the volumes in eV, of the same cell or per
        atom.
    form : str, optional
        ``"birch_murnaghan"`` (third order), ``"vinet"`` or ``"murnaghan"``.
        Default ``"birch_murnaghan"``.

    Returns
    -------
    EquationOfState
        The parameters, their errors, and the energy and pressure at any
        volume.

    Raises
    ------
    InvalidParameter
        For an unknown form, fewer than five volumes, volumes that are not
        positive, values that are not finite, or energies without a minimum
        within the volumes.
    """
    if form not in _FORMS:
        raise InvalidParameter(f"form must be one of {FORMS}, not {form!r}")
    volumes = _numbers(volumes, "volumes").reshape(-1)
    energies = _numbers(energies, "energies").reshape(-1)
    if len(volumes) < 5 or energies.shape != volumes.shape:
        raise InvalidParameter(
            "volumes and energies must be five or more each, as many"
        )
    if not (volumes > 0).all():
        raise InvalidParameter("volumes must be positive")
    # the third-order Birch-Murnaghan form is a cubic in V^(-2/3): its least
    # squares are linear, and start the other forms
    t = volumes ** (-2 / 3)
    cubic = fit_polynomial(t, energies, 3)
    slope = np.polynomial.polynomial.polyder(cubic.coefficients)
    found = None
    for root in np.polynomial.polynomial.polyroots(slope):
        if (
            abs(root.imag) > 1e-9 * abs(root.real)
            or not t.min() <= root.real <= t.max()
        ):
            continue
        t0 = root.real
        if cubic(t0, 2) > 0:
            found = t0
    if found is None:
        raise InvalidParameter("the energies have no minimum within the volumes")
    v0 = found ** (-1.5)
    # derivatives with respect to V at V0, where dE/dt = 0
    dt = -2 / 3 * v0 ** (-5 / 3)
    d2t = 10 / 9 * v0 ** (-8 / 3)
    second = cubic(found, 2) * dt**2
    third = cubic(found, 3) * dt**3 + 3 * cubic(found, 2) * dt * d2t
    guess = np.array([float(cubic(found)), v0, v0 * second, -1 - v0 * third / second])
    model, _ = _FORMS[form]
    fitted = fit_curve(model, volumes, energies, guess)
    return EquationOfState(
        form,
        fitted.parameters,
        np.array(fitted.errors),
        np.array(fitted.residuals),
        fitted.r_squared,
    )
