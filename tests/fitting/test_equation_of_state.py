# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import pytest

from mdapy import units
from mdapy.build import build_crystal
from mdapy.calculators import EAM
from mdapy.errors import InvalidParameter
from mdapy.fitting import FORMS, fit_eos
from mdapy.fitting._eos import _FORMS

ASE_NAMES = {
    "birch_murnaghan": "birchmurnaghan",
    "vinet": "vinet",
    "murnaghan": "murnaghan",
}


@pytest.fixture(scope="module")
def copper():
    """Energies of the cubic cell of copper of the Zhou EAM at 11 lattice
    constants."""
    potential = EAM.zhou(["Cu"])
    a = np.linspace(3.45, 3.80, 11)
    energies = [
        potential.compute(build_crystal("fcc", s, elements="Cu"), ("energy",)).energy
        for s in a
    ]
    return a**3, np.array(energies)


@pytest.mark.parametrize("form", FORMS)
def test_as_ase(copper, form):
    eos = pytest.importorskip("ase.eos")
    ase_units = pytest.importorskip("ase.units")
    volumes, energies = copper
    ours = fit_eos(volumes, energies, form=form)
    v0, e0, b0 = eos.EquationOfState(volumes, energies, eos=ASE_NAMES[form]).fit()
    assert ours.v0 == pytest.approx(v0, rel=1e-8)
    assert ours.e0 == pytest.approx(e0, rel=1e-10)
    assert ours.b0 == pytest.approx(b0 / ase_units.GPa, rel=1e-6)
    assert ours.form == form and form in repr(ours)


@pytest.mark.parametrize("form", FORMS)
def test_exact_data_and_the_pressure(form):
    truth = (-3.5, 11.8, 140.0 * units.GPA_TO_EV_PER_A3, 4.8)
    volumes = np.linspace(10.5, 13.5, 9)
    model, pressure = _FORMS[form]
    fitted = fit_eos(volumes, model(volumes, *truth), form=form)
    assert (fitted.e0, fitted.v0, fitted.b0, fitted.b0_prime) == pytest.approx(
        (-3.5, 11.8, 140.0, 4.8), rel=1e-7
    )
    assert fitted.r_squared == pytest.approx(1)
    np.testing.assert_allclose(fitted.residuals, 0, atol=1e-10)
    np.testing.assert_allclose(
        fitted.energy(volumes), model(volumes, *truth), atol=1e-10
    )
    # the pressure is -dE/dV, zero at the minimum
    h = 1e-5
    numeric = -(fitted.energy(volumes + h) - fitted.energy(volumes - h)) / (2 * h)
    np.testing.assert_allclose(
        fitted.pressure(volumes), numeric * units.EV_PER_A3_TO_GPA, rtol=1e-6, atol=1e-6
    )
    assert fitted.pressure(fitted.v0) == pytest.approx(0, abs=1e-9)
    assert fitted.errors.shape == (4,) and (fitted.errors < 1e-5).all()


def test_errors_grow_with_noise(copper):
    volumes, energies = copper
    rng = np.random.default_rng(0)
    noisy = fit_eos(volumes, energies + rng.normal(0, 1e-3, len(volumes)))
    assert noisy.b0 == pytest.approx(fit_eos(volumes, energies).b0, rel=0.05)
    assert 0 < noisy.errors[2] < 10  # GPa


def test_eos_errors():
    volumes = np.linspace(10, 14, 6)
    with pytest.raises(InvalidParameter, match="form must be one of"):
        fit_eos(volumes, volumes, form="birch")
    with pytest.raises(InvalidParameter, match="five or more"):
        fit_eos(volumes[:4], volumes[:4])
    with pytest.raises(InvalidParameter, match="positive"):
        fit_eos(volumes - 12, volumes)
    with pytest.raises(InvalidParameter, match="no minimum within the volumes"):
        fit_eos(volumes, -volumes)  # falling all the way
