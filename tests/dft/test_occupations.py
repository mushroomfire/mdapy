# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The occupations of the smearings: f falls from 1 to 0 with the derivative
-delta of unit weight and no first moment, e' = u delta, and the free energy
has dF/de_n = w f_max f_n at a fixed number of electrons; the Fermi level holds
the electrons; fixed occupations fill the lowest bands; errors."""

import math

import numpy as np
import pytest

_dft = pytest.importorskip("mdapy_dft_cuda12._dft")

SMEARINGS = ["gaussian", "fermi-dirac", "methfessel-paxton", "marzari-vanderbilt"]
U = np.linspace(-40, 40, 80001)  # the tails of Fermi-Dirac reach far


def table(name):
    return np.array([_dft.smeared(name, u) for u in U]).T


@pytest.mark.parametrize("name", SMEARINGS)
def test_the_smeared_step(name):
    f, e = table(name)
    assert f[0] == pytest.approx(1.0, abs=1e-12) and f[-1] == pytest.approx(
        0, abs=1e-12
    )
    delta = -np.gradient(f, U)
    du = U[1] - U[0]
    assert delta.sum() * du == pytest.approx(1.0, abs=1e-6)
    assert (U * delta).sum() * du == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_allclose(np.gradient(e, U), U * delta, atol=2e-6)
    assert e[0] == pytest.approx(0, abs=1e-12) and e[-1] == pytest.approx(0, abs=1e-12)


def test_closed_forms():
    u = 0.37
    assert _dft.smeared("gaussian", u)[0] == pytest.approx(0.5 * math.erfc(u))
    assert _dft.smeared("fermi-dirac", u)[0] == pytest.approx(1 / (1 + math.exp(u)))
    assert _dft.smeared("gaussian", 0.0)[1] == pytest.approx(-0.5 / math.sqrt(math.pi))
    assert _dft.smeared("fermi-dirac", 0.0)[1] == pytest.approx(-math.log(2))
    assert _dft.smeared("fermi-dirac", 800.0) == (0.0, 0.0)
    assert _dft.smeared("fermi-dirac", -800.0) == (1.0, 0.0)


def bands(seed=0):
    rng = np.random.default_rng(seed)
    e = np.sort(rng.uniform(-0.3, 0.6, size=(5, 9)), axis=1)
    w = rng.uniform(0.5, 1.0, size=5)
    return e, w / w.sum()


@pytest.mark.parametrize("name", SMEARINGS)
def test_the_free_energy(name):
    e, w = bands()
    sigma, electrons = 0.02, 7.3

    def free(values):
        f, mu, ts, _ = _dft.occupy(values, w, electrons, 2.0, name, sigma)
        return (w[:, None] * 2 * f * values).sum() + ts, f, mu

    energy, f, mu = free(e)
    assert (w[:, None] * 2 * f).sum() == pytest.approx(electrons, abs=1e-12)
    h = 1e-6
    for k, n in [(0, 3), (2, 4), (4, 8)]:
        up, down = e.copy(), e.copy()
        up[k, n] += h
        down[k, n] -= h
        slope = (free(up)[0] - free(down)[0]) / (2 * h)
        assert slope == pytest.approx(w[k] * 2 * f[k, n], abs=1e-8)


def test_fixed_occupations():
    e, w = bands()
    f, homo, ts, lumo = _dft.occupy(e, w, 8.0, 2.0, "fixed")
    np.testing.assert_array_equal(f, np.repeat([[1.0] * 4 + [0.0] * 5], 5, axis=0))
    assert homo == e[:, 3].max() and lumo == e[:, 4].min() and ts == 0.0
    f, homo, _, lumo = _dft.occupy(e, w, 9.0, 1.0, "fixed")
    assert homo == lumo == e[:, 8].max()


def test_errors():
    e, w = bands()
    with pytest.raises(ValueError, match="whole bands"):
        _dft.occupy(e, w, 7.0, 2.0, "fixed")
    with pytest.raises(ValueError, match="cannot hold"):
        _dft.occupy(e, w, 19.0, 2.0, "gaussian", 0.01)
    with pytest.raises(ValueError, match="positive width"):
        _dft.occupy(e, w, 7.0, 2.0, "gaussian", 0.0)
    with pytest.raises(ValueError, match="no smearing named"):
        _dft.occupy(e, w, 7.0, 2.0, "cold", 0.01)
    with pytest.raises(ValueError, match="every k-point"):
        _dft.occupy(e, w[:3], 7.0, 2.0, "gaussian", 0.01)
