# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The one-center terms of PAW: the datasets of pslibrary read, the quadrature
of the directions and the harmonics with their derivatives, the radial
Hartree potential against that of Gaussians, and the strengths of the NumPy
reference against the differences of its energy."""

import math

import numpy as np
import pytest

from mdapy.calculators._dft import paw, radial
from mdapy.calculators._dft.upf import read_upf

from dft import espresso, paw_reference

_dft = pytest.importorskip("mdapy_dft_cuda12._dft")


def dataset(element):
    name = {"Al": "Al.pbe-n-kjpaw_psl.1.0.0.UPF", "Fe": "Fe.pbe-spn-kjpaw_psl.1.0.0.UPF",
            "Si": "Si.pbe-n-kjpaw_psl.1.0.0.UPF", "Ni": "Ni.pbe-spn-kjpaw_psl.1.0.0.UPF"}[element]  # fmt: skip
    return read_upf(espresso.pseudo("pslibrary/" + name))


def test_a_paw_dataset():
    pp = dataset("Fe")
    data = pp.paw
    assert pp.ultrasoft and data is not None and data.lmax_rho == 4
    assert data.augmentation_points == 868 and pp.kkbeta == 868
    assert data.occupations.sum() == pytest.approx(pp.valence)
    n = len(pp.projectors)
    for i in range(n):
        for j in range(i, n):
            if pp.projectors[i].angular != pp.projectors[j].angular:
                continue
            # q_ij = int (phi_i phi_j - phi~_i phi~_j) dr, the L = 0 multipole
            diff = (
                data.ae_waves[i] * data.ae_waves[j]
                - data.ps_waves[i] * data.ps_waves[j]
            )
            assert radial.simpson(diff, pp.rab, pp.kkbeta) == pytest.approx(
                pp.q[i, j], abs=1e-6
            )
            assert data.multipoles[i, j, 0] == pytest.approx(pp.q[i, j], abs=1e-10)
    for (i, j, ell), charge in pp.augmentation.items():
        assert not charge[data.augmentation_points :].any()
        assert abs(data.multipoles[i, j, ell]) >= 1e-12
    assert data.core_energy < 0 and data.ae_core[0] > pp.core[0]


@pytest.mark.parametrize("lmax_rho", [0, 2, 4])
@pytest.mark.parametrize("gga", [False, True])
def test_quadrature(lmax_rho, gga):
    q = paw.quadrature(lmax_rho, gga)
    y = q["ylm"]
    assert y.shape[1] == q["lh"] and q["ww"].sum() == pytest.approx(4 * math.pi)
    # the harmonics up to lh are orthonormal on the directions
    np.testing.assert_allclose(
        (q["ww"][:, None] * y).T @ y, np.eye(q["lh"]), atol=1e-13
    )
    # the derivatives against differences of the harmonics
    theta, phi = q["theta"], q["phi"]
    top = math.isqrt(q["lh"]) - 1
    h = 1e-5

    def diff(dt, dp):
        a = paw._harmonics(top, theta + dt, phi + dp)
        b = paw._harmonics(top, theta - dt, phi - dp)
        return ((a - b) / (2 * h)).T

    np.testing.assert_allclose(q["dylmt"], diff(h, 0), atol=1e-8)
    np.testing.assert_allclose(q["dylmp"], diff(0, h) / q["sin"][:, None], atol=1e-8)


def trapezoid_from_zero(f, s):
    return np.concatenate([[0.0], np.cumsum(0.5 * np.diff(s) * (f[1:] + f[:-1]))])


def test_hartree_of_gaussians():
    dx, xmin = 0.0125, -7.0
    r = np.exp(xmin + dx * np.arange(1100)) / 13.0
    s = np.linspace(1e-9, 12.0, 240001)
    some = (r > 0.05) & (r < 8.0)
    for ell in range(4):
        v = paw_reference.hartree(
            ell, 4 * math.pi / (2 * ell + 1) * r ** (ell + 2) * np.exp(-r * r), r, dx
        )
        # V_l = 4 pi / (2l + 1) (r^-(l+1) int_0^r n s^(l+2) ds + r^l int_r^inf n
        # s^(1-l) ds) of n = r^l exp(-r^2)
        ns = s**ell * np.exp(-s * s)
        inner = trapezoid_from_zero(ns * s ** (ell + 2), s)
        outer = trapezoid_from_zero(ns * s ** (1.0 - ell), s)
        outer = outer[-1] - outer
        x = r[some]
        exact = 4 * math.pi / (2 * ell + 1) * (
            np.interp(x, s, inner) / x ** (ell + 1) + x**ell * np.interp(x, s, outer)
        )  # fmt: skip
        np.testing.assert_allclose(v[some], exact, rtol=1e-6, atol=1e-9)
    # l = 0 in closed form, pi^(3/2) erf(r) / r, everywhere
    v0 = paw_reference.hartree(0, 4 * math.pi * r * r * np.exp(-r * r), r, dx)
    erf = np.vectorize(math.erf)
    np.testing.assert_allclose(v0, math.pi**1.5 * erf(r) / r, rtol=1e-7)


def xc_of(name):
    def xc(rho, sigma, polarized, signed_gradient):
        return _dft.xc(name, np.ascontiguousarray(rho), np.ascontiguousarray(sigma),
                       polarized=polarized, signed_gradient=signed_gradient)  # fmt: skip

    return xc


@pytest.mark.parametrize("spins", [1, 2])
@pytest.mark.parametrize("name", ["lda_pw", "pbe_pw"])
def test_reference_strengths_are_derivatives(spins, name):
    if _dft.device_count() == 0:
        pytest.skip("no GPU")
    pp = dataset("Fe")
    data = paw.onecenter(pp, name.startswith("pbe"))
    rng = np.random.default_rng(1)
    becsum = paw.initial_becsum(pp, spins, 2.0 if spins == 2 else 0.0)
    becsum = becsum + 0.02 * rng.normal(size=becsum.shape)
    energies, d = paw_reference.onecenter(data, becsum, name, xc_of(name))
    assert np.isfinite(energies["total"])
    h = 1e-5
    for p in rng.choice(becsum.shape[1], 6, replace=False):
        for s in range(spins):
            up, down = becsum.copy(), becsum.copy()
            up[s, p] += h
            down[s, p] -= h
            fd = (paw_reference.onecenter(data, up, name, xc_of(name))[0]["total"]
                  - paw_reference.onecenter(data, down, name, xc_of(name))[0]["total"]) / (2 * h)  # fmt: skip
            # exact but for the discretization of QE: the ends of the radial
            # integrals and, with GGA, its radial derivatives and divergence
            tolerance = (1e-4 if name.startswith("pbe") else 5e-6) * np.abs(d).max()
            assert d[s, p] == pytest.approx(fd, abs=tolerance)


@pytest.mark.parametrize("spins", [1, 2])
@pytest.mark.parametrize("name", ["lda_pw", "pbe_pw"])
@pytest.mark.parametrize("element", ["Fe", "Al"])
def test_gpu_against_reference(element, spins, name):
    """Two atoms of one species and different occupations, the GPU's
    energies and strengths against the reference, atom by atom."""
    if _dft.device_count() == 0:
        pytest.skip("no GPU")
    pp = dataset(element)
    data = paw.onecenter(pp, name.startswith("pbe"))
    width = data["functions"]
    rng = np.random.default_rng(3)
    first = paw.initial_becsum(pp, spins, 1.0 if spins == 2 else 0.0)
    atoms = [first + 0.02 * rng.normal(size=first.shape) for _ in range(2)]
    gpu = _dft.OneCenter([data], [0, 0], [width, width], spins, name)
    energies, d = gpu.compute(np.ascontiguousarray(np.concatenate(atoms, axis=1)))
    n = first.shape[1]
    expected = {k: 0.0 for k in ("hartree_ae", "xc_ae", "hartree_ps", "xc_ps", "total")}
    for i, b in enumerate(atoms):
        e, di = paw_reference.onecenter(data, b, name, xc_of(name))
        for k in expected:
            expected[k] += e[k]
        np.testing.assert_allclose(
            d[:, i * n : (i + 1) * n], di, rtol=0, atol=1e-11 * np.abs(di).max()
        )
    for k, value in expected.items():
        assert energies[k] == pytest.approx(value, rel=1e-11, abs=1e-11)
