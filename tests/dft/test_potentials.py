# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The potentials of a density on the GPU: the Hartree energy and potential
against NumPy on the sphere of the density cutoff; the exchange-correlation
energy against libxc with the gradients of NumPy; and each potential the
derivative of its energy, by central differences of the energy along a change
of the density, for LDA and PBE, with and without spin and core charge."""

import math

import numpy as np
import pytest

from dft import gpu

_dft = gpu.module()
if _dft.device_count() == 0:
    pytest.skip("no GPU", allow_module_level=True)

BOHR = 0.529177210903
CUTOFF = 12.0  # hartree, of the density


def cell():
    rng = np.random.default_rng(3)
    return np.diag([8.0, 9.0, 10.0]) + rng.uniform(-1.0, 1.0, (3, 3))


def waves(grid, seed, amplitude, modes=8):
    """A sum of a few cosines of low Fourier components, well inside the
    sphere, of mean zero."""
    rng = np.random.default_rng(seed)
    j = np.stack(np.meshgrid(*(np.arange(n) / n for n in grid), indexing="ij"), axis=-1)
    values = np.zeros(tuple(grid))
    for _ in range(modes):
        m = rng.integers(-2, 3, 3)
        if not m.any():
            m[0] = 1
        values += (
            amplitude
            / modes
            * np.cos(2 * math.pi * (j @ m) + rng.uniform(0, 2 * math.pi))
        )
    return values


def smooth(a, grid, seed, mean=0.05, amplitude=0.3):
    """A positive density: mean (1 + waves of the relative amplitude)."""
    return mean * (1.0 + waves(grid, seed, amplitude))


def sphere_part(values, a, grid, cutoff):
    """The Fourier components on the sphere, and the values of those alone."""
    miller, _, box, _ = _dft.sphere(a, [0, 0, 0], cutoff, grid)
    c = np.fft.fftn(values).ravel()[box] / values.size
    return miller, c


def test_hartree_against_numpy():
    a = cell()
    grid = _dft.fft_grid(a, CUTOFF)
    rho = smooth(a, grid, 0)[None]
    hxc, (hartree, xc, _) = _dft.potentials(a, CUTOFF, "lda_pw", rho)
    miller, c = sphere_part(rho[0], a, grid, CUTOFF)
    g = miller @ (2 * math.pi * np.linalg.inv(a).T)
    g2 = (g * g).sum(axis=1)
    volume = abs(np.linalg.det(a))
    expected = 0.5 * volume * (4 * math.pi * np.abs(c[1:]) ** 2 / g2[1:]).sum()
    assert hartree == pytest.approx(expected, rel=1e-12)


def energy(a, functional, rho, core=None):
    return _dft.potentials(a, CUTOFF, functional, rho, core)[1]


@pytest.mark.parametrize("functional", ["lda_pw", "pbe"])
@pytest.mark.parametrize("spins", [1, 2])
@pytest.mark.parametrize("with_core", [False, True])
def test_the_potential_is_the_derivative_of_the_energy(functional, spins, with_core):
    a = cell()
    grid = _dft.fft_grid(a, CUTOFF)
    rho = np.stack([smooth(a, grid, 1 + s, mean=0.05 / spins) for s in range(spins)])
    core = smooth(a, grid, 9, mean=0.02) if with_core else None
    hxc, _ = _dft.potentials(a, CUTOFF, functional, rho, core)
    volume = abs(np.linalg.det(a))
    dv = volume / np.prod(grid)
    for s in range(spins):
        delta = np.zeros_like(rho)
        delta[s] = waves(grid, 20 + s, 0.05 * rho[s].mean())
        eps = 1e-3
        plus = energy(a, functional, rho + eps * delta, core)
        minus = energy(a, functional, rho - eps * delta, core)
        numeric = (plus[0] + plus[1] - minus[0] - minus[1]) / (2 * eps)
        analytic = dv * (hxc[s] * delta[s]).sum()
        assert abs(analytic) > 1e-6  # a change the energy feels
        # the energies are sums on the GPU in no fixed order, each to about
        # 1e-15 hartree: 1e-12 in their difference over 2 eps
        assert numeric == pytest.approx(analytic, rel=1e-6, abs=5e-12)


def test_xc_against_libxc():
    pylibxc = pytest.importorskip("pylibxc")
    a = cell()
    grid = _dft.fft_grid(a, CUTOFF)
    rho = np.stack([smooth(a, grid, 4, mean=0.03), smooth(a, grid, 5, mean=0.02)])
    _, (_, xc, _) = _dft.potentials(a, CUTOFF, "pbe", rho)
    miller, _ = sphere_part(rho[0], a, grid, CUTOFF)
    _, _, box, _ = _dft.sphere(a, [0, 0, 0], CUTOFF, grid)
    g = miller @ (2 * math.pi * np.linalg.inv(a).T)
    grads = []
    for s in range(2):
        c = np.fft.fftn(rho[s]).ravel()[box] / rho[s].size
        parts = []
        for j in range(3):
            full = np.zeros(np.prod(grid), dtype=complex)
            full[box] = 1j * g[:, j] * c
            parts.append(np.fft.ifftn(full.reshape(grid)).real * np.prod(grid))
        grads.append(np.stack(parts))
    sigma = np.stack([(grads[0] * grads[0]).sum(0), (grads[0] * grads[1]).sum(0),
                      (grads[1] * grads[1]).sum(0)], axis=-1).reshape(-1, 3)  # fmt: skip
    r = np.stack([rho[0], rho[1]], axis=-1).reshape(-1, 2)
    total = np.zeros(len(r))
    for part in ("gga_x_pbe", "gga_c_pbe"):
        out = pylibxc.LibXCFunctional(part, "polarized").compute(
            {"rho": r.ravel(), "sigma": sigma.ravel()}
        )
        total += out["zk"].ravel() * r.sum(axis=1)
    volume = abs(np.linalg.det(a))
    assert xc == pytest.approx(volume / np.prod(grid) * total.sum(), rel=1e-11)
