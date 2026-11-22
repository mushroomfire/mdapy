# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The exchange-correlation functionals of the plane-wave DFT on the GPU against
libxc, point by point: the energy per volume and its derivatives by the spin
densities and by the products of their gradients, for LDA with both sets of
constants of Perdew and Wang and for PBE, polarized and not."""

import numpy as np
import pytest

_dft = pytest.importorskip("mdapy_dft_cuda12._dft")
pylibxc = pytest.importorskip("pylibxc")
if _dft.device_count() == 0:
    pytest.skip("no GPU", allow_module_level=True)

PARTS = {
    "lda_pw": ["lda_x", "lda_c_pw"],
    "lda_pw_mod": ["lda_x", "lda_c_pw_mod"],
    "pbe": ["gga_x_pbe", "gga_c_pbe"],
}


def points(n=2000, seed=0, polarized=True):
    rng = np.random.default_rng(seed)
    total = 10 ** rng.uniform(-3, 1, n)
    zeta = rng.uniform(-0.95, 0.95, n) if polarized else np.zeros(n)
    rho = np.stack([total * (1 + zeta) / 2, total * (1 - zeta) / 2], axis=1)
    # gradients of each spin density, of reduced gradients up to about 3
    scale = rho * np.cbrt(rho) * rng.uniform(0.0, 3.0, (n, 1))
    gu = rng.normal(size=(n, 3)) * scale[:, :1]
    gd = (
        gu * (rho[:, 1:] / rho[:, :1])
        if not polarized
        else rng.normal(size=(n, 3)) * scale[:, 1:]
    )
    sigma = np.stack([(gu * gu).sum(1), (gu * gd).sum(1), (gd * gd).sum(1)], axis=1)
    return rho, sigma


def libxc(name, rho, sigma):
    f = np.zeros(len(rho))
    v = np.zeros_like(rho)
    vs = np.zeros_like(sigma)
    for part in PARTS[name]:
        func = pylibxc.LibXCFunctional(part, "polarized")
        out = func.compute({"rho": rho.ravel(), "sigma": sigma.ravel()})
        f += out["zk"].ravel() * rho.sum(axis=1)
        v += out["vrho"].reshape(-1, 2)
        if "vsigma" in out:
            vs += out["vsigma"].reshape(-1, 3)
    return f, v, vs


@pytest.mark.parametrize("name", list(PARTS))
@pytest.mark.parametrize("polarized", [True, False])
def test_against_libxc(name, polarized):
    rho, sigma = points(polarized=polarized)
    f, v, vs = _dft.xc(name, rho, sigma)
    ef, ev, evs = libxc(name, rho, sigma)
    np.testing.assert_allclose(f, ef, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(v, ev, rtol=1e-9, atol=1e-12)
    if name == "pbe":
        np.testing.assert_allclose(vs, evs, rtol=1e-9, atol=1e-12)
    else:
        assert not vs.any()


def test_thresholds_and_names():
    rho = np.array([[1e-12, 1e-12], [0.5, 0.0], [0.3, 0.2]])
    sigma = np.zeros((3, 3))
    f, v, vs = _dft.xc("pbe", rho, sigma, threshold=1e-10)
    assert f[0] == 0.0 and not v[0].any()  # below the threshold
    assert f[1] == 0.0  # a spin density of zero
    assert f[2] < 0.0
    with pytest.raises(ValueError, match="no functional"):
        _dft.xc("b3lyp", rho, sigma)
