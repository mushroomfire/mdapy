# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The exchange-correlation functionals of the plane-wave DFT on the GPU against
libxc, point by point: the energy per volume and its derivatives by the spin
densities and by the products of their gradients, for LDA with both sets of
constants of Perdew and Wang and for PBE and PBEsol, polarized and not."""

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
    "pbesol": ["gga_x_pbe_sol", "gga_c_pbe_sol"],
}


def points(n=2000, seed=0, polarized=True):
    rng = np.random.default_rng(seed)
    total = 10 ** rng.uniform(-3, 1, n)
    zeta = rng.uniform(-0.95, 0.95, n) if polarized else np.zeros(n)
    rho = np.stack([total * (1 + zeta) / 2, total * (1 - zeta) / 2], axis=1)
    # gradients of each spin density, of reduced gradients up to about 3
    scale = rho * np.cbrt(rho) * rng.uniform(0.05, 3.0, (n, 1))
    gu = rng.normal(size=(n, 3)) * scale[:, :1]
    gd = (
        gu * (rho[:, 1:] / rho[:, :1])
        if not polarized
        else rng.normal(size=(n, 3)) * scale[:, 1:]
    )
    sigma = np.stack([(gu * gu).sum(1), (gu * gd).sum(1), (gd * gd).sum(1)], axis=1)
    # away from the gradient thresholds of QE, which libxc has not
    keep = sigma @ [1, 2, 1] > 1e-10
    return rho[keep], sigma[keep]


# the derivatives by sigma in the potential of each spin when both spins have
# the same gradient, which is all a calculation without spin uses
SHARE = np.array([[2.0, 0.0], [1.0, 1.0], [0.0, 2.0]])


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
    f, v, vs = _dft.xc(name, rho, sigma, polarized=polarized)
    ef, ev, evs = libxc(name, rho, sigma)
    np.testing.assert_allclose(f, ef, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(v, ev, rtol=1e-9, atol=1e-12)
    if name.startswith("pbe"):
        if not polarized:
            vs, evs = vs @ SHARE, evs @ SHARE
        np.testing.assert_allclose(vs, evs, rtol=1e-9, atol=1e-12)
    else:
        assert not vs.any()


def test_thresholds_and_names():
    rho = np.array([[1e-12, 1e-12], [0.5, 0.0], [0.3, 0.2], [-0.3, -0.2]])
    sigma = np.zeros((4, 3))
    f, v, vs = _dft.xc("pbe", rho, sigma, threshold=1e-10)
    assert f[0] == 0.0 and not v[0].any()  # below the threshold
    assert f[1] < 0.0 and np.isfinite(v[1]).all()  # a fully polarized point
    # without a gradient PBE is its LDA
    lda = _dft.xc("lda_pw_mod", rho, sigma)
    np.testing.assert_allclose(f[1:], lda[0][1:], rtol=1e-14)
    # a negative density as QE: the energy of |n| with its sign, the
    # potentials of that point
    assert f[3] == pytest.approx(-f[2], rel=1e-14)
    np.testing.assert_allclose(v[3], v[2][::-1], rtol=1e-14)
    with pytest.raises(ValueError, match="no functional"):
        _dft.xc("b3lyp", rho, sigma)


def test_negative_densities_with_gradients():
    rho = np.array([[0.3, 0.2], [-0.3, -0.2], [0.02, -0.01], [-0.02, 0.01]])
    sigma = np.tile([0.04, 0.01, 0.03], (4, 1))
    # with spin, no gradient terms where n is negative
    f, v, vs = _dft.xc("pbe", rho, sigma)
    lda = _dft.xc("lda_pw_mod", rho, sigma)
    assert f[0] != pytest.approx(lda[0][0], rel=1e-3)
    np.testing.assert_allclose(f[[1, 3]], lda[0][[1, 3]], rtol=1e-14)
    np.testing.assert_allclose(v[[1, 3]], lda[1][[1, 3]], rtol=1e-14)
    assert not vs[[1, 3]].any()
    # the exchange of the positive spin alone where n is positive, without
    # correlation below its threshold
    tiny = np.array([[2e-7, -1e-7]])
    g = np.array([[1e-4, 0.0, 0.0]])
    f, v, vs = _dft.xc("pbe", tiny, g)
    assert vs[0, 0] != 0.0 and vs[0, 1] == 0.0 and vs[0, 2] == 0.0
    # without spin, the energy of |n| with its sign, its potentials as they are
    n = np.array([[0.25, 0.25], [-0.25, -0.25]])
    s2 = np.tile([0.01, 0.01, 0.01], (2, 1))
    f, v, vs = _dft.xc("pbe", n, s2, polarized=False)
    assert f[1] == pytest.approx(-f[0], rel=1e-14)
    np.testing.assert_allclose(v[1], v[0], rtol=1e-14)
    np.testing.assert_allclose(vs[1], vs[0], rtol=1e-14)


def test_without_spin_as_with():
    """Without spin the gradient terms come of n and |grad n|^2, the same as
    of two equal spins."""
    rng = np.random.default_rng(4)
    n = 10 ** rng.uniform(-3, 1, 200)
    g2 = (n ** (4 / 3) * rng.uniform(0.1, 3.0, 200)) ** 2
    rho = np.stack([n / 2, n / 2], axis=1)
    sigma = np.stack([g2 / 4, g2 / 4, g2 / 4], axis=1)
    for name in ("pbe", "pbe_pw", "pbesol", "pbesol_pw"):
        a = _dft.xc(name, rho, sigma, polarized=False)
        b = _dft.xc(name, rho, sigma, polarized=True)
        for x, y in zip(a[:2], b[:2]):
            np.testing.assert_allclose(x, y, rtol=1e-11)
        np.testing.assert_allclose(a[2] @ SHARE, b[2] @ SHARE, rtol=1e-11)
