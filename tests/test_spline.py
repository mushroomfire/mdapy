# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""Tests for the general (non-uniform) ``CubicSpline`` exposed as ``Spline``.

Each test compares against ``scipy.interpolate.CubicSpline`` with the matching
boundary condition. Because we implement the same textbook tridiagonal system
as scipy, the two should agree to floating-point precision on well-conditioned
data.
"""

import numpy as np
import pytest
from mdapy.spline import Spline

scipy_cs = pytest.importorskip("scipy.interpolate").CubicSpline


# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------
def uniform_grid(n=21, a=0.0, b=2 * np.pi):
    return np.linspace(a, b, n)


def nonuniform_grid(n=21, a=0.0, b=2 * np.pi, seed=0):
    """Jittered grid so spacings vary by up to ~30%."""
    rng = np.random.default_rng(seed)
    x = np.linspace(a, b, n)
    jitter = rng.uniform(-0.3, 0.3, size=n - 2) * (b - a) / (n - 1)
    x[1:-1] += jitter
    x.sort()
    return x


# ---------------------------------------------------------------------------
# Reproduction at grid points
# ---------------------------------------------------------------------------
class TestGridReproduction:
    """A spline must pass through every sample point exactly."""

    @pytest.mark.parametrize("bc_type", ["not-a-knot", "natural", "clamped"])
    @pytest.mark.parametrize(
        "grid_fn", [uniform_grid, nonuniform_grid], ids=["uniform", "nonuniform"]
    )
    def test_hits_every_knot(self, bc_type, grid_fn):
        x = grid_fn()
        y = np.sin(x)
        sp = Spline(x, y, bc_type=bc_type)
        for xi, yi in zip(x, y):
            assert abs(sp.evaluate(xi) - yi) < 1e-12

    def test_clamped_user_derivs(self):
        x = uniform_grid()
        y = np.sin(x)
        sp = Spline(x, y, bc_type="clamped", dy0=np.cos(x[0]), dyn=np.cos(x[-1]))
        for xi, yi in zip(x, y):
            assert abs(sp.evaluate(xi) - yi) < 1e-12


# ---------------------------------------------------------------------------
# Bit-level agreement with scipy
# ---------------------------------------------------------------------------
class TestVsScipy:
    """Same BC -> same output as scipy (modulo FP rounding)."""

    probe = np.linspace(0.01, 2 * np.pi - 0.01, 257)

    @pytest.mark.parametrize(
        "grid_fn", [uniform_grid, nonuniform_grid], ids=["uniform", "nonuniform"]
    )
    def test_not_a_knot(self, grid_fn):
        x = grid_fn()
        y = np.sin(x)
        mda = Spline(x, y, bc_type="not-a-knot")
        sc = scipy_cs(x, y, bc_type="not-a-knot")
        assert np.max(np.abs(mda.evaluate(self.probe) - sc(self.probe))) < 1e-13

    @pytest.mark.parametrize(
        "grid_fn", [uniform_grid, nonuniform_grid], ids=["uniform", "nonuniform"]
    )
    def test_natural(self, grid_fn):
        x = grid_fn()
        y = np.sin(x)
        mda = Spline(x, y, bc_type="natural")
        sc = scipy_cs(x, y, bc_type="natural")
        assert np.max(np.abs(mda.evaluate(self.probe) - sc(self.probe))) < 1e-13

    @pytest.mark.parametrize(
        "grid_fn", [uniform_grid, nonuniform_grid], ids=["uniform", "nonuniform"]
    )
    def test_clamped(self, grid_fn):
        x = grid_fn()
        y = np.sin(x)
        mda = Spline(x, y, bc_type="clamped", dy0=np.cos(x[0]), dyn=np.cos(x[-1]))
        sc = scipy_cs(x, y, bc_type=((1, np.cos(x[0])), (1, np.cos(x[-1]))))
        assert np.max(np.abs(mda.evaluate(self.probe) - sc(self.probe))) < 1e-13

    def test_derivative_matches_scipy(self):
        x = uniform_grid(51)
        y = np.sin(x)
        mda = Spline(x, y)
        sc = scipy_cs(x, y)
        d_mda = mda.derivative(self.probe)
        d_sc = sc(self.probe, 1)
        assert np.max(np.abs(d_mda - d_sc)) < 1e-13

    def test_second_derivative_matches_scipy(self):
        x = uniform_grid(51)
        y = np.sin(x)
        mda = Spline(x, y)
        sc = scipy_cs(x, y)
        dd_mda = mda.second_derivative(self.probe)
        dd_sc = sc(self.probe, 2)
        assert np.max(np.abs(dd_mda - dd_sc)) < 1e-13


# ---------------------------------------------------------------------------
# Polynomial reproduction
# ---------------------------------------------------------------------------
class TestPolynomialReproduction:
    """A cubic spline reproduces any cubic exactly (given the right BC)."""

    probe = np.linspace(0.1, 1.9, 100)

    def test_cubic_reproduced_by_clamped(self):
        # A clamped spline with the exact endpoint slopes reproduces any
        # cubic polynomial exactly.
        x = np.linspace(0, 2, 11)
        # p(x) = 1 + 2x - x^2 + 0.5 x^3
        coeffs = np.array([0.5, -1.0, 2.0, 1.0])  # highest power first
        p = np.poly1d(coeffs)
        dp = p.deriv()
        y = p(x)
        sp = Spline(x, y, bc_type="clamped", dy0=dp(x[0]), dyn=dp(x[-1]))
        y_mda = sp.evaluate(self.probe)
        y_true = p(self.probe)
        assert np.max(np.abs(y_mda - y_true)) < 1e-12

    def test_cubic_reproduced_by_notaknot(self):
        # not-a-knot also reproduces cubics (since the first two pieces being
        # the same polynomial is automatically satisfied).
        x = np.linspace(0, 2, 11)
        p = np.poly1d([0.5, -1.0, 2.0, 1.0])
        y = p(x)
        sp = Spline(x, y, bc_type="not-a-knot")
        y_mda = sp.evaluate(self.probe)
        y_true = p(self.probe)
        assert np.max(np.abs(y_mda - y_true)) < 1e-12

    def test_linear_reproduced_by_natural(self):
        # Natural BC (y''=0 at ends) reproduces LINEAR functions exactly
        # (but not quadratics - because a quadratic has nonzero y'').
        x = np.linspace(0, 2, 11)
        p = np.poly1d([3.0, -2.0])  # linear
        y = p(x)
        sp = Spline(x, y, bc_type="natural")
        y_mda = sp.evaluate(self.probe)
        y_true = p(self.probe)
        assert np.max(np.abs(y_mda - y_true)) < 1e-12


# ---------------------------------------------------------------------------
# Derivative accuracy
# ---------------------------------------------------------------------------
class TestDerivativeAccuracy:
    """Analytic derivative matches numerical differentiation of the spline."""

    def test_first_derivative_matches_fd(self):
        x = np.linspace(0, 2 * np.pi, 41)
        y = np.sin(x)
        sp = Spline(x, y)
        xq = np.linspace(0.1, 2 * np.pi - 0.1, 50)
        eps = 1e-6
        fd = (sp.evaluate(xq + eps) - sp.evaluate(xq - eps)) / (2 * eps)
        d_analytic = sp.derivative(xq)
        assert np.max(np.abs(fd - d_analytic)) < 1e-6

    def test_second_derivative_matches_fd(self):
        x = np.linspace(0, 2 * np.pi, 41)
        y = np.sin(x)
        sp = Spline(x, y)
        xq = np.linspace(0.1, 2 * np.pi - 0.1, 50)
        eps = 1e-4
        fd = (sp.derivative(xq + eps) - sp.derivative(xq - eps)) / (2 * eps)
        dd_analytic = sp.second_derivative(xq)
        # Second-derivative of a cubic spline is piecewise linear; coarser
        # tolerance for FD.
        assert np.max(np.abs(fd - dd_analytic)) < 1e-4


# ---------------------------------------------------------------------------
# Large dynamic range / pathological shapes
# ---------------------------------------------------------------------------
class TestLargeDynamicRange:
    """Error magnitude on functions that vary over many decades."""

    @pytest.mark.parametrize("bc_type", ["not-a-knot", "natural"])
    def test_agrees_with_scipy_on_exponential(self, bc_type):
        # Pure exponential decay over ~4 decades.
        x = np.linspace(0, 4, 41)
        y = np.exp(-2.0 * x)
        mda = Spline(x, y, bc_type=bc_type)
        sc = scipy_cs(x, y, bc_type=bc_type)
        xq = np.linspace(0.01, 3.99, 200)
        assert np.max(np.abs(mda.evaluate(xq) - sc(xq))) < 1e-13

    def test_lj_like_shape_agrees_with_scipy(self):
        # 12-6 Lennard-Jones from r=0.9 sigma out to r=3 sigma. Steep repulsive
        # wall at the left, flat well bottom, shallow attractive tail.
        # Strong curvature variation - canonical stress test for cubic splines.
        r = np.linspace(0.9, 3.0, 60)
        LJ = 4.0 * (r ** (-12) - r ** (-6))
        mda = Spline(r, LJ)  # not-a-knot default
        sc = scipy_cs(r, LJ)
        rq = np.linspace(0.91, 2.99, 300)
        assert np.max(np.abs(mda.evaluate(rq) - sc(rq))) < 1e-12

    def test_runge_function_agrees_with_scipy(self):
        # Runge's function 1/(1+25 x^2) on [-1, 1]. Classic hard case for
        # high-order polynomial interpolation; cubic splines handle it fine
        # but we verify we get identical results to scipy.
        x = np.linspace(-1, 1, 21)
        y = 1.0 / (1.0 + 25.0 * x ** 2)
        mda = Spline(x, y)
        sc = scipy_cs(x, y)
        xq = np.linspace(-0.99, 0.99, 300)
        assert np.max(np.abs(mda.evaluate(xq) - sc(xq))) < 1e-13


# ---------------------------------------------------------------------------
# Input validation & error handling
# ---------------------------------------------------------------------------
class TestValidation:
    def test_unknown_bc_type_raises(self):
        x = np.linspace(0, 1, 5)
        y = x ** 2
        with pytest.raises(ValueError, match="Unknown bc_type"):
            Spline(x, y, bc_type="periodic")  # not supported here

    def test_partial_clamped_derivs_raises(self):
        x = np.linspace(0, 1, 5)
        y = x ** 2
        with pytest.raises(ValueError, match="both dy0 and dyn"):
            Spline(x, y, bc_type="clamped", dy0=0.0)  # missing dyn

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="Length"):
            Spline([0.0, 1.0, 2.0], [0.0, 1.0])

    def test_too_few_points_raise(self):
        with pytest.raises(ValueError, match="at least 2"):
            Spline([0.0], [0.0])

    def test_non_increasing_x_raises(self):
        # Underlying C++ raises invalid_argument; surfaces as ValueError.
        with pytest.raises(Exception):
            Spline([0.0, 0.0, 1.0], [0.0, 1.0, 2.0])

    def test_out_of_range_scalar_raises(self):
        x = np.linspace(0, 1, 5)
        y = x ** 2
        sp = Spline(x, y)
        with pytest.raises(IndexError):
            sp.evaluate(1.5)
        with pytest.raises(IndexError):
            sp.derivative(-0.1)

    def test_out_of_range_array_returns_nan(self):
        x = np.linspace(0, 1, 5)
        y = x ** 2
        sp = Spline(x, y)
        xq = np.array([-0.1, 0.5, 1.5])
        result = sp.evaluate(xq)
        assert np.isnan(result[0])
        assert not np.isnan(result[1])
        assert np.isnan(result[2])


# ---------------------------------------------------------------------------
# Small-n edge cases
# ---------------------------------------------------------------------------
class TestSmallN:
    def test_n2_is_linear(self):
        # Two points -> straight line, regardless of bc_type.
        x = np.array([0.0, 1.0])
        y = np.array([2.0, 5.0])
        for bc in ["not-a-knot", "natural", "clamped"]:
            sp = Spline(x, y, bc_type=bc)
            assert abs(sp.evaluate(0.3) - (2.0 + 3.0 * 0.3)) < 1e-12

    def test_n3_not_a_knot_falls_back_to_natural(self):
        # With only 3 points the not-a-knot BC at x[1] and x[n-2] are the same
        # equation, leaving the spline underdetermined; we fall back to
        # natural. Check it still reproduces the knots and gives a sensible
        # interpolation between them.
        x = np.array([0.0, 0.5, 1.0])
        y = np.array([0.0, 0.25, 1.0])  # y = x^2
        sp = Spline(x, y, bc_type="not-a-knot")
        for xi, yi in zip(x, y):
            assert abs(sp.evaluate(xi) - yi) < 1e-12
        # natural BC gives s''(0)=s''(1)=0; for y=x^2 the interpolated value
        # at 0.25 should be close to 0.0625 but not exactly (since natural
        # doesn't reproduce quadratics).
        assert 0.03 < sp.evaluate(0.25) < 0.1


# ---------------------------------------------------------------------------
# Batch/scalar API parity
# ---------------------------------------------------------------------------
class TestBatchScalarParity:
    def test_scalar_equals_array_element(self):
        x = np.linspace(0, 2 * np.pi, 17)
        y = np.sin(x)
        sp = Spline(x, y)
        xq = np.array([0.1, 1.2, 3.4, 5.6])
        batch = sp.evaluate(xq)
        for xi, bi in zip(xq, batch):
            assert abs(sp.evaluate(float(xi)) - bi) < 1e-14

    def test_call_alias(self):
        x = np.linspace(0, 1, 10)
        y = x ** 3
        sp = Spline(x, y)
        assert sp(0.5) == sp.evaluate(0.5)

    def test_accepts_list_input(self):
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y = [0.0, 1.0, 4.0, 9.0, 16.0]
        sp = Spline(x, y)
        assert abs(sp.evaluate(2.0) - 4.0) < 1e-12
        result = sp.evaluate([0.5, 1.5, 2.5])
        assert result.shape == (3,)
