# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import pytest

from mdapy import units
from mdapy.errors import InvalidParameter
from mdapy.fitting import (
    MODELS,
    Model,
    arrhenius,
    exponential,
    fit_curve,
    fit_polynomial,
    fit_surface,
    gaussian,
    lorentzian,
    power_law,
    stretched_exponential,
)

optimize = pytest.importorskip("scipy.optimize")


def test_polynomial_as_numpy():
    rng = np.random.default_rng(0)
    x = np.linspace(300, 1500, 30)
    y = 1e-3 * x + 2e-7 * x**2 + rng.normal(0, 1e-3, 30)
    fit = fit_polynomial(x, y, 2)
    coefficients, covariance = np.polyfit(x, y, 2, cov=True)
    np.testing.assert_allclose(fit.coefficients, coefficients[::-1], rtol=1e-9)
    np.testing.assert_allclose(fit.covariance, covariance[::-1, ::-1], rtol=1e-8)
    np.testing.assert_allclose(
        fit.residuals, y - np.polyval(coefficients, x), atol=1e-12
    )
    np.testing.assert_allclose(fit(x), np.polyval(coefficients, x), rtol=1e-12)
    np.testing.assert_allclose(fit(x, 1), 1e-3 + 4e-7 * x, rtol=1e-2)
    np.testing.assert_allclose(fit(x, 2), 2 * fit.coefficients[2], rtol=1e-9)
    assert 0.99 < fit.r_squared < 1
    assert repr(fit) == "PolynomialFit(degree 2)"


@pytest.mark.parametrize("absolute", [False, True])
def test_polynomial_with_sigma_as_scipy(absolute):
    rng = np.random.default_rng(1)
    x = np.linspace(-2, 3, 25)
    sigma = 0.05 + 0.1 * rng.random(25)
    y = 1 - x + 0.5 * x**3 + rng.normal(0, sigma)
    fit = fit_polynomial(x, y, 3, sigma=sigma, absolute_sigma=absolute)
    p, covariance = optimize.curve_fit(
        lambda t, a, b, c, d: a + b * t + c * t**2 + d * t**3,
        x,
        y,
        p0=np.zeros(4),
        sigma=sigma,
        absolute_sigma=absolute,
    )
    np.testing.assert_allclose(fit.coefficients, p, atol=1e-9)
    np.testing.assert_allclose(fit.covariance, covariance, rtol=1e-6, atol=1e-12)


def test_polynomial_exact_and_without_spread():
    x = np.array([0.0, 1, 2, 3])
    fit = fit_polynomial(x, 2 - x + x**3, 3)
    np.testing.assert_allclose(fit.coefficients, [2, -1, 0, 1], atol=1e-12)
    assert np.isnan(fit.errors).all()  # as many coefficients as points
    assert fit.r_squared == pytest.approx(1)
    constant = fit_polynomial([1, 1, 1], [2, 2, 2], 0)  # a single point repeated
    assert constant(5) == pytest.approx(2) and constant.r_squared == 1


def test_polynomial_errors():
    with pytest.raises(InvalidParameter, match="same length"):
        fit_polynomial([1, 2], [1, 2, 3], 1)
    with pytest.raises(InvalidParameter, match="degree must be from 0 to 2"):
        fit_polynomial([1, 2, 3], [1, 2, 3], 3)
    with pytest.raises(InvalidParameter, match="degree must be"):
        fit_polynomial([1, 2, 3], [1, 2, 3], True)
    with pytest.raises(InvalidParameter, match="finite"):
        fit_polynomial([1, 2, np.inf], [1, 2, 3], 1)
    with pytest.raises(InvalidParameter, match="sigma must be positive"):
        fit_polynomial([1, 2, 3], [1, 2, 3], 1, sigma=[1, 0, 1])
    with pytest.raises(InvalidParameter, match="derivative must be"):
        fit_polynomial([1, 2, 3], [1, 2, 3], 1)(1.0, -1)


def test_surface():
    rng = np.random.default_rng(2)
    x, y = rng.random(40) * 4 + 100, rng.random(40) * 2 - 1
    f = lambda a, b: 3 - a + 0.5 * a * b + 0.1 * b**2  # noqa: E731
    exact = fit_surface(x, y, f(x, y), 2)
    terms = dict(zip(map(tuple, exact.powers.tolist()), exact.coefficients))
    expected = {(0, 0): 3, (1, 0): -1, (0, 1): 0, (2, 0): 0, (1, 1): 0.5, (0, 2): 0.1}
    for key, value in expected.items():
        assert terms[key] == pytest.approx(value, abs=1e-8)
    px, py = np.array([101.0, 102.5]), np.array([0.3, -0.7])
    np.testing.assert_allclose(exact(px, py), f(px, py), rtol=1e-12)
    np.testing.assert_allclose(exact(px, py, dx=1), -1 + 0.5 * py, atol=1e-9)
    np.testing.assert_allclose(exact(px, py, dy=1), 0.5 * px + 0.2 * py, rtol=1e-9)
    np.testing.assert_allclose(exact(px, py, dx=2, dy=1), 0, atol=1e-9)
    assert repr(exact) == "SurfaceFit(degree 2)"
    # noisy: least squares of the same terms by NumPy
    z = f(x, y) + rng.normal(0, 0.01, 40)
    noisy = fit_surface(x, y, z, 1)
    design = np.stack([np.ones(40), x, y], axis=1)
    p, *_ = np.linalg.lstsq(design, z, rcond=None)
    left = z - design @ p
    covariance = np.linalg.inv(design.T @ design) * (left @ left) / (40 - 3)
    order = [noisy.powers.tolist().index(t) for t in ([0, 0], [1, 0], [0, 1])]
    np.testing.assert_allclose(noisy.coefficients[order], p, rtol=1e-9)
    np.testing.assert_allclose(
        noisy.covariance[np.ix_(order, order)], covariance, rtol=1e-7
    )


def test_surface_errors():
    with pytest.raises(InvalidParameter, match="same length"):
        fit_surface([1, 2], [1, 2], [1, 2, 3], 1)
    with pytest.raises(InvalidParameter, match="at most 3 terms"):
        fit_surface([1, 2, 3], [1, 2, 4], [1, 2, 3], 2)
    with pytest.raises(InvalidParameter, match="degree must be"):
        fit_surface([1, 2, 3], [1, 2, 4], [1, 2, 3], -1)
    with pytest.raises(InvalidParameter, match="dy must be"):
        fit_surface([1, 2, 3], [1, 2, 4], [1, 2, 3], 1)(1.0, 1.0, dy=-1)


def scipy_fit(model, x, y, guess, **options):
    p, covariance = optimize.curve_fit(model.function, x, y, p0=guess, **options)
    return p, np.sqrt(np.diagonal(covariance))


CASES = {
    "exponential": (exponential, np.linspace(0, 5, 40), (2.0, 1.3), 0.02),
    "stretched exponential": (
        stretched_exponential,
        np.linspace(0.01, 6, 50),
        (1.5, 2.0, 0.6),
        0.01,
    ),
    "gaussian": (gaussian, np.linspace(-3, 5, 60), (1.5, 1.0, 0.7), 0.03),
    "lorentzian": (lorentzian, np.linspace(-3, 5, 60), (1.5, 1.0, 0.7), 0.03),
    "arrhenius": (arrhenius, np.linspace(300, 1500, 25), (1e-3, 0.5), None),
    "power law": (power_law, np.linspace(1, 50, 30), (0.3, 1.7), 0.05),
}


@pytest.mark.parametrize("name", CASES)
def test_models_as_scipy(name):
    model, x, truth, noise = CASES[name]
    rng = np.random.default_rng(3)
    clean = model(x, *truth)
    y = (
        clean * (1 + rng.normal(0, 0.02, len(x)))
        if noise is None
        else clean + rng.normal(0, noise, len(x))
    )
    fit = fit_curve(model, x, y)  # from the guess of the model
    p, errors = scipy_fit(model, x, y, model.guess(x, y))
    np.testing.assert_allclose(fit.parameters, p, rtol=1e-5)
    np.testing.assert_allclose(fit.errors, errors, rtol=1e-3)
    np.testing.assert_allclose(fit.residuals, y - model(x, *fit.parameters), atol=1e-12)
    np.testing.assert_allclose(fit(x), model(x, *fit.parameters))
    for k, parameter in enumerate(model.parameters):
        assert fit[parameter] == fit.parameters[k]
    assert model.name in repr(fit)


def test_noise_free_data_give_the_parameters():
    for model, x, truth, _ in CASES.values():
        fit = fit_curve(model, x, model(x, *truth))
        np.testing.assert_allclose(fit.parameters, truth, rtol=1e-7)
        assert fit.r_squared == pytest.approx(1)
    assert {m.name for m in MODELS} == set(CASES)
    # the Arrhenius energy is in eV with the temperature in K
    t = np.array([400.0, 800.0])
    assert arrhenius(t, 1.0, 0.1) == pytest.approx(np.exp(-0.1 / (units.BOLTZMANN * t)))


def test_errors_cover_the_truth():
    """Of many noisy data, about 68 % give parameters within one error."""
    rng = np.random.default_rng(4)
    t = np.linspace(0, 5, 30)
    inside = 0
    for _ in range(300):
        y = 2 * np.exp(-t / 1.3) + rng.normal(0, 0.05, 30)
        fit = fit_curve(exponential, t, y)
        inside += abs(fit["time"] - 1.3) < fit.errors[1]
    assert 0.6 < inside / 300 < 0.76


@pytest.mark.parametrize("absolute", [False, True])
def test_sigma_as_scipy(absolute):
    rng = np.random.default_rng(5)
    x = np.linspace(0, 5, 30)
    sigma = 0.01 + 0.05 * rng.random(30)
    y = 2 * np.exp(-x / 1.3) + rng.normal(0, sigma)
    fit = fit_curve(exponential, x, y, sigma=sigma, absolute_sigma=absolute)
    p, errors = scipy_fit(
        exponential, x, y, [1, 1], sigma=sigma, absolute_sigma=absolute
    )
    np.testing.assert_allclose(fit.parameters, p, rtol=1e-6)
    np.testing.assert_allclose(fit.errors, errors, rtol=1e-3)


def test_a_function_of_two_variables_and_a_plain_function():
    rng = np.random.default_rng(6)
    xy = rng.random((2, 50)) * 3

    def plane(points, a, b, c):
        return a + b * points[0] + c * np.sin(points[1])

    z = plane(xy, 1.0, -2.0, 0.5) + rng.normal(0, 0.01, 50)
    fit = fit_curve(plane, xy, z, [0, 0, 0])
    p, errors = optimize.curve_fit(plane, xy, z, p0=[0, 0, 0])
    np.testing.assert_allclose(fit.parameters, p, rtol=1e-6)
    np.testing.assert_allclose(fit.errors, np.sqrt(np.diagonal(errors)), rtol=1e-3)
    assert fit.model.parameters == ("p0", "p1", "p2") and fit.model.name == "plane"
    own = Model("line", lambda x, a, b: a + b * x, ("intercept", "slope"))
    line = fit_curve(own, [0, 1, 2, 3], [1, 3, 5, 7], [0, 0])
    assert line["slope"] == pytest.approx(2) and line["intercept"] == pytest.approx(1)


def test_curve_errors():
    x = np.linspace(0, 1, 10)
    with pytest.raises(InvalidParameter, match="needs a guess"):
        fit_curve(lambda t, a: a * t, x, x)
    with pytest.raises(InvalidParameter, match="must be a Model or a function"):
        fit_curve(3, x, x, [1])
    with pytest.raises(InvalidParameter, match="needs a guess"):
        fit_curve(Model("line", lambda t, a: a * t, ("a",)), x, x)
    with pytest.raises(InvalidParameter, match="2 parameters, not 3"):
        fit_curve(exponential, x, x, [1, 1, 1])
    with pytest.raises(InvalidParameter, match="not finite at the guess"):
        fit_curve(exponential, x, x, [1, 0])
    with pytest.raises(InvalidParameter, match="did not converge within 1 steps"):
        fit_curve(gaussian, x, np.exp(-((x - 0.3) ** 2) / 0.01), [1, 5, 1], max_steps=1)
    with pytest.raises(KeyError):
        fit_curve(exponential, x, np.exp(-x))["width"]
