# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import pytest

from mdapy.errors import InvalidParameter
from mdapy.fitting import GridSpline, Spline

interpolate = pytest.importorskip("scipy.interpolate")


def points(count=12, seed=0):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.random(count)) * 5
    return x, np.sin(x) + 0.3 * x


@pytest.mark.parametrize("boundary", ["not-a-knot", "natural"])
@pytest.mark.parametrize("count", [3, 4, 12])
def test_as_scipy(boundary, count):
    x, y = points(count)
    ours, theirs = (
        Spline(x, y, boundary=boundary),
        interpolate.CubicSpline(x, y, bc_type=boundary),
    )
    at = np.linspace(x[0], x[-1], 301)
    for derivative in range(4):
        np.testing.assert_allclose(
            ours(at, derivative), theirs(at, derivative), atol=1e-10
        )
    assert ours.integral(x[1], x[-2]) == pytest.approx(
        theirs.integrate(x[1], x[-2]), abs=1e-12
    )
    assert ours.integral(x[-2], x[1]) == pytest.approx(
        -theirs.integrate(x[1], x[-2]), abs=1e-12
    )


def test_clamped_and_periodic_as_scipy():
    x, y = points()
    at = np.linspace(x[0], x[-1], 301)
    ours = Spline(x, y, boundary="clamped", slopes=(0.5, -1.0))
    theirs = interpolate.CubicSpline(x, y, bc_type=((1, 0.5), (1, -1.0)))
    for derivative in range(4):
        np.testing.assert_allclose(
            ours(at, derivative), theirs(at, derivative), atol=1e-10
        )
    wave = np.sin(2 * np.pi * (x - x[0]) / (x[-1] - x[0]))
    ours = Spline(x, wave, boundary="periodic")
    theirs = interpolate.CubicSpline(x, wave, bc_type="periodic")
    for derivative in range(4):
        np.testing.assert_allclose(
            ours(at, derivative), theirs(at, derivative), atol=1e-10
        )
    # beyond the points the curve repeats
    period = x[-1] - x[0]
    np.testing.assert_allclose(ours(at + 3 * period), ours(at), atol=1e-12)


def test_several_curves_at_once():
    x, y = points()
    curves = np.stack([y, 2 * y + 1, np.cos(x)], axis=1)
    together = Spline(x, curves)
    at = np.linspace(x[0], x[-1], 50)
    assert together(at).shape == (50, 3)
    assert together(2.0).shape == (3,)
    for k in range(3):
        np.testing.assert_allclose(
            together(at)[:, k], Spline(x, curves[:, k])(at), atol=1e-13
        )
    assert together.integral(x[0], x[-1]).shape == (3,)
    clamped = Spline(x, curves, boundary="clamped", slopes=[[0, 1, 2], [3, 4, 5]])
    np.testing.assert_allclose(clamped(x[0], 1), [0, 1, 2], atol=1e-12)
    np.testing.assert_allclose(clamped(x[-1], 1), [3, 4, 5], atol=1e-12)


def test_a_cubic_is_reproduced():
    x = np.array([0.0, 0.3, 1.1, 1.5, 2.6, 3.0])
    cubic = lambda t: 1 - 2 * t + 0.5 * t**2 + 0.25 * t**3  # noqa: E731
    spline = Spline(x, cubic(x))
    at = np.linspace(0, 3, 40)
    np.testing.assert_allclose(spline(at), cubic(at), atol=1e-12)
    np.testing.assert_allclose(spline(at, 3), 1.5, atol=1e-10)
    assert spline.integral(0, 3) == pytest.approx(3 - 9 + 4.5 + 81 / 16, rel=1e-12)
    # a line through two points
    np.testing.assert_allclose(Spline([0, 2], [1, 5])([0.5, 1.5]), [2, 4])


def test_outside_the_points():
    x, y = points()
    spline = Spline(x, y)
    values = spline([x[0] - 1, x[0], x[-1], x[-1] + 1])
    assert np.isnan(values[[0, 3]]).all() and np.isfinite(values[[1, 2]]).all()
    reaching = Spline(x, y, extrapolate=True)
    assert np.isfinite(reaching(x[-1] + 1))
    assert reaching.integral(x[0] - 1, x[0]) == pytest.approx(
        interpolate.CubicSpline(x, y).integrate(x[0] - 1, x[0]), abs=1e-10
    )
    with pytest.raises(InvalidParameter, match="within the points"):
        spline.integral(x[0] - 1, x[0])


def test_spline_errors():
    with pytest.raises(InvalidParameter, match="strictly increasing"):
        Spline([0, 2, 1], [1, 2, 3])
    with pytest.raises(InvalidParameter, match="two or more"):
        Spline([0], [1])
    with pytest.raises(InvalidParameter, match="x must be numbers"):
        Spline(["a", "b"], [1, 2])
    with pytest.raises(InvalidParameter, match="y must be numbers"):
        Spline([0, 1], ["a", "b"])
    with pytest.raises(InvalidParameter, match="3 values"):
        Spline([0, 1, 2], [1, 2])
    with pytest.raises(InvalidParameter, match="finite"):
        Spline([0, 1], [1, np.nan])
    with pytest.raises(InvalidParameter, match="boundary must be"):
        Spline([0, 1], [1, 2], boundary="free")
    with pytest.raises(InvalidParameter, match="needs slopes"):
        Spline([0, 1], [1, 2], boundary="clamped")
    with pytest.raises(InvalidParameter, match="clamped spline only"):
        Spline([0, 1], [1, 2], slopes=(0, 0))
    with pytest.raises(InvalidParameter, match="2 values, or 2"):
        Spline(
            [0, 1, 2],
            [[1, 2], [3, 4], [5, 6]],
            boundary="clamped",
            slopes=[1, 2, 3, 4, 5, 6],
        )
    with pytest.raises(InvalidParameter, match="equal first and last"):
        Spline([0, 1, 2], [1, 2, 3], boundary="periodic")
    with pytest.raises(InvalidParameter, match="derivative must be"):
        Spline([0, 1], [1, 2])(0.5, 4)
    assert repr(Spline([0, 1], [1, 2])) == "Spline(2 points, not-a-knot)"


def grid(seed=0):
    rng = np.random.default_rng(seed)
    x, y = np.linspace(0, 3, 7), np.sort(rng.random(9)) * 2
    z = np.sin(x[:, None]) * np.cos(y[None, :]) + x[:, None] * y[None, :]
    return x, y, z


def tensor(x, y, z, px, py, dx=0, dy=0, bc="not-a-knot"):
    """The tensor product of SciPy's splines: along x, then along y."""
    return np.array(
        [
            interpolate.CubicSpline(
                y, interpolate.CubicSpline(x, z, bc_type=bc)(a, dx), bc_type=bc
            )(b, dy)
            for a, b in zip(px, py)
        ]
    )


def test_grid_as_scipy():
    x, y, z = grid()
    rng = np.random.default_rng(1)
    px, py = rng.random(60) * 3, y[0] + rng.random(60) * (y[-1] - y[0])
    surface = GridSpline(x, y, z)
    np.testing.assert_allclose(surface(px, py), tensor(x, y, z, px, py), atol=1e-12)
    rect = interpolate.RectBivariateSpline(x, y, z, s=0)
    np.testing.assert_allclose(surface(px, py), rect(px, py, grid=False), atol=1e-12)
    for dx, dy in ((1, 0), (0, 2), (1, 1), (3, 3)):
        np.testing.assert_allclose(
            surface(px, py, dx=dx, dy=dy), tensor(x, y, z, px, py, dx, dy), atol=1e-9
        )
    # through every value of the grid
    gx, gy = np.meshgrid(x, y, indexing="ij")
    np.testing.assert_allclose(surface(gx, gy), z, atol=1e-12)


def test_periodic_grid():
    x, y, _ = grid()
    z = np.sin(2 * np.pi * x[:, None] / 3) * np.cos(
        2 * np.pi * (y[None, :] - y[0]) / (y[-1] - y[0])
    )
    surface = GridSpline(x, y, z, boundary="periodic")
    rng = np.random.default_rng(2)
    px, py = rng.random(40) * 3, y[0] + rng.random(40) * (y[-1] - y[0])
    np.testing.assert_allclose(
        surface(px, py), tensor(x, y, z, px, py, bc="periodic"), atol=1e-12
    )
    np.testing.assert_allclose(
        surface(px + 3, py - 2 * (y[-1] - y[0])), surface(px, py), atol=1e-12
    )
    natural = GridSpline(x, y, z, boundary=("natural", "periodic"))
    assert natural.boundary == ("natural", "periodic")


def test_a_bicubic_is_reproduced_and_the_outside():
    x, y = np.array([0.0, 0.5, 1.5, 2.0, 3.0]), np.array([-1.0, 0.0, 0.4, 1.0, 2.0])
    f = lambda a, b: 1 + a * b - a**3 + 2 * b**3 * a**2  # noqa: E731
    surface = GridSpline(x, y, f(x[:, None], y[None, :]))
    px, py = np.linspace(0, 3, 9), np.linspace(-1, 2, 9)
    np.testing.assert_allclose(surface(px, py), f(px, py), atol=1e-11)
    np.testing.assert_allclose(
        surface(px, py, dx=1, dy=1), 1 + 12 * py**2 * px, atol=1e-9
    )
    assert np.isnan(surface([-0.1, 1.0], [0.0, 2.5])).all()
    assert np.isfinite(
        GridSpline(x, y, f(x[:, None], y[None, :]), extrapolate=True)(-0.1, 2.5)
    )
    assert surface(np.ones((2, 3)), 0.5).shape == (2, 3)
    assert repr(surface) == "GridSpline(5 x 5 points)"


def test_grid_errors():
    x, y, z = grid()
    with pytest.raises(InvalidParameter, match=r"z must be \(7, 9\)"):
        GridSpline(x, y, z[:, :3])
    with pytest.raises(InvalidParameter, match="z must be numbers"):
        GridSpline(x, y, [["a"]])
    with pytest.raises(InvalidParameter, match="z must be finite"):
        GridSpline(x, y, np.full_like(z, np.inf))
    with pytest.raises(InvalidParameter, match="boundary must be"):
        GridSpline(x, y, z, boundary="clamped")
    with pytest.raises(InvalidParameter, match="boundary must be"):
        GridSpline(x, y, z, boundary=("natural",))
    with pytest.raises(InvalidParameter, match="dx must be"):
        GridSpline(x, y, z)(1.0, 1.0, dx=4)
