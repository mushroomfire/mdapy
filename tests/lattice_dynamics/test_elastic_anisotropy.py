# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Moduli of a crystal along directions, and its anisotropy index."""

import numpy as np
import pytest

from mdapy.errors import InvalidParameter
from mdapy.lattice_dynamics import ElasticResult

VOIGT = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def directions(count: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(count, 3))


def unit(n: np.ndarray) -> np.ndarray:
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def cubic(c11: float, c12: float, c44: float) -> np.ndarray:
    c = np.zeros((6, 6))
    c[:3, :3] = c12
    np.fill_diagonal(c, [c11, c11, c11, c44, c44, c44])
    return c


def hexagonal(c11, c12, c13, c33, c44) -> np.ndarray:
    c = np.zeros((6, 6))
    c[:2, :2] = [[c11, c12], [c12, c11]]
    c[:2, 2] = c[2, :2] = c13
    c[2, 2] = c33
    c[3, 3] = c[4, 4] = c44
    c[5, 5] = (c11 - c12) / 2
    return c


def triclinic(seed: int) -> np.ndarray:
    """A stiffness with no symmetry, positive definite."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(6, 6))
    return 30 * (a @ a.T) + 50 * np.eye(6)


def stress_vector(tensor: np.ndarray) -> np.ndarray:
    return np.array([tensor[a, b] for a, b in VOIGT])


def strain_tensor(voigt: np.ndarray) -> np.ndarray:
    """The tensor of a strain in Voigt order with engineering shears."""
    e = np.empty((3, 3))
    for k, (a, b) in enumerate(VOIGT):
        e[a, b] = e[b, a] = voigt[k] if a == b else voigt[k] / 2
    return e


def stiffness_tensor(c: np.ndarray) -> np.ndarray:
    full = np.empty((3, 3, 3, 3))
    for m, (i, j) in enumerate(VOIGT):
        for n, (k, m2) in enumerate(VOIGT):
            for p, q in {(i, j), (j, i)}:
                for r, s in {(k, m2), (m2, k)}:
                    full[p, q, r, s] = c[m, n]
    return full


def rotated(c: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    full = np.einsum(
        "ai,bj,ck,dl,ijkl->abcd",
        rotation,
        rotation,
        rotation,
        rotation,
        stiffness_tensor(c),
    )
    return np.array([[full[i, j, k, m] for k, m in VOIGT] for i, j in VOIGT])


def random_rotation(seed: int) -> np.ndarray:
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q


def test_an_isotropic_crystal_is_the_same_along_every_direction():
    lame, shear = 80.0, 45.0
    c = cubic(lame + 2 * shear, lame, shear)
    result = ElasticResult(c)
    n = directions(200)
    young = shear * (3 * lame + 2 * shear) / (lame + shear)
    bulk = lame + 2 * shear / 3
    np.testing.assert_allclose(result.young_modulus(n), young, rtol=1e-13)
    np.testing.assert_allclose(
        result.linear_compressibility(n), 1 / (3 * bulk), rtol=1e-13
    )
    assert abs(result.universal_anisotropy()) < 1e-13


@pytest.mark.parametrize("constants", [(168.6, 121.6, 75.4), (240.0, 140.0, 40.0)])
def test_cubic_crystals_follow_their_closed_form(constants):
    c = cubic(*constants)
    s = np.linalg.inv(c)
    result = ElasticResult(c)
    n = unit(directions(300, seed=1))
    l2 = n**2
    orientation = l2[:, 0] * l2[:, 1] + l2[:, 1] * l2[:, 2] + l2[:, 2] * l2[:, 0]
    expected = 1 / (s[0, 0] - 2 * (s[0, 0] - s[0, 1] - s[3, 3] / 2) * orientation)
    np.testing.assert_allclose(result.young_modulus(n), expected, rtol=1e-12)
    assert result.young_modulus([0, 0, 2]) == pytest.approx(1 / s[0, 0], rel=1e-13)
    assert result.young_modulus([1, 1, 1]) == pytest.approx(
        3 / (s[0, 0] + 2 * s[0, 1] + s[3, 3]), rel=1e-13
    )
    zener = 2 * constants[2] / (constants[0] - constants[1])
    assert result.universal_anisotropy() == pytest.approx(
        6 / 5 * (np.sqrt(zener) - 1 / np.sqrt(zener)) ** 2, rel=1e-12
    )


def test_hexagonal_crystals_follow_their_closed_form():
    c = hexagonal(162.4, 92.0, 69.0, 180.7, 46.7)  # titanium
    s = np.linalg.inv(c)
    result = ElasticResult(c)
    n = unit(directions(300, seed=2))
    z2 = n[:, 2] ** 2
    young = 1 / (
        (1 - z2) ** 2 * s[0, 0]
        + z2**2 * s[2, 2]
        + z2 * (1 - z2) * (2 * s[0, 2] + s[3, 3])
    )
    compressibility = (s[0, 0] + s[0, 1] + s[0, 2]) * (1 - z2) + (
        2 * s[0, 2] + s[2, 2]
    ) * z2
    np.testing.assert_allclose(result.young_modulus(n), young, rtol=1e-12)
    np.testing.assert_allclose(
        result.linear_compressibility(n), compressibility, rtol=1e-12
    )


@pytest.mark.parametrize("seed", [3, 4])
def test_any_crystal_matches_the_strain_under_a_stress(seed):
    c = triclinic(seed)
    s = np.linalg.inv(c)
    result = ElasticResult(c)
    n = unit(directions(50, seed=seed))
    young, compressibility = [], []
    for direction in n:
        # a stress of 1 GPa along the direction: the strain along it
        strain = strain_tensor(s @ stress_vector(np.outer(direction, direction)))
        young.append(1 / (direction @ strain @ direction))
        # a pressure of 1 GPa: the shortening along the direction
        strain = strain_tensor(s @ stress_vector(-np.eye(3)))
        compressibility.append(-(direction @ strain @ direction))
    np.testing.assert_allclose(result.young_modulus(n), young, rtol=1e-12)
    np.testing.assert_allclose(
        result.linear_compressibility(n), compressibility, rtol=1e-12
    )


@pytest.mark.parametrize("seed", [5, 6])
def test_the_moduli_turn_with_the_crystal(seed):
    c = triclinic(seed)
    rotation = random_rotation(seed)
    result, turned = ElasticResult(c), ElasticResult(rotated(c, rotation))
    n = directions(50, seed=seed)
    np.testing.assert_allclose(
        turned.young_modulus(n @ rotation.T), result.young_modulus(n), rtol=1e-12
    )
    np.testing.assert_allclose(
        turned.linear_compressibility(n @ rotation.T),
        result.linear_compressibility(n),
        rtol=1e-12,
    )
    assert turned.universal_anisotropy() == pytest.approx(
        result.universal_anisotropy(), rel=1e-12
    )


def test_the_compressibilities_of_three_axes_add_up_to_the_volume():
    result = ElasticResult(triclinic(7))
    axes = random_rotation(7)
    total = result.linear_compressibility(axes).sum()
    assert total == pytest.approx(1 / result.bulk_modulus("reuss"), rel=1e-12)


def test_shapes_and_lengths():
    result = ElasticResult(cubic(168.6, 121.6, 75.4))
    one = result.young_modulus([2.0, 0.0, 0.0])
    assert isinstance(one, float) and one == result.young_modulus((1, 0, 0))
    assert isinstance(result.linear_compressibility([0, 1, 0]), float)
    many = result.young_modulus(np.eye(3) * 5)
    assert many.shape == (3,) and many.dtype == np.float64
    assert result.linear_compressibility(np.eye(3)).shape == (3,)


@pytest.mark.parametrize(
    "direction",
    [
        [1, 0],
        np.ones((2, 4)),
        np.ones((2, 2, 3)),
        [0, 0, 0],
        [[1, 0, 0], [0, 0, 0]],
        [np.nan, 0, 1],
    ],
)
def test_errors(direction):
    result = ElasticResult(cubic(168.6, 121.6, 75.4))
    with pytest.raises(InvalidParameter):
        result.young_modulus(direction)
    with pytest.raises(InvalidParameter):
        result.linear_compressibility(direction)


def test_constants_that_cannot_be_inverted():
    c = cubic(168.6, 121.6, 75.4)
    c[2, :] = c[:, 2] = 0
    with pytest.raises(InvalidParameter, match="cannot be inverted"):
        ElasticResult(c).young_modulus([1, 0, 0])


def perpendicular(n: np.ndarray, seed: int) -> np.ndarray:
    """A direction across each of n, at random."""
    m = np.random.default_rng(seed).normal(size=n.shape)
    u = unit(n)
    return m - (m * u).sum(axis=-1, keepdims=True) * u


def test_an_isotropic_crystal_shears_and_contracts_alike_everywhere():
    lame, shear = 80.0, 45.0
    result = ElasticResult(cubic(lame + 2 * shear, lame, shear))
    n = directions(100, seed=8)
    m = perpendicular(n, seed=9)
    poisson = lame / (2 * (lame + shear))
    np.testing.assert_allclose(
        result.shear_modulus(normal=n, direction=m), shear, rtol=1e-13
    )
    np.testing.assert_allclose(result.poisson_ratio(n, m), poisson, rtol=1e-13)
    young = shear * (3 * lame + 2 * shear) / (lame + shear)
    for average in ("voigt", "reuss", "hill"):
        assert result.young_modulus(average=average) == pytest.approx(young, rel=1e-13)
        assert result.poisson_ratio(average=average) == pytest.approx(
            poisson, rel=1e-13
        )


@pytest.mark.parametrize("constants", [(168.6, 121.6, 75.4), (240.0, 140.0, 40.0)])
def test_cubic_crystals_shear_and_contract_by_their_closed_form(constants):
    c11, c12, c44 = constants
    s = np.linalg.inv(cubic(*constants))
    result = ElasticResult(cubic(*constants))
    n = unit(directions(300, seed=10))
    m = unit(perpendicular(n, seed=11))
    k = s[0, 0] - s[0, 1] - s[3, 3] / 2
    mixed = (n**2 * m**2).sum(axis=1)
    l2 = n**2
    orientation = l2[:, 0] * l2[:, 1] + l2[:, 1] * l2[:, 2] + l2[:, 2] * l2[:, 0]
    np.testing.assert_allclose(
        result.shear_modulus(normal=n, direction=m),
        1 / (s[3, 3] + 4 * k * mixed),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        result.poisson_ratio(n, m),
        -(s[0, 1] + k * mixed) / (s[0, 0] - 2 * k * orientation),
        rtol=1e-11,
    )
    # the cube shear, the shear C' and that of the slip systems of fcc
    assert result.shear_modulus(normal=[0, 0, 1], direction=[1, 0, 0]) == pytest.approx(
        c44
    )
    assert result.shear_modulus(
        normal=[1, 1, 0], direction=[1, -1, 0]
    ) == pytest.approx((c11 - c12) / 2, rel=1e-12)
    assert result.shear_modulus(
        normal=[1, 1, 1], direction=[1, -1, 0]
    ) == pytest.approx(3 * c44 * (c11 - c12) / (4 * c44 + c11 - c12), rel=1e-12)
    assert result.poisson_ratio([1, 0, 0], [0, 1, 0]) == pytest.approx(
        -s[0, 1] / s[0, 0]
    )


@pytest.mark.parametrize("seed", [12, 13])
def test_any_crystal_shears_and_contracts_as_its_strain_under_a_stress(seed):
    c = triclinic(seed)
    s = np.linalg.inv(c)
    result = ElasticResult(c)
    n = unit(directions(40, seed=seed))
    m = unit(perpendicular(n, seed=seed + 100))
    shear, poisson = [], []
    for a, b in zip(n, m):
        # a shear stress of 1 GPa on the plane a along b: the engineering strain
        strain = strain_tensor(s @ stress_vector(np.outer(a, b) + np.outer(b, a)))
        shear.append(1 / (2 * a @ strain @ b))
        # a stretch along a: the contraction along b over the stretch
        strain = strain_tensor(s @ stress_vector(np.outer(a, a)))
        poisson.append(-(b @ strain @ b) / (a @ strain @ a))
    np.testing.assert_allclose(
        result.shear_modulus(normal=n, direction=m), shear, rtol=1e-12
    )
    np.testing.assert_allclose(result.poisson_ratio(n, m), poisson, rtol=1e-11)


@pytest.mark.parametrize("seed", [14, 15])
def test_shear_and_poisson_turn_with_the_crystal(seed):
    c = triclinic(seed)
    rotation = random_rotation(seed)
    result, turned = ElasticResult(c), ElasticResult(rotated(c, rotation))
    n = directions(40, seed=seed)
    m = perpendicular(n, seed=seed + 1)
    np.testing.assert_allclose(
        turned.shear_modulus(normal=n @ rotation.T, direction=m @ rotation.T),
        result.shear_modulus(normal=n, direction=m),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        turned.poisson_ratio(n @ rotation.T, m @ rotation.T),
        result.poisson_ratio(n, m),
        rtol=1e-11,
    )
    for average in ("voigt", "reuss", "hill"):
        assert turned.young_modulus(average=average) == pytest.approx(
            result.young_modulus(average=average), rel=1e-12
        )


@pytest.mark.parametrize("average", ["voigt", "reuss", "hill"])
def test_the_averages_of_a_polycrystal(average):
    result = ElasticResult(triclinic(16))
    bulk = result.bulk_modulus(average)
    shear = result.shear_modulus(average)
    assert result.young_modulus(average=average) == pytest.approx(
        9 * bulk * shear / (3 * bulk + shear), rel=1e-14
    )
    assert result.poisson_ratio(average=average) == pytest.approx(
        (3 * bulk - 2 * shear) / (2 * (3 * bulk + shear)), rel=1e-14
    )
    assert isinstance(result.young_modulus(average=average), float)


def test_shapes_of_pairs():
    result = ElasticResult(cubic(168.6, 121.6, 75.4))
    one = result.shear_modulus(normal=[0, 0, 3], direction=[2, 0, 0])
    assert isinstance(one, float) and one == pytest.approx(75.4)
    assert isinstance(result.poisson_ratio([1, 1, 0], [0, 0, 1]), float)
    many = result.poisson_ratio(np.eye(3), np.roll(np.eye(3), 1, axis=0))
    assert many.shape == (3,) and many.dtype == np.float64


@pytest.mark.parametrize(
    "call, message",
    [
        (lambda r: r.young_modulus([1, 0, 0], average="voigt"), "polycrystal"),
        (
            lambda r: r.shear_modulus("reuss", normal=[0, 0, 1], direction=[1, 0, 0]),
            "polycrystal",
        ),
        (
            lambda r: r.poisson_ratio([1, 0, 0], [0, 1, 0], average="voigt"),
            "polycrystal",
        ),
        (lambda r: r.shear_modulus(normal=[0, 0, 1]), "both"),
        (lambda r: r.shear_modulus(direction=[1, 0, 0]), "both"),
        (lambda r: r.poisson_ratio([1, 0, 0]), "both"),
        (lambda r: r.poisson_ratio(transverse=[1, 0, 0]), "both"),
        (
            lambda r: r.shear_modulus(normal=[1, 1, 0], direction=[1, 0, 0]),
            "perpendicular",
        ),
        (lambda r: r.poisson_ratio([1, 0, 0], [[0, 1, 0], [0, 0, 1]]), "same shape"),
        (lambda r: r.poisson_ratio([1, 0, 0], [0, 0, 0]), "zero"),
        (lambda r: r.young_modulus(average="mean"), "average"),
        (lambda r: r.poisson_ratio(average="mean"), "average"),
    ],
)
def test_errors_of_pairs_and_averages(call, message):
    with pytest.raises(InvalidParameter, match=message):
        call(ElasticResult(cubic(168.6, 121.6, 75.4)))
