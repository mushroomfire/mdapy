# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The basis of the plane-wave DFT on the GPU: the reciprocal cell; FFT grids
that hold their spheres; spheres of plane waves against a count in NumPy, as
the sphere of the density of silicon; k-points with time reversal;
the transforms between plane waves and the grid, their round trip, Parseval's
identity, real wavefunctions of Gamma, and functions carried from a coarse grid
to a fine one exactly."""

import itertools
import math

import numpy as np
import pytest

from dft import gpu

_dft = gpu.module()
if _dft.device_count() == 0:
    pytest.skip("no GPU", allow_module_level=True)

BOHR = 0.529177210903  # Å


def silicon():
    """The primitive cell of diamond silicon, 5.431 Å, in bohr."""
    half = 5.431 / 2 / BOHR
    return np.array([[0, half, half], [half, 0, half], [half, half, 0]], dtype=float)


def triclinic(seed=0):
    rng = np.random.default_rng(seed)
    return np.diag([7.0, 8.0, 9.0]) + rng.uniform(-1.5, 1.5, (3, 3))


def brute_sphere(a, k, cutoff):
    """Every m with |k + G|^2 / 2 <= cutoff, and their kinetic energies."""
    b, _ = _dft.reciprocal(a)
    reach = math.sqrt(2 * cutoff) + np.linalg.norm(np.asarray(k) @ b)
    bound = [
        int(math.ceil(reach * np.linalg.norm(a[i]) / (2 * math.pi))) + 1
        for i in range(3)
    ]
    found = {}
    for m in itertools.product(*(range(-n, n + 1) for n in bound)):
        q = (np.asarray(k) + m) @ b
        e = 0.5 * q @ q
        if e <= cutoff:
            found[m] = e
    return found


def test_the_reciprocal_cell():
    a = triclinic()
    b, volume = _dft.reciprocal(a)
    np.testing.assert_allclose(a @ b.T, 2 * math.pi * np.eye(3), atol=1e-12)
    assert volume == pytest.approx(abs(np.linalg.det(a)), rel=1e-14)
    with pytest.raises(ValueError, match="no volume"):
        _dft.reciprocal(np.zeros((3, 3)))


def test_good_sizes():
    assert [_dft.good_size(n) for n in (1, 11, 13, 17, 29, 31, 49, 55, 97)] == [
        1,
        12,
        15,
        18,
        30,
        32,
        50,
        60,
        100,
    ]
    for n in range(1, 300):
        m = _dft.good_size(n)
        assert m >= n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        assert m == 1
    # multiples of a factor, whose own primes count as good
    assert [_dft.good_size(n, f) for n, f in ((49, 4), (51, 4), (13, 3), (20, 7))] == [
        60,
        60,
        15,
        21,
    ]


@pytest.mark.parametrize("cell", ["silicon", "triclinic"])
def test_the_grid_of_the_largest_miller_index(cell):
    """Along each vector the good size of 2 max|m_i| + 1, max|m_i| that of the
    G of the sphere by brute force, as Quantum ESPRESSO chooses it."""
    a = silicon() if cell == "silicon" else triclinic(4)
    cutoff = 30.0
    most = np.abs(np.array(list(brute_sphere(a, [0, 0, 0], cutoff)))).max(axis=0)
    expected = [_dft.good_size(2 * int(m) + 1) for m in most]
    assert list(_dft.fft_grid(a, cutoff)) == expected


def test_the_density_of_silicon():
    """The sphere of the density of silicon at 80 hartree, the plane waves to
    20 (QE run live compares the grid and the plane waves in test_scf.py):
    every G of the sphere by brute force, and a grid that holds twice its
    largest Miller index in each direction."""
    a = silicon()
    grid = _dft.fft_grid(a, 80.0)
    miller, kinetic, box, partner = _dft.sphere(a, [0, 0, 0], 80.0, grid)
    assert {tuple(m) for m in miller} == set(brute_sphere(a, [0, 0, 0], 80.0))
    assert (np.asarray(grid) > 2 * np.abs(miller).max(axis=0)).all()
    assert (partner == -1).all()


@pytest.mark.parametrize("k", [[0, 0, 0], [0.25, -0.125, 0.4]])
def test_spheres_against_numpy(k):
    a = triclinic(1)
    cutoff = 6.0
    grid = _dft.fft_grid(a, 4 * cutoff)
    miller, kinetic, box, _ = _dft.sphere(a, k, cutoff, grid)
    expected = brute_sphere(a, k, cutoff)
    assert {tuple(m) for m in miller} == set(expected)
    np.testing.assert_allclose(
        kinetic, [expected[tuple(m)] for m in miller], rtol=1e-12
    )
    assert (np.diff(kinetic) >= 0).all()
    wrapped = np.mod(miller, grid)
    np.testing.assert_array_equal(
        box, (wrapped[:, 0] * grid[1] + wrapped[:, 1]) * grid[2] + wrapped[:, 2]
    )


def test_the_half_sphere_of_gamma():
    a = triclinic(2)
    grid = _dft.fft_grid(a, 20.0)
    full, *_ = _dft.sphere(a, [0, 0, 0], 5.0, grid)
    half, _, box, partner = _dft.sphere(a, [0, 0, 0], 5.0, grid, half=True)
    assert len(half) == (len(full) + 1) // 2
    halves = {tuple(m) for m in half}
    for m in full:  # each G or -G in the half, never both but for G = 0
        assert (tuple(m) in halves) != (tuple(-m) in halves) or not m.any()
    assert (half[:, 2] >= 0).all()
    last = grid[2] // 2 + 1
    plane = (half[:, 2] == 0) & half.any(axis=1)
    minus = np.mod(-half[plane], grid)
    np.testing.assert_array_equal(
        partner[plane], (minus[:, 0] * grid[1] + minus[:, 1]) * last
    )
    assert (partner[~plane] == -1).all()
    with pytest.raises(ValueError, match="Gamma only"):
        _dft.sphere(a, [0.5, 0, 0], 5.0, grid, half=True)


def test_a_grid_too_small_is_refused():
    a = silicon()
    with pytest.raises(ValueError, match="too small"):
        _dft.sphere(a, [0, 0, 0], 80.0, [20, 30, 30])


@pytest.mark.parametrize(("n", "offset", "count"),
                         [((4, 4, 4), (0, 0, 0), 36), ((4, 4, 4), (0.5, 0.5, 0.5), 32),
                          ((3, 3, 1), (0, 0, 0), 5), ((2, 3, 5), (0, 0.5, 0), 16)])  # fmt: skip
def test_k_points_with_time_reversal(n, offset, count):
    k, w = _dft.monkhorst_pack(n, offset)
    assert len(k) == count and w.sum() == pytest.approx(1.0)
    assert ((k >= -0.5) & (k < 0.5)).all()
    # every point of the full grid is a kept point or minus one, its weight
    full, wf = _dft.monkhorst_pack(n, offset, time_reversal=False)
    assert len(full) == np.prod(n) and np.allclose(wf, 1 / np.prod(n))
    total = np.zeros(len(k))
    for q in full:
        for i, p in enumerate(k):
            if np.allclose(np.mod(q - p + 0.5, 1) - 0.5, 0, atol=1e-9) or np.allclose(
                np.mod(q + p + 0.5, 1) - 0.5, 0, atol=1e-9
            ):
                total[i] += 1 / np.prod(n)
                break
    np.testing.assert_allclose(total, w, atol=1e-12)


def test_the_grid_of_a_spacing():
    a = triclinic(3)
    b, _ = _dft.reciprocal(a)
    for spacing in (0.1, 0.25, 0.6, 5.0):
        expected = [
            max(1, math.ceil(np.linalg.norm(b[i]) / spacing - 1e-12)) for i in range(3)
        ]
        assert list(_dft.grid_of_spacing(a, spacing)) == expected


def random_coefficients(n, count, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(count, n)) + 1j * rng.normal(size=(count, n))


def test_plane_waves_on_the_grid():
    a = triclinic(4)
    grid = _dft.fft_grid(a, 12.0)
    miller, _, box, _ = _dft.sphere(a, [0, 0, 0], 3.0, grid)
    pick = [0, 7, len(miller) - 1]
    c = np.zeros((3, len(miller)), dtype=complex)
    for band, g in enumerate(pick):
        c[band, g] = 1.0
    values = _dft.to_real_space(c, box, grid)
    j = np.stack(np.meshgrid(*(np.arange(n) for n in grid), indexing="ij"), axis=-1)
    for band, g in enumerate(pick):
        phase = 2 * math.pi * (j / np.asarray(grid)) @ miller[g]
        np.testing.assert_allclose(values[band], np.exp(1j * phase), atol=1e-12)


def test_the_round_trip_and_parseval():
    a = triclinic(5)
    grid = _dft.fft_grid(a, 16.0)
    _, _, box, _ = _dft.sphere(a, [0.1, 0.2, -0.3], 4.0, grid)
    c = random_coefficients(len(box), 5)
    values = _dft.to_real_space(c, box, grid)
    np.testing.assert_allclose(
        (np.abs(values) ** 2).sum(axis=(1, 2, 3)),
        np.prod(grid) * (np.abs(c) ** 2).sum(axis=1),
        rtol=1e-12,
    )
    np.testing.assert_allclose(_dft.to_coefficients(values, box), c, atol=1e-12)


def test_real_wavefunctions_of_gamma():
    a = triclinic(6)
    grid = _dft.fft_grid(a, 16.0)
    half, _, hbox, partner = _dft.sphere(a, [0, 0, 0], 4.0, grid, half=True)
    c = random_coefficients(len(half), 4, seed=1)
    c[:, 0] = c[:, 0].real  # G = 0 first: real
    values = _dft.to_real_space_gamma(c, hbox, partner, grid)
    # the same function from the whole sphere, c(-G) = conj(c(G))
    full, _, fbox, _ = _dft.sphere(a, [0, 0, 0], 4.0, grid)
    where = {tuple(m): i for i, m in enumerate(half)}
    whole = np.empty((4, len(full)), dtype=complex)
    for i, m in enumerate(full):
        whole[:, i] = (
            c[:, where[tuple(m)]]
            if tuple(m) in where
            else c[:, where[tuple(-m)]].conj()
        )
    np.testing.assert_allclose(
        values, _dft.to_real_space(whole, fbox, grid).real, atol=1e-11
    )
    np.testing.assert_allclose(_dft.to_coefficients_gamma(values, hbox), c, atol=1e-12)


def test_functions_carried_to_a_fine_grid():
    a = triclinic(7)
    coarse = _dft.fft_grid(a, 8.0)
    fine = [_dft.good_size(2 * n) for n in coarse]
    _, _, cbox, _ = _dft.sphere(a, [0, 0, 0], 2.0, coarse)
    _, _, fbox, _ = _dft.sphere(a, [0, 0, 0], 2.0, fine)
    c = random_coefficients(len(cbox), 3, seed=2)
    on_fine = _dft.to_fine_grid(_dft.to_real_space(c, cbox, coarse), fine)
    np.testing.assert_allclose(on_fine, _dft.to_real_space(c, fbox, fine), atol=1e-11)
