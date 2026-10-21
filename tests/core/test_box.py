# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

import itertools

import numpy as np
import pytest

from mdapy import Box
from mdapy.errors import InvalidParameter


def random_lammps_like_box(rng):
    """Restricted triclinic box with tilts within half the edge lengths, rotated."""
    lx, ly, lz = rng.uniform(3, 12, size=3)
    xy, xz = rng.uniform(-0.5, 0.5, size=2) * lx
    yz = rng.uniform(-0.5, 0.5) * ly
    m = np.array([[lx, 0, 0], [xy, ly, 0], [xz, yz, lz]])
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return m @ q.T


@pytest.fixture
def rng():
    return np.random.default_rng(2026)


# --- construction ----------------------------------------------------------


def test_construction_shortcuts():
    np.testing.assert_array_equal(Box(4.0).matrix, 4.0 * np.eye(3))
    np.testing.assert_array_equal(Box([1, 2, 3]).matrix, np.diag([1.0, 2.0, 3.0]))
    b = Box(np.eye(3), origin=[1, 2, 3], pbc=[True, False, True])
    np.testing.assert_array_equal(b.origin, [1, 2, 3])
    np.testing.assert_array_equal(b.pbc, [True, False, True])
    np.testing.assert_array_equal(Box(1.0, pbc=False).pbc, [False] * 3)
    np.testing.assert_array_equal(Box(1.0, pbc=[1, 0, 1]).pbc, [True, False, True])


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"matrix": [[1, 0], [0, 1]]}, "shape"),
        ({"matrix": "abc"}, "numeric"),
        ({"matrix": [1, 2, np.nan]}, "finite"),
        ({"matrix": [[1, 0, 0], [2, 0, 0], [0, 0, 1]]}, "linearly dependent"),
        ({"matrix": 0.0}, "linearly dependent"),
        ({"matrix": 1.0, "origin": [0, 0]}, "origin"),
        ({"matrix": 1.0, "origin": "xyz"}, "origin"),
        ({"matrix": 1.0, "origin": [0, np.inf, 0]}, "origin"),
        ({"matrix": 1.0, "pbc": [True, False]}, "pbc"),
        ({"matrix": 1.0, "pbc": [2, 0, 1]}, "pbc"),
        ({"matrix": 1.0, "pbc": ["T", "T", "T"]}, "pbc"),
        ({"matrix": 1.0, "pbc": 1.0}, "pbc"),
    ],
)
def test_invalid_construction(kwargs, message):
    with pytest.raises(InvalidParameter, match=message):
        Box(**kwargs)


def test_arrays_are_read_only():
    b = Box(2.0)
    for array in (b.matrix, b.origin, b.pbc, b.inverse):
        with pytest.raises(ValueError):
            array[0] = 0


def test_input_is_copied():
    m = np.eye(3)
    b = Box(m)
    m[0, 0] = 5.0
    assert b.matrix[0, 0] == 1.0


# --- geometry ----------------------------------------------------------------


def test_volume_inverse_and_lengths(rng):
    for _ in range(20):
        m = random_lammps_like_box(rng)
        b = Box(m)
        assert b.volume == pytest.approx(abs(np.linalg.det(m)), rel=1e-13)
        np.testing.assert_allclose(b.matrix @ b.inverse, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(b.lengths, np.linalg.norm(m, axis=1), rtol=1e-14)


def test_thickness_is_distance_between_faces(rng):
    for _ in range(20):
        b = Box(random_lammps_like_box(rng))
        a, bb, c = b.matrix
        expected = [
            b.volume / np.linalg.norm(np.cross(bb, c)),
            b.volume / np.linalg.norm(np.cross(a, c)),
            b.volume / np.linalg.norm(np.cross(a, bb)),
        ]
        np.testing.assert_allclose(b.thickness, expected, rtol=1e-12)
    np.testing.assert_allclose(Box([2, 3, 4]).thickness, [2, 3, 4])


def test_lengths_and_angles_round_trip(rng):
    for _ in range(50):
        lengths = rng.uniform(2, 10, size=3)
        while True:
            angles = rng.uniform(50, 130, size=3)
            try:
                b = Box.from_lengths_angles(*lengths, *angles)
                break
            except InvalidParameter:
                continue
        np.testing.assert_allclose(b.lengths, lengths, rtol=1e-12)
        np.testing.assert_allclose(b.angles, angles, rtol=1e-10)
        assert b.matrix[0, 1] == b.matrix[0, 2] == b.matrix[1, 2] == 0.0


def test_hexagonal_box():
    b = Box.from_lengths_angles(2.5, 2.5, 4.0, 90, 90, 120)
    np.testing.assert_allclose(
        b.matrix,
        [[2.5, 0, 0], [-1.25, 2.5 * np.sqrt(3) / 2, 0], [0, 0, 4.0]],
        atol=1e-14,
    )
    np.testing.assert_allclose(b.angles, [90, 90, 120], atol=1e-12)


@pytest.mark.parametrize(
    "args",
    [(0, 1, 1), (1, 1, 1, 90, 90, 0), (1, 1, 1, 10, 10, 120), (1, 1, 1, 150, 150, 150)],
)
def test_invalid_lengths_angles(args):
    with pytest.raises(InvalidParameter):
        Box.from_lengths_angles(*args)


def test_is_orthogonal():
    assert Box([1, 2, 3]).is_orthogonal
    assert not Box.from_lengths_angles(1, 1, 1, 90, 90, 100).is_orthogonal
    assert not Box([-1, 2, 3]).is_orthogonal


# --- coordinates -------------------------------------------------------------


def test_fractional_round_trip(rng):
    b = Box(random_lammps_like_box(rng), origin=rng.normal(size=3))
    r = rng.normal(scale=20, size=(100, 3))
    np.testing.assert_allclose(b.to_cartesian(b.to_fractional(r)), r, atol=1e-12)
    np.testing.assert_allclose(
        b.to_fractional(b.origin + b.matrix[1]), [0, 1, 0], atol=1e-14
    )


@pytest.mark.parametrize("pbc", [True, [True, False, True], False])
def test_wrap(rng, pbc):
    b = Box(random_lammps_like_box(rng), origin=rng.normal(size=3), pbc=pbc)
    r = b.to_cartesian(rng.uniform(-5, 5, size=(500, 3)))
    wrapped, image = b.wrap(r)
    frac = b.to_fractional(wrapped)
    p = b.pbc
    assert np.all(frac[:, p] >= 0) and np.all(frac[:, p] < 1)
    assert np.all(image[:, ~p] == 0)
    np.testing.assert_allclose(frac[:, ~p], b.to_fractional(r)[:, ~p], atol=1e-12)
    np.testing.assert_allclose(wrapped + image @ b.matrix, r, atol=1e-10)
    assert image.dtype == np.int64


def test_wrap_never_returns_the_upper_face():
    b = Box(1.0)
    wrapped, image = b.wrap([[-1e-17, 0.5, 1.0], [0.0, 0.0, 0.0]])
    assert np.all(b.to_fractional(wrapped) < 1.0)
    np.testing.assert_array_equal(image[1], [0, 0, 0])


def test_minimum_image_matches_brute_force(rng):
    images = np.array(list(itertools.product(range(-3, 4), repeat=3)), dtype=float)
    for _ in range(30):
        b = Box(random_lammps_like_box(rng))
        # within +-2 box lengths, so +-3 images surely contain the shortest one
        d = rng.uniform(-2, 2, size=(50, 3)) @ b.matrix
        got = b.minimum_image(d)
        candidates = d[:, None, :] + images @ b.matrix
        expected = np.min(np.linalg.norm(candidates, axis=2), axis=1)
        np.testing.assert_allclose(np.linalg.norm(got, axis=1), expected, rtol=1e-12)
        # the result differs from the input by a lattice vector
        shift = b.to_fractional(got) - b.to_fractional(d)
        np.testing.assert_allclose(shift, np.round(shift), atol=1e-9)


def test_minimum_image_leaves_non_periodic_directions(rng):
    b = Box(random_lammps_like_box(rng), pbc=[True, True, False])
    d = rng.normal(scale=15, size=(50, 3))
    frac_in = d @ b.inverse
    frac_out = b.minimum_image(d) @ b.inverse
    np.testing.assert_allclose(frac_out[:, 2], frac_in[:, 2], atol=1e-12)
    np.testing.assert_allclose(Box(1.0, pbc=False).minimum_image(d), d)


# --- LAMMPS ----------------------------------------------------------------------


def test_to_lammps_preserves_geometry(rng):
    for _ in range(30):
        b = Box(random_lammps_like_box(rng), origin=rng.normal(size=3))
        lmp = b.to_lammps()
        rot = lmp.rotation
        np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(rot) == pytest.approx(1.0)
        restricted = Box.from_lammps(lmp.lo, lmp.hi, lmp.tilt)
        assert np.all(lmp.hi > lmp.lo)
        np.testing.assert_allclose(b.matrix @ rot, restricted.matrix, atol=1e-10)
        np.testing.assert_allclose(restricted.lengths, b.lengths, rtol=1e-12)
        np.testing.assert_allclose(restricted.angles, b.angles, rtol=1e-10)
        # positions transform with the same rotation
        r = b.to_cartesian(rng.uniform(0, 1, size=(10, 3)))
        np.testing.assert_allclose(
            restricted.to_fractional(r @ rot), b.to_fractional(r), atol=1e-10
        )


def test_to_lammps_is_exact_for_restricted_boxes():
    b = Box([[3.0, 0, 0], [0.7, 4.0, 0], [-0.2, 0.3, 5.0]], origin=[1, 2, 3])
    lmp = b.to_lammps()
    np.testing.assert_array_equal(lmp.rotation, np.eye(3))
    np.testing.assert_array_equal(lmp.lo, [1, 2, 3])
    np.testing.assert_array_equal(lmp.hi, [4, 6, 8])
    np.testing.assert_array_equal(lmp.tilt, [0.7, -0.2, 0.3])
    assert Box.from_lammps(lmp.lo, lmp.hi, lmp.tilt, pbc=True) == b


def test_to_lammps_left_handed_box():
    b = Box([[3.0, 0, 0], [0, 0, 5.0], [0, 4.0, 0]])  # det < 0
    lmp = b.to_lammps()
    restricted = Box.from_lammps(lmp.lo, lmp.hi, lmp.tilt)
    assert restricted.volume == pytest.approx(b.volume)
    assert np.linalg.det(restricted.matrix) > 0
    # same lattice: each original vector is an integer combination of the new ones
    frac = (b.matrix @ lmp.rotation) @ restricted.inverse
    np.testing.assert_allclose(frac, np.round(frac), atol=1e-12)


# --- enclosing box, comparison -----------------------------------------------


def test_enclosing_box(rng):
    r = rng.normal(scale=5, size=(40, 3))
    b = Box.enclosing(r, padding=2.0)
    assert not b.pbc.any() and b.is_orthogonal
    frac = b.to_fractional(r)
    assert np.all(frac > 0) and np.all(frac < 1)
    np.testing.assert_allclose(b.origin, r.min(axis=0) - 2.0)
    planar = Box.enclosing([[0, 0, 0], [1, 1, 0]])  # flat molecule: still a 3D box
    assert planar.volume > 0
    assert Box.enclosing(np.empty((0, 3))).volume == pytest.approx(8.0)
    with pytest.raises(InvalidParameter, match="padding"):
        Box.enclosing(r, padding=0.0)


def test_equality_hash_and_allclose():
    a = Box([[1, 0, 0], [0, 1, 0], [0, 0, 1]], origin=[0.0, -0.0, 0.0])
    b = Box(1.0)
    assert a == b and hash(a) == hash(b)
    assert a != Box(1.0, pbc=False)
    assert a != Box(1.0 + 1e-12)
    assert a.allclose(Box(1.0 + 1e-12))
    assert not a.allclose(Box(1.0, pbc=[True, True, False]))
    assert (a == "box") is False
    assert len({a, b, Box(2.0)}) == 2


def test_repr_round_trips_through_eval():
    b = Box([[1.5, 0, 0], [0.25, 2, 0], [0, 0, 3]], origin=[1, 2, 3], pbc=[1, 0, 1])
    text = repr(b)
    assert text.startswith("Box(matrix=")
    assert eval(text, {"Box": Box}).allclose(b)


def test_wrap_and_minimum_image_keep_input_precision(rng):
    """Subtracting whole lattice vectors must not add rounding error."""
    lengths = np.array([10.0, 8.0, 12.0])
    b = Box(lengths)
    base = rng.integers(0, 64, size=(100, 3)) * (lengths / 64)  # exactly representable
    shifts = rng.integers(-5, 6, size=(100, 3))
    wrapped, image = b.wrap(base + shifts * lengths)
    np.testing.assert_array_equal(wrapped, base)
    np.testing.assert_array_equal(image, shifts)
    np.testing.assert_array_equal(
        b.minimum_image([[9.0, -7.0, 13.0]]), [[-1.0, 1.0, 1.0]]
    )
