# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import itertools
import math

import numpy as np
import pytest

import mdapy
from mdapy.build import build_crystal
from mdapy.errors import Cancelled, InvalidParameter
from mdapy.mlip import QUESTS, entropy, information_gain
from mdapy.neighbor import NeighborList
from mdapy.transform import replicate

# numba, compiling QUESTS, warns of its casts on some platforms
QUESTS_WARNINGS = pytest.mark.filterwarnings("ignore:.*unsafe cast:Warning")
TRICLINIC = np.array([[9.0, 0, 0], [2.0, 8.5, 0], [-1.5, 1.0, 9.5]])


def scattered(pbc=True, seed=0, n=60, closest=1.8):
    """Atoms at random in a triclinic box, no two closer than closest."""
    rng = np.random.default_rng(seed)
    box = mdapy.Box(TRICLINIC, origin=[1.0, -2.0, 0.5], pbc=pbc)
    points = []
    while len(points) < n:
        p = box.to_cartesian(rng.random((1, 3)))[0]
        steps = box.to_fractional(p - np.array(points).reshape(-1, 3) + box.origin)
        steps -= np.round(steps) * np.asarray(box.pbc)
        if (
            not len(points)
            or np.linalg.norm(steps @ box.matrix, axis=1).min() >= closest
        ):
            points.append(p)
    return mdapy.System(np.array(points), box, elements=["Al"] * n)


def shaken_aluminium(seed=1, repeat=(3, 3, 3)):
    """fcc aluminium strained and shaken, so that no two neighbors of an
    atom are at the same distance."""
    rng = np.random.default_rng(seed)
    crystal = build_crystal("fcc", 4.05, elements="Al", repeat=repeat)
    box = mdapy.Box(crystal.box.matrix @ (np.eye(3) + rng.normal(0, 0.03, (3, 3))))
    strained = crystal.with_box(box, scale_positions=True)
    return strained.with_columns(
        positions=strained["positions"] + rng.normal(0, 0.1, (len(crystal), 3))
    )


def brute_force(system, cutoff, k):
    """The descriptor of QUESTS from every periodic image in reach, in
    NumPy."""
    box = system.box
    x = system["positions"]
    reach = [
        int(math.ceil(cutoff / spacing)) if periodic else 0
        for spacing, periodic in zip(box.thickness, box.pbc, strict=True)
    ]
    images = np.array(
        list(itertools.product(*[range(-r, r + 1) for r in reach])), dtype=float
    )
    out = np.zeros((len(system), 2 * k - 1))
    for i in range(len(system)):
        vectors, keys = [], []
        for j in range(len(system)):
            for shift in images:
                if j == i and not shift.any():
                    continue
                d = x[j] + shift @ box.matrix - x[i]
                vectors.append(d)
                keys.append((np.linalg.norm(d), j, *shift))
        keys = np.array(keys)
        vectors = np.array(vectors)
        order = np.lexsort(keys.T[::-1])
        order = order[keys[order, 0] < cutoff][:k]
        r = keys[order, 0]
        d = vectors[order]
        w = (1 - (r / cutoff) ** 2) ** 2
        m = len(r)
        out[i, :m] = w / r
        cross = np.sqrt(np.outer(w, w)) / np.linalg.norm(
            d[:, None] - d[None, :], axis=2
        ).clip(1e-300)
        np.fill_diagonal(cross, -np.inf)
        rows = -np.sort(-cross, axis=1)[:, : m - 1]
        out[i, k : k + m - 1] = rows.sum(axis=0) / k
    return out


class TestDescriptor:
    @pytest.mark.parametrize(
        "pbc", [True, False, [True, False, True]], ids=["periodic", "free", "mixed"]
    )
    @pytest.mark.parametrize("cutoff, k", [(5.0, 32), (4.0, 8), (6.0, 50)])
    def test_as_brute_force(self, pbc, cutoff, k):
        system = scattered(pbc)
        assert np.abs(
            QUESTS(cutoff, k).descriptors(system) - brute_force(system, cutoff, k)
        ).max() == pytest.approx(0, abs=1e-13)

    def test_small_cell_reaches_far_images(self):
        """Two atoms in a cell much smaller than the cutoff."""
        system = mdapy.System(
            [[0.0, 0.0, 0.0], [1.3, 0.9, 1.1]],
            mdapy.Box([[2.6, 0, 0], [0.4, 2.4, 0], [0.3, -0.5, 2.8]]),
        )
        assert np.abs(
            QUESTS(5.0, 32).descriptors(system) - brute_force(system, 5.0, 32)
        ).max() == pytest.approx(0, abs=1e-13)

    @pytest.mark.parametrize("cutoff, k", [(5.0, 32), (4.0, 8), (6.5, 64), (3.0, 40)])
    @QUESTS_WARNINGS
    def test_as_quests(self, cutoff, k):
        """The descriptor of the QUESTS package, live; it rounds the
        fractional positions to 1e-12, so the values agree to about
        that."""
        pytest.importorskip("quests")
        from quests.descriptor import get_descriptors

        from mdapy.interop import to_ase

        for system in (shaken_aluminium(), scattered(), scattered(False)):
            ours = QUESTS(cutoff, k).descriptors(system)
            theirs = get_descriptors([to_ase(system)], k=k, cutoff=cutoff)
            assert ours.shape == theirs.shape == (len(system), 2 * k - 1)
            assert np.abs(ours - theirs).max() < 1e-10

    def test_invariances(self):
        system = scattered(seed=3)
        quests = QUESTS()
        x = quests.descriptors(system)
        # rotated, with its box
        angle = 0.7
        rotation = np.array(
            [
                [math.cos(angle), -math.sin(angle), 0],
                [math.sin(angle), math.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        rotated = mdapy.System(
            system["positions"] @ rotation.T,
            mdapy.Box(system.box.matrix @ rotation.T, origin=system.box.origin),
        )
        assert np.abs(quests.descriptors(rotated) - x).max() < 1e-13
        # moved and reordered
        order = np.random.default_rng(0).permutation(len(system))
        moved = system.gather(order).with_columns(
            positions=system["positions"][order] + [3.0, -1.0, 7.0]
        )
        assert np.abs(quests.descriptors(moved) - x[order]).max() < 1e-13
        # replicated: each copy of an atom has its environment
        copies = quests.descriptors(replicate(system, (2, 1, 2)))
        assert np.abs(copies - np.tile(x, (4, 1))).max() < 1e-13

    def test_missing_neighbors_are_zeros(self):
        """An atom alone has a zero descriptor; one with fewer neighbors
        than asked, zeros after its own."""
        box = mdapy.Box(np.eye(3) * 30, pbc=False)
        lone = mdapy.System([[1.0, 1.0, 1.0], [20.0, 20.0, 20.0], [22.0, 20, 20]], box)
        x = QUESTS(5.0, 4).descriptors(lone)
        assert not x[0].any()
        w = (1 - (2 / 5) ** 2) ** 2
        np.testing.assert_allclose(x[1], [w / 2, 0, 0, 0, 0, 0, 0], atol=1e-15)
        assert QUESTS(5.0, 1).descriptors(lone).shape == (3, 1)

    def test_neighbors_threads_and_repr(self):
        system = shaken_aluminium()
        quests = QUESTS(4.5, 20)
        x = quests.descriptors(system)
        longer = NeighborList.build(system, 6.0)
        assert np.array_equal(quests.descriptors(system, neighbors=longer), x)
        with mdapy.num_threads(1):
            assert np.array_equal(quests.descriptors(system), x)
        with pytest.raises(InvalidParameter, match="below the required"):
            quests.descriptors(system, neighbors=NeighborList.build(system, 4.0))
        assert quests.dimension == 39
        assert repr(quests) == "QUESTS(cutoff=4.5, num_neighbors=20)"

    def test_progress_and_cancel(self):
        system = shaken_aluminium()
        calls = []
        QUESTS().descriptors(system, progress=lambda d, t: calls.append((d, t)))
        assert calls and calls[-1][0] == calls[-1][1]
        token = mdapy.CancelToken()
        token.cancel()
        with pytest.raises(Cancelled):
            QUESTS().descriptors(system, cancel=token)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"cutoff": 0.0}, "cutoff"),
            ({"cutoff": math.inf}, "cutoff"),
            ({"cutoff": True}, "cutoff"),
            ({"cutoff": "5"}, "cutoff"),
            ({"num_neighbors": 0}, "num_neighbors"),
            ({"num_neighbors": 2.0}, "num_neighbors"),
            ({"num_neighbors": True}, "num_neighbors"),
        ],
    )
    def test_invalid(self, kwargs, match):
        with pytest.raises(InvalidParameter, match=match):
            QUESTS(**kwargs)


def kde_logs(x, y, h):
    """log sum_j exp(-|x_i - y_j|^2 / 2h^2), directly."""
    d2 = ((x[:, None, :] - y[None, :, :]) ** 2).sum(axis=2)
    return np.log(np.exp(-d2 / (2 * h * h)).sum(axis=1))


class TestEntropy:
    def test_limits(self):
        """Zero for copies of one point, log N for points far apart."""
        same = np.tile([0.1, 0.2, 0.3], (50, 1))
        assert entropy(same) == pytest.approx(0, abs=1e-14)
        apart = np.arange(40.0)[:, None] * [1.0, 0.0]
        assert entropy(apart) == pytest.approx(math.log(40), abs=1e-14)

    def test_as_numpy(self):
        rng = np.random.default_rng(2)
        x = rng.normal(0, 0.02, (300, 5))
        h = 0.015
        expected = -np.mean(kde_logs(x, x, h) - math.log(len(x)))
        assert entropy(x, h) == pytest.approx(expected, rel=1e-13)
        # scaled with its bandwidth, and reordered
        assert entropy(7 * x, 7 * h) == pytest.approx(expected, rel=1e-12)
        assert entropy(x[::-1], h) == pytest.approx(expected, rel=1e-13)

    @QUESTS_WARNINGS
    def test_as_quests(self):
        pytest.importorskip("quests")
        from quests.entropy import entropy as reference

        system = shaken_aluminium()
        x = QUESTS().descriptors(system)
        assert entropy(x) == pytest.approx(reference(x), rel=1e-12)
        assert entropy(x, 0.05) == pytest.approx(reference(x, 0.05), rel=1e-12)


class TestInformationGain:
    def test_one_reference_point(self):
        """The squared distance over twice the squared bandwidth, also far
        away, where the sum of the kernels is below the smallest double."""
        y = np.array([[0.0, 0.0, 0.0]])
        x = np.array([[0.01, 0.0, 0.0], [0.0, 3.0, 4.0]])
        gain = information_gain(x, y, 0.015)
        np.testing.assert_allclose(
            gain, [0.01**2 / (2 * 0.015**2), 25 / (2 * 0.015**2)], rtol=1e-13
        )
        assert np.isfinite(gain).all()

    def test_as_numpy(self):
        rng = np.random.default_rng(3)
        y = rng.normal(0, 0.03, (400, 6))
        x = np.concatenate([y[:5], rng.normal(0, 0.05, (60, 6))])
        gain = information_gain(x, y, 0.02)
        np.testing.assert_allclose(gain, -kde_logs(x, y, 0.02), rtol=1e-12)
        # a point of the reference has at least its own kernel
        assert (gain[:5] <= 0).all()
        with mdapy.num_threads(1):
            assert np.array_equal(information_gain(x, y, 0.02), gain)

    @QUESTS_WARNINGS
    def test_as_quests(self):
        pytest.importorskip("quests")
        from quests.entropy import delta_entropy

        reference = QUESTS().descriptors(shaken_aluminium(seed=1))
        new = QUESTS().descriptors(shaken_aluminium(seed=2))
        np.testing.assert_allclose(
            information_gain(new, reference),
            delta_entropy(new, reference),
            rtol=1e-11,
            atol=1e-12,
        )

    def test_progress_and_cancel(self):
        x = np.random.default_rng(4).random((200, 3))
        calls = []
        information_gain(x, x, progress=lambda d, t: calls.append((d, t)))
        assert calls and calls[-1][0] == calls[-1][1]
        token = mdapy.CancelToken()
        token.cancel()
        with pytest.raises(Cancelled):
            information_gain(x, x, cancel=token)
        with pytest.raises(Cancelled):
            entropy(x, cancel=token)

    @pytest.mark.parametrize(
        "points, reference, bandwidth, match",
        [
            (np.zeros(3), np.zeros((2, 3)), 0.1, "points must be"),
            (np.zeros((0, 3)), np.zeros((2, 3)), 0.1, "points must be"),
            (np.zeros((2, 3)), np.zeros((2, 4)), 0.1, "values each"),
            (
                np.zeros((2, 3)),
                np.full((2, 3), np.nan),
                0.1,
                "reference must be finite",
            ),
            (np.zeros((2, 3)), np.zeros((2, 3)), 0.0, "bandwidth"),
            (np.zeros((2, 3)), np.zeros((2, 3)), True, "bandwidth"),
            (np.zeros((2, 3)), np.zeros((2, 3)), math.nan, "bandwidth"),
        ],
    )
    def test_invalid(self, points, reference, bandwidth, match):
        with pytest.raises(InvalidParameter, match=match):
            information_gain(points, reference, bandwidth)
        if match not in ("values each", "reference must be finite"):
            with pytest.raises(InvalidParameter, match=match):
                entropy(points, bandwidth)
