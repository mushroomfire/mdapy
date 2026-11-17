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
from mdapy.calculators import EAM
from mdapy.elements import COVALENT_RADII, SYMBOLS
from mdapy.errors import Cancelled, InvalidParameter, MissingProperty
from mdapy.mlip import generate_structures, perturb
from mdapy.mlip._generation import _BASIS, _Entropy, _Generator
from mdapy.transform import replicate


def radius(element):
    return COVALENT_RADII[SYMBOLS.index(element)]


def images(system, reach):
    """Every pair within reach over the periodic images: owner, other and
    vector."""
    box = system.box
    x = system["positions"]
    counts = [
        int(math.ceil(reach / t)) if p else 0
        for t, p in zip(box.thickness, box.pbc, strict=True)
    ]
    shifts = np.array(
        list(itertools.product(*[range(-c, c + 1) for c in counts])), dtype=float
    )
    pairs = []
    for i in range(len(x)):
        for j in range(len(x)):
            for s in shifts:
                if i == j and not s.any():
                    continue
                d = x[j] + s @ box.matrix - x[i]
                if np.linalg.norm(d) < reach:
                    pairs.append((i, j, d))
    return pairs


def chebyshev(s, cutoff, count):
    x = 2 * (s / cutoff - 1) ** 2 - 1
    fc = 0.5 * (1 + np.cos(np.pi * s / cutoff))
    return np.array([0.5 * (np.cos(k * np.arccos(x)) + 1) * fc for k in range(count)])


def brute_force(system, elements):
    """The descriptor of the generation, from its definition."""
    radial, angular, l_max, chemical, cutoff, angular_cutoff = _BASIS
    radii = np.array([radius(e) for e in system["element"]])
    kinds = [list(elements).index(e) for e in system["element"]]
    several = len(elements) > 1
    width = radial + angular * l_max + (chemical * len(elements) if several else 0)
    out = np.zeros((len(system), width))
    near = {i: [] for i in range(len(system))}
    for i, j, d in images(system, 2 * radii.max() * cutoff):
        r = np.linalg.norm(d)
        s = r / (radii[i] + radii[j])
        if s >= cutoff:
            continue
        f = chebyshev(s, cutoff, max(radial, chemical))
        out[i, :radial] += f[:radial]
        if several:
            start = radial + angular * l_max + kinds[j] * chemical
            out[i, start : start + chemical] += f[:chemical]
        if s < angular_cutoff:
            near[i].append((d / r, chebyshev(s, angular_cutoff, angular)))
    for i, pairs in near.items():
        for (u, f), (v, g) in itertools.product(pairs, pairs):
            c = np.clip(u @ v, -1, 1)
            for n in range(angular):
                for degree in range(1, l_max + 1):
                    p = np.polynomial.legendre.legval(c, [0] * degree + [1])
                    out[i, radial + n * l_max + degree - 1] += f[n] * g[n] * p
    return out


def scattered(elements, n=12, seed=0, pbc=True):
    """Atoms of the elements at random in a small triclinic box, apart."""
    rng = np.random.default_rng(seed)
    box = mdapy.Box([[7.0, 0, 0], [1.5, 6.5, 0], [-1.0, 0.8, 7.5]], pbc=pbc)
    points = []
    while len(points) < n:
        p = box.to_cartesian(rng.random((1, 3)))[0]
        f = box.to_fractional(p - np.array(points).reshape(-1, 3) + box.origin)
        f -= np.round(f) * np.asarray(box.pbc)
        if not points or np.linalg.norm(f @ box.matrix, axis=1).min() > 1.9:
            points.append(p)
    return mdapy.System(
        np.array(points), box, elements=list(rng.choice(elements, size=n))
    )


def generator(elements):
    return _Generator(tuple(elements), (2, 8), (10.0, 30.0), 0.75, None)


class TestDescriptor:
    @pytest.mark.parametrize("elements", [["Al"], ["Cu", "Zr"], ["Ni", "Al", "Cr"]])
    @pytest.mark.parametrize("pbc", [True, False], ids=["periodic", "free"])
    def test_as_brute_force(self, elements, pbc):
        system = scattered(elements, pbc=pbc)
        ours = generator(elements).frame_descriptors(system)
        np.testing.assert_allclose(ours, brute_force(system, elements), atol=1e-11)

    def test_invariances(self):
        system = scattered(["Cu", "Zr"], seed=3)
        g = generator(["Cu", "Zr"])
        x = g.frame_descriptors(system)
        order = np.random.default_rng(1).permutation(len(system))
        moved = system.gather(order).with_columns(
            positions=system["positions"][order] + [2.0, -3.0, 1.0]
        )
        np.testing.assert_allclose(g.frame_descriptors(moved), x[order], atol=1e-12)
        copies = g.frame_descriptors(replicate(system, (2, 1, 1)))
        np.testing.assert_allclose(copies, np.tile(x, (2, 1)), atol=1e-12)
        with mdapy.num_threads(1):
            assert np.array_equal(g.frame_descriptors(system), x)

    def test_one_element_has_no_block_of_elements(self):
        x = generator(["Al"]).frame_descriptors(scattered(["Al"]))
        assert x.shape[1] == 8 + 4 * 4
        y = generator(["Cu", "Zr"]).frame_descriptors(scattered(["Cu", "Zr"]))
        assert y.shape[1] == 8 + 4 * 4 + 2 * 2


class TestEntropy:
    def test_log_determinant(self):
        """Half the log determinant of the covariance of the points with a
        thousandth on its diagonal, raised by points that spread them."""
        rng = np.random.default_rng(0)
        z = rng.normal(size=(50, 3)) * [1.0, 2.0, 0.5]
        entropy = _Entropy(3)
        assert entropy.raised_by(z)
        entropy.add(z[:10])
        entropy.add(z[10:30])
        covariance = np.cov(z.T, bias=True) + 1e-3 * np.eye(3)
        expected = 0.5 * np.linalg.slogdet(covariance)[1]
        assert entropy.with_points(z[30:]) == pytest.approx(expected, rel=1e-12)
        assert entropy.raised_by(10 * z[30:])
        assert not entropy.raised_by(np.repeat(z[:30].mean(0, keepdims=True), 5, 0))

    def test_sizes_and_dense_cells(self):
        """As many random cells of each doubling of the number of atoms;
        cells too dense for atoms placed one by one take a jittered grid."""
        g = _Generator(("Al",), (2, 127), (10.0, 30.0), 0.75, np.random.default_rng(0))
        sizes = np.array([len(g.random().kinds) for _ in range(600)])
        counts = np.histogram(sizes, bins=[2, 4, 8, 16, 32, 64, 128])[0]
        assert counts.sum() == 600 and counts.min() > 70 and counts.max() < 130
        dense = _Generator(
            ("Al",), (64, 64), (8.3, 8.3), 0.75, np.random.default_rng(1)
        )
        for _ in range(3):
            cell = dense.random()
            assert len(cell.kinds) == 64 and dense.apart(cell)

    def test_slabs_and_clusters(self):
        """The fraction of random cells given vacuum, half slabs and half
        clusters; moves keep their atoms in the box of the atoms, the volume
        per atom of that box in range, a slab as thick as the reach, and the
        vacuum as wide."""
        g = _Generator(
            ("Cu", "Zr"), (2, 8), (12.0, 30.0), 0.75, np.random.default_rng(2), 0.4
        )
        cells = [g.random() for _ in range(500)]
        kinds = [tuple(c.vacuum) for c in cells]
        assert 60 < kinds.count((False, False, True)) < 140
        assert 60 < kinds.count((True, True, True)) < 140
        assert kinds.count((False, False, False)) > 240
        for cell in cells[:40]:
            for _ in range(60):
                trial = g.move(cell)
                if trial is not None and g.apart(trial):
                    cell = trial
            f = cell.positions @ np.linalg.inv(cell.core)
            assert ((f >= 0) & (f < 1))[:, cell.vacuum].all()
            volume = abs(np.linalg.det(cell.core)) / len(cell.kinds)
            assert 12.0 - 1e-9 <= volume <= 30.0 + 1e-9
            if cell.vacuum[2] and not cell.vacuum[0]:
                # a slab at least the reach of the descriptor thick
                assert mdapy.Box(cell.core).thickness[2] >= g.reach - 1e-9
            widened = mdapy.Box(cell.matrix).thickness - mdapy.Box(cell.core).thickness
            np.testing.assert_allclose(
                widened, np.where(cell.vacuum, 10.0, 0.0), atol=1e-9
            )


def pair_distances_ok(system, closest):
    radii = np.array([radius(e) for e in system["element"]])
    for i, j, d in images(system, 2 * radii.max() * closest):
        if np.linalg.norm(d) < closest * (radii[i] + radii[j]) - 1e-9:
            return False
    return True


class TestGenerate:
    def test_cells_keep_their_bounds(self):
        frames = generate_structures(
            ["Cu", "Zr"], 12, volume=(14.0, 28.0), atoms=(3, 10), vacuum=0.0,
            steps=5, seed=4,
        )  # fmt: skip
        assert len(frames) == 12
        for frame in frames:
            assert 3 <= len(frame) <= 10
            assert set(frame["element"]) <= {"Cu", "Zr"}
            assert frame.box.pbc.all()
            assert 14.0 - 1e-9 <= frame.box.volume / len(frame) <= 28.0 + 1e-9
            fractional = frame.box.to_fractional(frame["positions"])
            assert ((fractional >= -1e-12) & (fractional < 1 + 1e-12)).all()
            assert pair_distances_ok(frame, 0.75)
            assert frame.info == {"config_type": "bulk"}
        assert {len(set(frame["element"])) for frame in frames} == {1, 2}

    def test_slabs_and_clusters_in_vacuum(self):
        """With vacuum everywhere, periodic cells whose atoms leave at least
        10 Å free across the third axis, slabs, or across all three,
        clusters; the atoms apart and in the box."""
        frames = generate_structures(
            ["Cu", "Zr"], 16, volume=(14.0, 20.0), atoms=(3, 8), vacuum=1.0,
            steps=5, seed=6,
        )  # fmt: skip
        free = []
        for frame in frames:
            assert frame.box.pbc.all()
            fractional = frame.box.to_fractional(frame["positions"])
            assert ((fractional >= 0) & (fractional < 1)).all()
            spread = fractional.max(axis=0) - fractional.min(axis=0)
            free.append(tuple(frame.box.thickness * (1 - spread) >= 10.0 - 1e-9))
            assert pair_distances_ok(frame, 0.75)
            kind = "cluster" if free[-1][0] else "slab"
            assert frame.info == {"config_type": kind}
        assert set(free) == {(False, False, True), (True, True, True)}

    def test_same_seed_same_structures(self):
        first = generate_structures("Al", 5, volume=(10.0, 25.0), steps=3, seed=7)
        again = generate_structures("Al", 5, volume=(10.0, 25.0), steps=3, seed=7)
        other = generate_structures("Al", 5, volume=(10.0, 25.0), steps=3, seed=8)
        for a, b in zip(first, again, strict=True):
            assert np.array_equal(a["positions"], b["positions"])
            assert np.array_equal(a.box.matrix, b.box.matrix)
        assert not np.array_equal(first[1]["positions"], other[1]["positions"])

    @pytest.mark.parametrize("per", ["atom", "structure"])
    def test_moves_raise_the_entropy(self, per):
        """Structures moved to raise the entropy spread their descriptors
        more than the random cells they start from: a larger log
        determinant of their covariance, in units of the spread of those of
        the random cells."""
        volume = (8.0, 33.0)
        moved = generate_structures(
            "Al", 40, volume=volume, atoms=(2, 24), per=per, seed=1
        )
        random = generate_structures(
            "Al", 40, volume=volume, atoms=(2, 24), steps=0, seed=1
        )
        g = generator(["Al"])

        def points(frames):
            x = [g.frame_descriptors(f) for f in frames]
            if per == "structure":
                x = [v.mean(axis=0, keepdims=True) for v in x]
            return np.concatenate(x)

        scale = points(random).std(axis=0)

        def entropy(frames):
            z = points(frames) / scale
            covariance = np.cov(z.T, bias=True) + 1e-3 * np.eye(len(scale))
            return 0.5 * np.linalg.slogdet(covariance)[1]

        assert entropy(moved) > entropy(random) + 1.0

    def test_reference_is_complemented(self):
        """A reference of fcc crystals is part of the set: structures moved
        against it raise the entropy of both together."""
        fcc = build_crystal("fcc", 4.05, elements="Al", repeat=(2, 2, 2))
        reference = perturb(fcc, 20, displacement=0.05, seed=2)
        frames = generate_structures(
            "Al", 5, volume=(10.0, 25.0), steps=4, reference=reference, seed=3
        )
        assert len(frames) == 5
        with pytest.raises(InvalidParameter, match="not given"):
            generate_structures("Cu", 1, volume=(10.0, 25.0), reference=reference)
        with pytest.raises(MissingProperty, match="element"):
            generate_structures(
                "Al",
                1,
                volume=(10.0, 25.0),
                reference=[mdapy.System(fcc["positions"], fcc.box)],
            )
        with pytest.raises(InvalidParameter, match="sequence of System"):
            generate_structures(
                "Al", 1, volume=(10.0, 25.0), reference=[fcc["positions"]]
            )

    def test_energy_bound(self):
        eam = EAM.zhou(["Al"])
        frames = generate_structures(
            "Al",
            10,
            volume=(12.0, 25.0),
            atoms=(2, 16),
            steps=4,
            calculator=eam,
            max_energy=-2.8,
            seed=5,
        )
        energies = [eam.compute(f, ("energy",)).energy / len(f) for f in frames]
        assert len(frames) == 10 and max(energies) <= -2.8

    def test_stops_when_too_few_are_kept(self, monkeypatch):
        """With every other candidate kept, a tolerance above one half stops
        the generation once 50 candidates were tried; with none kept, it
        stops after 100 in a row."""
        state = {"calls": 0}

        def every_other(self, z):
            state["calls"] += 1
            return state["calls"] % 2 == 1

        monkeypatch.setattr(_Entropy, "raised_by", every_other)
        frames = generate_structures(
            "Al", 1000, volume=(10.0, 25.0), steps=0, tolerance=0.6, seed=0
        )
        assert len(frames) == 25 and state["calls"] == 50
        frames = generate_structures(
            "Al", 1000, volume=(10.0, 25.0), steps=0, tolerance=0.4, seed=0
        )
        assert len(frames) == 1000
        monkeypatch.setattr(_Entropy, "raised_by", lambda self, z: False)
        assert generate_structures("Al", 10, volume=(10.0, 25.0), steps=0) == []

    def test_volume_too_small(self):
        with pytest.raises(InvalidParameter, match="too small for closest"):
            generate_structures("Al", 1, volume=(1.0, 1.0), atoms=(8, 8))
        frames = generate_structures(
            "Al", 3, volume=(16.0, 16.0), atoms=(1, 1), vacuum=0.0, steps=5, seed=0
        )
        for frame in frames:
            assert frame.box.volume == pytest.approx(16.0, rel=1e-12)

    def test_progress_and_cancel(self):
        calls = []
        generate_structures(
            "Al", 3, volume=(10.0, 25.0), steps=1, seed=0,
            progress=lambda done, total: calls.append((done, total)),
        )  # fmt: skip
        assert calls == [(1, 3), (2, 3), (3, 3)]
        token = mdapy.CancelToken()
        token.cancel()
        with pytest.raises(Cancelled):
            generate_structures("Al", 3, volume=(10.0, 25.0), cancel=token)
        assert generate_structures("Al", 0, volume=(10.0, 25.0)) == []

    @pytest.mark.parametrize(
        "elements, kwargs, match",
        [
            ([], {}, "at least one element"),
            ("Xx", {}, "covalent radius"),
            (["Al", "Al"], {}, "twice"),
            ("Al", {"count": -1}, "count"),
            ("Al", {"count": True}, "count"),
            ("Al", {"volume": 10.0}, "pair"),
            ("Al", {"volume": (20.0, 10.0)}, "smallest <= largest"),
            ("Al", {"volume": (0.0, 10.0)}, "smallest <= largest"),
            ("Al", {"volume": ("a", 10.0)}, "two numbers"),
            ("Al", {"atoms": (2.5, 4)}, "two numbers"),
            ("Al", {"atoms": (0, 4)}, "smallest <= largest"),
            ("Al", {"vacuum": 1.5}, "vacuum"),
            ("Al", {"vacuum": -0.1}, "vacuum"),
            ("Al", {"vacuum": True}, "vacuum"),
            ("Al", {"closest": 0.0}, "closest"),
            ("Al", {"closest": True}, "closest"),
            ("Al", {"per": "frame"}, "per"),
            ("Al", {"steps": -1}, "steps"),
            ("Al", {"max_energy": -3.0}, "together"),
            ("Al", {"tolerance": 1.0}, "tolerance"),
            ("Al", {"tolerance": "0.1"}, "tolerance"),
            ("Al", {"cancel": object()}, "CancelToken"),
        ],
    )
    def test_invalid(self, elements, kwargs, match):
        arguments = {"count": 1, "volume": (10.0, 25.0)} | kwargs
        with pytest.raises(InvalidParameter, match=match):
            generate_structures(elements, **arguments)
