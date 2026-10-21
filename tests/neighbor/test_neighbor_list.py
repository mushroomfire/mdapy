# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

import itertools
import math

import numpy as np
import pytest

import mdapy
from mdapy import NeighborList
from mdapy.errors import Cancelled, InvalidParameter


def brute_force(system, cutoff):
    """Every (i, j, shift) closer than cutoff, by enumerating images in NumPy,
    with the arithmetic in the same order as the compiled search."""
    box = system.box
    h = box.matrix
    x = system.positions
    fractional = box.to_fractional(x) if len(x) else np.zeros((0, 3))
    spread = np.ptp(fractional, axis=0) if len(x) else np.zeros(3)
    reach = [
        math.ceil(cutoff / t + s) + 1 if p else 0
        for t, s, p in zip(box.thickness, spread, box.pbc)
    ]
    pairs = {}
    for shift in itertools.product(*(range(-r, r + 1) for r in reach)):
        r = x[None, :, :] - x[:, None, :]
        r = r + shift[0] * h[0]
        r = r + shift[1] * h[1]
        r = r + shift[2] * h[2]
        r2 = r[..., 0] * r[..., 0] + r[..., 1] * r[..., 1] + r[..., 2] * r[..., 2]
        for i, j in zip(*np.nonzero(r2 < cutoff * cutoff)):
            if i == j and shift == (0, 0, 0):
                continue
            pairs[(int(i), int(j), shift)] = math.sqrt(r2[i, j])
    return pairs


def as_dict(nl):
    return {
        (int(i), int(j), tuple(int(s) for s in shift)): d
        for i, j, shift, d in zip(nl.i, nl.j, nl.shifts, nl.distances)
    }


BOXES = {
    "orthogonal": [[5.0, 0, 0], [0, 6.0, 0], [0, 0, 7.0]],
    "triclinic": [[5.0, 0, 0], [2.0, 4.5, 0], [-1.5, 1.0, 6.0]],
    "rotated": [[3.0, 4.0, 0], [-1.0, 3.0, 2.0], [0.5, -1.0, 5.0]],
    "left-handed": [[5.0, 0, 0], [0, 0, 6.0], [0, 6.0, 0]],
    "flat": [[9.0, 0, 0], [0, 9.0, 0], [0, 0, 2.0]],
}


def random_system(box_name, pbc=True, n=40, seed=0, outside=False):
    rng = np.random.default_rng(seed)
    box = mdapy.Box(BOXES[box_name], origin=[1.0, -2.0, 0.5], pbc=pbc)
    fractional = rng.random((n, 3))
    if outside:  # atoms anywhere, not wrapped into the box
        fractional = fractional * 3 - 1
    return mdapy.System(box.to_cartesian(fractional), box)


@pytest.mark.parametrize("box_name", BOXES)
@pytest.mark.parametrize(
    "pbc", [True, (True, True, False), (False, True, False), False]
)
@pytest.mark.parametrize("cutoff", [1.5, 3.2, 8.0])
def test_matches_brute_force(box_name, pbc, cutoff):
    system = random_system(box_name, pbc)
    nl = NeighborList.build(system, cutoff)
    expected = brute_force(system, cutoff)
    found = as_dict(nl)
    assert found.keys() == expected.keys()
    assert all(found[k] == expected[k] for k in found)  # same arithmetic: same bits


@pytest.mark.parametrize("pbc", [True, (True, False, True)])
def test_atoms_outside_the_box(pbc):
    system = random_system("triclinic", pbc, outside=True, seed=3)
    assert (
        as_dict(NeighborList.build(system, 3.0)).keys()
        == brute_force(system, 3.0).keys()
    )


def test_tiny_box_sees_its_own_images():
    system = mdapy.System([[0.0, 0.0, 0.0]], mdapy.Box(2.0))
    nl = NeighborList.build(system, 4.5)
    expected = brute_force(system, 4.5)
    assert as_dict(nl).keys() == expected.keys()
    assert (0, 0, (1, 0, 0)) in expected and len(nl) == len(expected)


def test_pairs_are_symmetric_and_sorted():
    system = random_system("triclinic", seed=4)
    nl = NeighborList.build(system, 4.0)
    pairs = set(as_dict(nl))
    assert all((j, i, tuple(-s for s in shift)) in pairs for i, j, shift in pairs)
    for i in range(len(system)):
        block = slice(nl.offsets[i], nl.offsets[i + 1])
        keys = [(j, tuple(s)) for j, s in zip(nl.j[block], nl.shifts[block].tolist())]
        assert keys == sorted(keys)
    assert (nl.i == np.repeat(np.arange(len(system)), nl.counts)).all()


def test_vectors_and_distances():
    system = random_system("rotated", seed=5)
    nl = NeighborList.build(system, 4.0)
    x = system.positions
    # computed as the search computes the distances, in compiled code
    np.testing.assert_allclose(
        nl.vectors, x[nl.j] - x[nl.i] + nl.shifts @ system.box.matrix, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        np.linalg.norm(nl.vectors, axis=1), nl.distances, rtol=1e-14
    )
    assert nl.vectors is nl.vectors  # computed once
    assert not nl.distances.flags.writeable and not nl.vectors.flags.writeable


def test_same_result_for_any_thread_count():
    system = random_system("triclinic", n=3000, seed=6)
    results = []
    for threads in (1, 3, 8):
        with mdapy.num_threads(threads):
            results.append(NeighborList.build(system, 3.5))
    for other in results[1:]:
        for name in ("offsets", "j", "shifts", "distances"):
            np.testing.assert_array_equal(
                getattr(other, name), getattr(results[0], name)
            )


# -------------------------------------------------------------- invariance


def test_translation_and_rotation():
    system = random_system("triclinic", seed=7)
    nl = NeighborList.build(system, 3.5)
    moved = system.with_positions(system.positions + [10.3, -4.2, 7.7])
    assert as_dict(NeighborList.build(moved, 3.5)).keys() == as_dict(nl).keys()
    angle = 0.7
    rotation = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle), math.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    box = system.box
    rotated = mdapy.System(
        system.positions @ rotation,
        mdapy.Box(box.matrix @ rotation, origin=box.origin @ rotation),
    )
    other = NeighborList.build(rotated, 3.5)
    a, b = as_dict(nl), as_dict(other)
    near_cutoff = {k for k, d in a.items() if abs(d - 3.5) < 1e-12}
    assert a.keys() - near_cutoff == b.keys() - near_cutoff
    np.testing.assert_allclose(
        [b[k] for k in a if k in b], [a[k] for k in a if k in b], rtol=1e-13
    )


def test_permutation():
    system = random_system("orthogonal", seed=8)
    order = np.random.default_rng(1).permutation(len(system))
    permuted = system.select(order)
    a = as_dict(NeighborList.build(system, 3.0))
    b = {
        (int(order[i]), int(order[j]), shift): d
        for (i, j, shift), d in as_dict(NeighborList.build(permuted, 3.0)).items()
    }
    assert a == b


def lattice(kind, a, repeat=1):
    basis = {"fcc": [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
             "bcc": [[0, 0, 0], [0.5, 0.5, 0.5]]}[kind]  # fmt: skip
    cells = np.array(list(itertools.product(range(repeat), repeat=3)))
    fractional = (cells[:, None, :] + np.array(basis)[None]).reshape(-1, 3) / repeat
    box = mdapy.Box(a * repeat)
    return mdapy.System(box.to_cartesian(fractional), box)


@pytest.mark.parametrize(
    "kind, shells", [("fcc", [(0.8, 12), (1.1, 18)]), ("bcc", [(0.9, 8), (1.1, 14)])]
)
@pytest.mark.parametrize("repeat", [1, 3])
def test_coordination_of_perfect_lattices(kind, shells, repeat):
    system = lattice(kind, 3.6, repeat)
    for factor, count in shells:
        nl = NeighborList.build(system, factor * 3.6)
        assert (nl.counts == count).all()


def test_primitive_cell_and_supercell_agree():
    a = 3.6
    primitive = mdapy.System(
        [[0.0, 0, 0]],
        mdapy.Box([[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]]),
    )
    supercell = lattice("fcc", a, 2)
    for cutoff in (2.6, 3.7, 4.5, 6.0):
        counts = NeighborList.build(supercell, cutoff).counts
        assert (counts == NeighborList.build(primitive, cutoff).counts[0]).all()


# ---------------------------------------------------- derived lists


def test_within_and_sorted():
    system = random_system("triclinic", seed=9)
    nl = NeighborList.build(system, 5.0)
    small = nl.within(3.0)
    assert as_dict(small) == as_dict(NeighborList.build(system, 3.0))
    assert small.cutoff == 3.0
    ordered = nl.sorted()
    assert as_dict(ordered) == as_dict(nl)
    for i in range(len(system)):
        block = ordered.distances[ordered.offsets[i] : ordered.offsets[i + 1]]
        assert (np.diff(block) >= 0).all()
    for bad in (0.0, 5.5):
        with pytest.raises(InvalidParameter, match="at most 5.0"):
            nl.within(bad)


@pytest.mark.parametrize("pbc", [True, False])
def test_nearest_neighbors(pbc):
    system = random_system("triclinic", pbc, n=30, seed=10)
    k = 6
    nl = NeighborList.nearest(system, k)
    assert (nl.counts == k).all()
    reference = brute_force(system, 12.0)
    for i in range(len(system)):
        distances = sorted(d for (a, _, _), d in reference.items() if a == i)[:k]
        np.testing.assert_array_equal(nl.distances[i * k : (i + 1) * k], distances)
    assert nl.cutoff > nl.distances.max()


def test_nearest_errors_and_empty_systems():
    system = mdapy.System(
        np.zeros((3, 3)) + [[0, 0, 0], [1, 0, 0], [0, 1, 0]], mdapy.Box(5.0, pbc=False)
    )
    with pytest.raises(InvalidParameter, match="only 3 atoms"):
        NeighborList.nearest(system, 3)
    with pytest.raises(InvalidParameter, match="at least 1"):
        NeighborList.nearest(system, 0)
    empty = mdapy.System(np.zeros((0, 3)), mdapy.Box(5.0))
    assert len(NeighborList.nearest(empty, 4)) == 0
    nl = NeighborList.build(empty, 3.0)
    assert len(nl) == 0 and nl.offsets.tolist() == [0]


# ------------------------------------------------------------- control


@pytest.mark.parametrize("cutoff", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_cutoffs(cutoff):
    with pytest.raises(InvalidParameter, match="positive number"):
        NeighborList.build(random_system("orthogonal"), cutoff)


def test_progress_and_cancellation():
    system = random_system("orthogonal", n=20000, seed=11)
    reports = []
    NeighborList.build(
        system, 2.0, progress=lambda done, total: reports.append((done, total))
    )
    assert reports[-1] == (20000, 20000)
    token = mdapy.CancelToken()
    token.cancel()
    with pytest.raises(Cancelled):
        NeighborList.build(system, 2.0, cancel=token)


def test_repr():
    nl = NeighborList.build(lattice("fcc", 3.6), 3.0)
    assert repr(nl) == "NeighborList(atoms=4, pairs=48, cutoff=3.0)"
