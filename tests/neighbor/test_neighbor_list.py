# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

import itertools
import math

import numpy as np
import pytest

import mdapy
from mdapy import NeighborList
from mdapy.errors import Cancelled, InvalidParameter
from mdapy.neighbor._neighbor_list import _given_or_built

# ------------------------------------------------------------ references


def image_vectors(system, reach):
    """r[i, j] = x[j] - x[i] + shift . H for every image shift within reach
    (one number per direction), with the arithmetic in the same order as the
    compiled code: (shifts, n, n) squared lengths and the shifts."""
    h = system.box.matrix
    x = system.positions
    shifts = list(itertools.product(*(range(-r, r + 1) for r in reach)))
    r2 = np.empty((len(shifts), len(x), len(x)))
    for a, shift in enumerate(shifts):
        r = x[None, :, :] - x[:, None, :]
        r = r + shift[0] * h[0]
        r = r + shift[1] * h[1]
        r = r + shift[2] * h[2]
        r2[a] = r[..., 0] * r[..., 0] + r[..., 1] * r[..., 1] + r[..., 2] * r[..., 2]
    return np.array(shifts, dtype=np.int64), r2


def reach_for(system, radius):
    """Images to enumerate so that every pair within radius is among them."""
    box = system.box
    x = system.positions
    fractional = box.to_fractional(x) if len(x) else np.zeros((0, 3))
    spread = np.ptp(fractional, axis=0) if len(x) else np.zeros(3)
    return [
        math.ceil(radius / t + s) + 1 if p else 0
        for t, s, p in zip(box.thickness, spread, box.pbc)
    ]


def brute_force(system, cutoff):
    """Every (i, j, shift) closer than cutoff, with its distance."""
    shifts, r2 = image_vectors(system, reach_for(system, cutoff))
    pairs = {}
    for a, i, j in zip(*np.nonzero(r2 < cutoff * cutoff)):
        shift = tuple(int(s) for s in shifts[a])
        if i == j and shift == (0, 0, 0):
            continue
        pairs[(int(i), int(j), shift)] = math.sqrt(r2[a, i, j])
    return pairs


def brute_nearest(system, k):
    """The k nearest (j, shift, distance) of each atom, ordered by distance,
    then j, then shift."""
    n = len(system)
    radius = 1.0
    while True:
        shifts, r2 = image_vectors(system, reach_for(system, radius))
        self_pair = (shifts == 0).all(axis=1)[:, None, None] & np.eye(n, dtype=bool)
        r2 = np.where(self_pair, np.inf, r2)
        farthest = math.sqrt(
            np.sort(r2.transpose(1, 0, 2).reshape(n, -1), axis=1)[:, k - 1].max()
        )
        if farthest <= radius:
            break
        radius = 2 * farthest
    result = []
    for i in range(n):
        a, j = np.meshgrid(np.arange(len(shifts)), np.arange(n), indexing="ij")
        d2 = r2[:, i, :].ravel()
        a, j = a.ravel(), j.ravel()
        s = shifts[a]
        order = np.lexsort((s[:, 2], s[:, 1], s[:, 0], j, d2))[:k]
        result.append(
            [(int(j[o]), tuple(int(v) for v in s[o]), math.sqrt(d2[o])) for o in order]
        )
    return result


def as_dict(nl):
    return {
        (int(i), int(j), tuple(int(s) for s in shift)): d
        for i, j, shift, d in zip(nl.i, nl.j, nl.shifts, nl.distances)
    }


def as_rows(nl):
    return [
        [
            (int(nl.j[p]), tuple(int(s) for s in nl.shifts[p]), float(nl.distances[p]))
            for p in range(nl.offsets[i], nl.offsets[i + 1])
        ]
        for i in range(len(nl.counts))
    ]


BOXES = {
    "orthogonal": [[5.0, 0, 0], [0, 6.0, 0], [0, 0, 7.0]],
    "triclinic": [[5.0, 0, 0], [2.0, 4.5, 0], [-1.5, 1.0, 6.0]],
    "rotated": [[3.0, 4.0, 0], [-1.0, 3.0, 2.0], [0.5, -1.0, 5.0]],
    "left-handed": [[5.0, 0, 0], [0, 0, 6.0], [0, 6.0, 0]],
    "flat": [[9.0, 0, 0], [0, 9.0, 0], [0, 0, 2.0]],
}
PBC = [True, (True, True, False), (False, True, False), False]


def random_system(box_name, pbc=True, n=40, seed=0, outside=False):
    rng = np.random.default_rng(seed)
    box = mdapy.Box(BOXES[box_name], origin=[1.0, -2.0, 0.5], pbc=pbc)
    fractional = rng.random((n, 3))
    if outside:  # atoms anywhere, not wrapped into the box
        fractional = fractional * 3 - 1
    return mdapy.System(box.to_cartesian(fractional), box)


def lattice(kind, a, repeat=1):
    basis = {"fcc": [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
             "bcc": [[0, 0, 0], [0.5, 0.5, 0.5]]}[kind]  # fmt: skip
    cells = np.array(list(itertools.product(range(repeat), repeat=3)))
    fractional = (cells[:, None, :] + np.array(basis)[None]).reshape(-1, 3) / repeat
    box = mdapy.Box(a * repeat)
    return mdapy.System(box.to_cartesian(fractional), box)


def check_arrays(nl, n):
    assert nl.offsets.dtype == np.int64 and nl.offsets.shape == (n + 1,)
    assert nl.offsets[0] == 0 and (np.diff(nl.offsets) >= 0).all()
    assert nl.j.dtype == mdapy.index_dtype and nl.i.dtype == mdapy.index_dtype
    assert nl.shifts.dtype == np.int8 and nl.shifts.shape == (len(nl), 3)
    assert nl.distances.dtype == np.float64
    for array in (nl.offsets, nl.j, nl.shifts, nl.distances, nl.i):
        assert not array.flags.writeable


# ------------------------------------------------------------ within a cutoff


@pytest.mark.parametrize("box_name", BOXES)
@pytest.mark.parametrize("pbc", PBC)
@pytest.mark.parametrize("cutoff", [1.5, 3.2, 8.0])
def test_matches_brute_force(box_name, pbc, cutoff):
    system = random_system(box_name, pbc)
    nl = NeighborList.build(system, cutoff)
    check_arrays(nl, len(system))
    expected = brute_force(system, cutoff)
    found = as_dict(nl)
    assert len(found) == len(nl)  # no pair twice
    assert found.keys() == expected.keys()
    assert all(found[k] == expected[k] for k in found)  # same arithmetic: same bits
    assert nl.num_neighbors is None and nl.cutoff == cutoff


@pytest.mark.parametrize("box_name", ["triclinic", "flat"])
def test_cutoffs_at_pair_distances(box_name):
    # a pair exactly at the cutoff is out, and in just above it
    system = random_system(box_name, n=20, seed=2)
    distances = np.unique(NeighborList.build(system, 6.0).distances)
    for d in distances[:: len(distances) // 12]:
        for cutoff in (d, np.nextafter(d, np.inf), np.nextafter(d, 0.0)):
            assert as_dict(NeighborList.build(system, cutoff)) == brute_force(
                system, cutoff
            )


@pytest.mark.parametrize("pbc", [True, (True, False, True)])
def test_atoms_outside_the_box(pbc):
    system = random_system("triclinic", pbc, outside=True, seed=3)
    assert as_dict(NeighborList.build(system, 3.0)) == brute_force(system, 3.0)


def test_tiny_box_sees_its_own_images():
    system = mdapy.System([[0.0, 0.0, 0.0]], mdapy.Box(2.0))
    nl = NeighborList.build(system, 4.5)
    expected = brute_force(system, 4.5)
    assert as_dict(nl) == expected
    assert (0, 0, (1, 0, 0)) in expected and (0, 0, (2, 1, 0)) in expected


def test_pairs_are_symmetric_and_grouped():
    system = random_system("triclinic", seed=4)
    nl = NeighborList.build(system, 4.0)
    pairs = as_dict(nl)
    for (i, j, shift), d in pairs.items():
        assert pairs[(j, i, tuple(-s for s in shift))] == d
    assert (nl.i == np.repeat(np.arange(len(system)), nl.counts)).all()


def test_vectors_and_distances():
    system = random_system("rotated", seed=5)
    nl = NeighborList.build(system, 4.0)
    x = system.positions
    # computed as the search computes the distances, in compiled code
    np.testing.assert_allclose(
        nl.vectors,
        x[nl.j] - x[nl.i] + np.einsum("pj,jk->pk", nl.shifts, system.box.matrix),
        rtol=0,
        atol=1e-12,
    )
    assert (np.sqrt((nl.vectors**2).sum(axis=1)) == nl.distances).all()
    assert nl.vectors is nl.vectors  # computed once
    assert not nl.vectors.flags.writeable


def test_shifts_up_to_127_boxes():
    def pair(x):
        box = mdapy.Box([1.0, 10.0, 10.0], pbc=(True, False, False))
        return mdapy.System([[0.0, 5.0, 5.0], [x, 5.0, 5.0]], box)

    nl = NeighborList.build(pair(127.3), 0.45)
    assert as_dict(nl) == brute_force(pair(127.3), 0.45)
    assert sorted(map(tuple, nl.shifts.tolist())) == [(-127, 0, 0), (127, 0, 0)]
    for kwargs in ({"cutoff": 0.45}, {"num_neighbors": 1}):
        with pytest.raises(InvalidParameter, match="int8 range"):
            NeighborList.build(pair(128.3), **kwargs)
    # wrapped into the box, the same atoms are neighbors without large shifts
    far = pair(128.3)
    positions, _ = far.box.wrap(far.positions)
    assert len(NeighborList.build(far.with_positions(positions), 0.45)) == 2


# ------------------------------------------------------------ nearest neighbors


@pytest.mark.parametrize("box_name", BOXES)
@pytest.mark.parametrize("pbc", PBC)
@pytest.mark.parametrize("n, k", [(40, 1), (40, 6), (12, 11), (3, 13)])
def test_nearest_matches_brute_force(box_name, pbc, n, k):
    system = random_system(box_name, pbc, n=n, seed=n + k)
    if not any(np.atleast_1d(pbc)) and n - 1 < k:
        return  # tested in test_nearest_errors
    nl = NeighborList.build(system, num_neighbors=k)
    check_arrays(nl, n)
    assert nl.num_neighbors == k and (nl.counts == k).all()
    assert as_rows(nl) == brute_nearest(system, k)  # order, images and bits
    assert nl.cutoff > nl.distances.max()
    assert nl.cutoff == np.nextafter(nl.distances.max(), np.inf)


@pytest.mark.parametrize("k", [1, 8, 12, 13, 19])
def test_nearest_breaks_ties_by_atom_then_image(k):
    # an fcc crystal: 12 neighbors at the same distance, 6 next
    system = lattice("fcc", 3.6, 2)
    nl = NeighborList.build(system, num_neighbors=k)
    assert as_rows(nl) == brute_nearest(system, k)
    first = min(k, 12)
    np.testing.assert_allclose(
        nl.distances.reshape(-1, k)[:, :first], 3.6 / math.sqrt(2)
    )


def test_nearest_in_tiny_boxes_sees_images():
    a = 3.6
    primitive = mdapy.System(
        [[0.0, 0, 0]],
        mdapy.Box([[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]]),
    )
    nl = NeighborList.build(primitive, num_neighbors=18)
    assert as_rows(nl) == brute_nearest(primitive, 18)
    np.testing.assert_allclose(nl.distances[:12], a / math.sqrt(2))
    np.testing.assert_allclose(nl.distances[12:], a)
    # a box much thinner than the distances along one direction only
    thin = mdapy.System([[0.0, 0, 0], [0.3, 0.2, 0.1]], mdapy.Box([0.5, 7.0, 9.0]))
    assert as_rows(NeighborList.build(thin, num_neighbors=40)) == brute_nearest(
        thin, 40
    )


@pytest.mark.parametrize("shape", ["cluster", "slab", "mixed"])
@pytest.mark.parametrize("pbc", [True, False])
def test_nearest_in_nonuniform_systems(shape, pbc):
    rng = np.random.default_rng(12)
    box = mdapy.Box(40.0, pbc=pbc)
    if shape == "cluster":  # a ball of atoms in vacuum
        x = rng.normal(20.0, 2.0, (150, 3))
    elif shape == "slab":  # dense layer, vacuum above and below
        x = rng.random((150, 3)) * [40.0, 40.0, 3.0] + [0, 0, 18.0]
    else:  # dense half, a few atoms in the other half
        x = np.concatenate(
            [
                rng.random((140, 3)) * [20.0, 40.0, 40.0],
                rng.random((10, 3)) * [20.0, 40.0, 40.0] + [20.0, 0, 0],
            ]
        )
    system = mdapy.System(x, box)
    for k in (1, 12):
        assert as_rows(NeighborList.build(system, num_neighbors=k)) == brute_nearest(
            system, k
        )


def test_nearest_agrees_with_a_sorted_cutoff_list():
    system = random_system("triclinic", n=300, seed=13)
    k = 10
    near = NeighborList.build(system, num_neighbors=k)
    wide = NeighborList.build(system, near.cutoff).sorted()
    for i in range(len(system)):
        first = slice(wide.offsets[i], wide.offsets[i] + k)
        np.testing.assert_array_equal(
            near.distances[i * k : (i + 1) * k], wide.distances[first]
        )
    # the farthest neighbor of some atom is just inside the list's cutoff
    assert wide.distances.max() == near.distances.max()
    assert as_rows(NeighborList.build(system, near.cutoff)._nearest(k)) == as_rows(near)


@pytest.mark.parametrize("k", [12, 14, 18])
def test_ties_of_a_cutoff_list_as_in_nearest(k):
    # 12 neighbors at the same distance and 6 next: the k nearest of a cutoff
    # list are those of a list of nearest neighbors, ties by atom and image
    system = lattice("fcc", 3.6, 2)
    near = NeighborList.build(system, num_neighbors=k)
    wide = NeighborList.build(system, 4.0)
    assert as_rows(wide._nearest(k)) == as_rows(near)
    assert as_rows(wide.sorted()._nearest(k)) == as_rows(near)


def test_nearest_of_atoms_outside_the_box():
    system = random_system("rotated", (True, False, True), n=30, seed=14, outside=True)
    assert as_rows(NeighborList.build(system, num_neighbors=7)) == brute_nearest(
        system, 7
    )


def test_nearest_errors_and_empty_systems():
    system = mdapy.System(
        [[0.0, 0, 0], [1, 0, 0], [0, 1, 0]], mdapy.Box(5.0, pbc=False)
    )
    with pytest.raises(InvalidParameter, match="only 3 atoms"):
        NeighborList.build(system, num_neighbors=3)
    assert len(NeighborList.build(system, num_neighbors=2)) == 6
    for bad in (0, -1, 2.5, True, "3"):
        with pytest.raises(InvalidParameter, match="positive integer"):
            NeighborList.build(system, num_neighbors=bad)
    assert len(NeighborList.build(system, num_neighbors=np.int64(1))) == 3
    for kwargs in ({}, {"cutoff": 2.0, "num_neighbors": 2}):
        with pytest.raises(
            InvalidParameter, match="one of cutoff, num_neighbors and voronoi"
        ):
            NeighborList.build(system, **kwargs)
    empty = mdapy.System(np.zeros((0, 3)), mdapy.Box(5.0))
    for nl in (
        NeighborList.build(empty, num_neighbors=4),
        NeighborList.build(empty, 3.0),
    ):
        assert len(nl) == 0 and nl.offsets.tolist() == [0] and len(nl.vectors) == 0
        check_arrays(nl, 0)


# ------------------------------------------------------------ invariances


@pytest.mark.parametrize(
    "mode", [{"cutoff": 3.5}, {"num_neighbors": 12}, {"voronoi": True}]
)
def test_same_result_for_any_thread_count(mode):
    system = random_system("triclinic", n=3000, seed=6)
    results = []
    for threads in (1, 3, 8):
        with mdapy.num_threads(threads):
            results.append(NeighborList.build(system, **mode))
    for other in results[1:]:
        for name in ("offsets", "j", "shifts", "distances"):
            np.testing.assert_array_equal(
                getattr(other, name), getattr(results[0], name)
            )


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
    a, b = as_dict(nl), as_dict(NeighborList.build(rotated, 3.5))
    near_cutoff = {k for k, d in a.items() if abs(d - 3.5) < 1e-12}
    assert a.keys() - near_cutoff == b.keys() - near_cutoff
    np.testing.assert_allclose(
        [b[k] for k in a if k in b], [a[k] for k in a if k in b], rtol=1e-13
    )
    near, near_rotated = (
        NeighborList.build(s, num_neighbors=8) for s in (system, rotated)
    )
    np.testing.assert_allclose(near_rotated.distances, near.distances, rtol=1e-13)


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
    near, near_permuted = (
        NeighborList.build(s, num_neighbors=5) for s in (system, permuted)
    )
    np.testing.assert_array_equal(
        near_permuted.distances.reshape(-1, 5), near.distances.reshape(-1, 5)[order]
    )


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


# ------------------------------------------------------------ derived lists


@pytest.mark.parametrize("box_name", ["triclinic", "flat"])
def test_follow_moved_atoms_without_a_search(box_name):
    system = random_system(box_name, pbc=(True, True, False), seed=4)
    wide = NeighborList.build(system, 3.0)
    rng = np.random.default_rng(5)
    # steps shorter than half the difference of the cutoffs, 0.14 sqrt(3) < 0.25
    for step in (0.0, 0.05, 0.14):
        moved = system.with_columns(
            positions=system.positions + rng.uniform(-step, step, (len(system), 3))
        )
        followed = wide._follow(moved, 2.5)
        fresh = NeighborList.build(moved, 2.5)
        np.testing.assert_array_equal(followed.offsets, fresh.offsets)
        assert as_dict(followed) == as_dict(fresh)
        vectors = dict(zip(as_dict(fresh), map(tuple, fresh.vectors.tolist())))
        assert (
            dict(zip(as_dict(followed), map(tuple, followed.vectors.tolist())))
            == vectors
        )
        assert followed.system is moved and followed.cutoff == 2.5
    # too far, another box, or a list of nearest neighbors
    far = system.with_columns(positions=system.positions + [0.26, 0.0, 0.0])
    assert wide._follow(far, 2.5) is None
    assert wide._follow(system.with_box(system.box.matrix * 1.01), 2.5) is None
    assert NeighborList.build(system, num_neighbors=4)._follow(system, 0.1) is None


def test_within_keeps_the_order_of_the_list():
    system = random_system("triclinic", seed=10)
    nl = NeighborList.build(system, 5.0).sorted()  # an order unlike the search
    small = nl.within(3.5)
    keep = nl.distances < 3.5
    np.testing.assert_array_equal(small.j, nl.j[keep])
    np.testing.assert_array_equal(small.shifts, nl.shifts[keep])
    np.testing.assert_array_equal(small.distances, nl.distances[keep])
    np.testing.assert_array_equal(
        small.counts, np.bincount(nl.i[keep], minlength=len(system))
    )
    # a limit for each pair of types
    types = np.arange(len(system)) % 2
    limits = np.array([[3.0, 4.0], [4.0, 5.0]])
    cut = nl._cut(5.0, limits, types)
    keep = nl.distances < limits[types[nl.i], types[nl.j]]
    np.testing.assert_array_equal(cut.j, nl.j[keep])
    np.testing.assert_array_equal(cut.distances, nl.distances[keep])
    native = mdapy._core.neighbor
    arrays = (nl.offsets, nl.j, np.ascontiguousarray(nl.shifts), nl.distances)
    for kind, wrong, message in [
        (None, np.ones((1, 2)), "square"),
        (np.full(len(system), 2, np.int32), limits, "from 0"),
        (np.zeros(3, np.int32), limits, "each atom"),
    ]:
        with pytest.raises(ValueError, match=message):
            native.within(*arrays, kind, wrong)
    with pytest.raises(ValueError, match="do not fit"):
        native.within(
            nl.offsets,
            nl.j,
            np.ascontiguousarray(nl.shifts),
            nl.distances[1:],
            None,
            limits,
        )


def test_within_and_sorted():
    system = random_system("triclinic", seed=9)
    nl = NeighborList.build(system, 5.0)
    small = nl.within(3.0)
    assert as_dict(small) == as_dict(NeighborList.build(system, 3.0))
    assert small.cutoff == 3.0
    assert nl.within(5.0) is nl
    ordered = nl.sorted()
    assert as_dict(ordered) == as_dict(nl)
    for i in range(len(system)):
        block = ordered.distances[ordered.offsets[i] : ordered.offsets[i + 1]]
        assert (np.diff(block) >= 0).all()
    for bad in (0.0, 5.5):
        with pytest.raises(InvalidParameter, match="at most 5.0"):
            nl.within(bad)
    near = NeighborList.build(system, num_neighbors=6)
    np.testing.assert_array_equal(near.sorted().j, near.j)  # nearest first already
    assert near.sorted().num_neighbors == 6
    inner = near.within(2.0)
    assert inner.num_neighbors is None and (inner.distances < 2.0).all()
    assert len(inner) == (near.distances < 2.0).sum()
    # atoms without pairs
    lone = mdapy.System(
        [[0.0, 0, 0], [1.0, 0, 0], [9.0, 9, 9]], mdapy.Box(20.0, pbc=False)
    )
    ordered = NeighborList.build(lone, 2.0).sorted()
    np.testing.assert_array_equal(ordered.counts, [1, 1, 0])


# ------------------------------------------------------------ Voronoi neighbors


@pytest.mark.parametrize("pbc", [True, (True, False, True)])
def test_voronoi_neighbors_are_the_faces_of_the_cells(pbc):
    from mdapy.analysis import VoronoiAnalysis

    system = random_system("triclinic", pbc=pbc, n=200, seed=16)
    nl = NeighborList.build(system, voronoi=True)
    check_arrays(nl, len(system))
    cells = VoronoiAnalysis().compute(system)
    atoms = cells.j >= 0  # the walls are left out
    np.testing.assert_array_equal(nl.counts, cells.coordination)
    np.testing.assert_array_equal(nl.j, cells.j[atoms])
    np.testing.assert_array_equal(nl.shifts, cells.shifts[atoms])
    np.testing.assert_array_equal(nl.areas, cells.areas[atoms])
    np.testing.assert_allclose(
        nl.distances, np.linalg.norm(nl.vectors, axis=1), rtol=1e-14
    )
    assert nl.voronoi and nl.num_neighbors is None
    assert nl.cutoff > nl.distances.max()
    # every face is a pair of both of its atoms, with the same area
    areas = {
        (int(i), int(j), tuple(int(t) for t in shift)): a
        for i, j, shift, a in zip(nl.i, nl.j, nl.shifts, nl.areas)
    }
    for (i, j, shift), area in areas.items():
        assert areas[j, i, tuple(-t for t in shift)] == pytest.approx(area, rel=1e-9)


@pytest.mark.parametrize("kind, a, shells", [("fcc", 3.6, [12]), ("bcc", 2.87, [8, 6])])
def test_voronoi_neighbors_of_perfect_lattices(kind, a, shells):
    nl = NeighborList.build(lattice(kind, a, repeat=3), voronoi=True)
    assert (nl.counts == sum(shells)).all()
    first = {"fcc": a / math.sqrt(2), "bcc": a * math.sqrt(3) / 2}[kind]
    assert (
        np.isclose(nl.distances, first, rtol=1e-12) | np.isclose(nl.distances, a)
    ).all()
    assert (
        repr(nl)
        == f"NeighborList(atoms={len(nl.counts)}, pairs={len(nl)}, voronoi=True)"
    )


def test_voronoi_neighbors_with_thresholds_and_errors():
    from mdapy.analysis import VoronoiAnalysis

    system = random_system("triclinic", n=200, seed=17)
    thresholds = {
        "face_threshold": 0.5,
        "relative_face_threshold": 0.02,
        "edge_threshold": 0.3,
    }
    nl = NeighborList.build(system, voronoi=True, **thresholds)
    cells = VoronoiAnalysis(**thresholds).compute(system)
    np.testing.assert_array_equal(nl.areas, cells.areas[cells.j >= 0])
    assert len(nl) < len(NeighborList.build(system, voronoi=True))
    ordered = nl.sorted()
    assert ordered.voronoi and as_dict(ordered) == as_dict(nl)
    by_pair = dict(
        zip(zip(nl.i.tolist(), nl.j.tolist(), map(tuple, nl.shifts.tolist())), nl.areas)
    )
    for i, j, shift, area in zip(ordered.i, ordered.j, ordered.shifts, ordered.areas):
        assert by_pair[int(i), int(j), tuple(shift.tolist())] == area
    assert nl._follow(system, 1.0) is None
    with pytest.raises(InvalidParameter, match="no pairs to cut"):
        nl.within(1.0)
    with pytest.raises(InvalidParameter, match="Voronoi neighbors does not hold every"):
        _given_or_built(system, 1.0, nl)
    with pytest.raises(
        InvalidParameter, match="one of cutoff, num_neighbors and voronoi"
    ):
        NeighborList.build(system, 3.0, voronoi=True)
    with pytest.raises(
        InvalidParameter, match="one of cutoff, num_neighbors and voronoi"
    ):
        NeighborList.build(system)
    with pytest.raises(InvalidParameter, match="need voronoi"):
        NeighborList.build(system, 3.0, face_threshold=0.1)
    with pytest.raises(InvalidParameter, match="relative_face_threshold"):
        NeighborList.build(system, voronoi=True, relative_face_threshold=1.0)
    twins = mdapy.System([[1.0, 1, 1], [1.0, 1, 1]], mdapy.Box(5.0))
    with pytest.raises(InvalidParameter, match="same position"):
        NeighborList.build(twins, voronoi=True)
    empty = NeighborList.build(
        mdapy.System(np.zeros((0, 3)), mdapy.Box(5.0)), voronoi=True
    )
    assert len(empty) == 0 and empty.cutoff == 0.0


# ------------------------------------------------------------ control


@pytest.mark.parametrize("cutoff", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_cutoffs(cutoff):
    with pytest.raises(InvalidParameter, match="positive number"):
        NeighborList.build(random_system("orthogonal"), cutoff)


@pytest.mark.parametrize(
    "mode", [{"cutoff": 2.0}, {"num_neighbors": 6}, {"voronoi": True}]
)
def test_progress_and_cancellation(mode):
    system = random_system("orthogonal", n=20000, seed=11)
    reports = []
    NeighborList.build(
        system, **mode, progress=lambda done, total: reports.append((done, total))
    )
    done = [d for d, _ in reports]
    assert done == sorted(done) and reports[-1][0] == reports[-1][1] > 0
    token = mdapy.CancelToken()
    token.cancel()
    with pytest.raises(Cancelled):
        NeighborList.build(system, **mode, cancel=token)


def test_repr():
    system = lattice("fcc", 3.6)
    assert (
        repr(NeighborList.build(system, 3.0))
        == "NeighborList(atoms=4, pairs=48, cutoff=3.0)"
    )
    assert (
        repr(NeighborList.build(system, num_neighbors=12))
        == "NeighborList(atoms=4, pairs=48, num_neighbors=12)"
    )


def test_given_or_built():
    system = random_system("triclinic", seed=15)
    built = _given_or_built(system, 3.0, None)
    assert built.cutoff == 3.0 and as_dict(built) == as_dict(
        NeighborList.build(system, 3.0)
    )
    wide = NeighborList.build(system, 4.0)
    assert _given_or_built(system, 3.0, wide) is wide
    with pytest.raises(InvalidParameter, match="below the required 4.5"):
        _given_or_built(system, 4.5, wide)
    with pytest.raises(InvalidParameter, match="another system"):
        _given_or_built(system.with_positions(system.positions), 3.0, wide)
    with pytest.raises(InvalidParameter, match="nearest neighbors"):
        _given_or_built(system, 1.0, NeighborList.build(system, num_neighbors=30))
