# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

import numpy as np
import pytest
from lattices import crystal

import mdapy
from mdapy.analysis import BondAngleDistribution, ClusterAnalysis, WarrenCowley
from mdapy.errors import InvalidParameter, MissingProperty


def b2(a=2.95, repeat=(3, 3, 3)):
    """CsCl order on a bcc lattice: Cu at the corners, Zn at the centers."""
    system = crystal("bcc", a, repeat)
    return system.with_columns(element=["Cu", "Zn"] * (len(system) // 2))


def components(system, cutoff):
    """Clusters by a breadth-first search over the bonds, by first atom."""
    nl = mdapy.NeighborList.build(system, cutoff).within(cutoff)
    label = np.full(len(system), -1)
    groups = []
    for start in range(len(system)):
        if label[start] >= 0:
            continue
        label[start] = len(groups)
        group, todo = [start], [start]
        while todo:
            i = todo.pop()
            for j in nl.j[nl.offsets[i] : nl.offsets[i + 1]]:
                if label[j] < 0:
                    label[j] = label[start]
                    group.append(j)
                    todo.append(j)
        groups.append(sorted(group))
    return groups


def test_clusters_of_blobs():
    rng = np.random.default_rng(0)
    centers = [[5.0, 5, 5], [20.0, 20, 20], [5.0, 20, 5]]
    sizes = [30, 50, 30]
    positions = np.vstack(
        [c + rng.normal(scale=0.8, size=(s, 3)) for c, s in zip(centers, sizes)]
        + [[[35.0, 35, 35]]]
    )
    system = mdapy.System(positions, mdapy.Box(40.0, pbc=False), elements="Ar")
    result = ClusterAnalysis(cutoff=2.0).compute(system)
    groups = sorted(components(system, 2.0), key=lambda g: (-len(g), g[0]))
    np.testing.assert_array_equal(result.sizes, [len(g) for g in groups])
    for label, group in enumerate(groups):
        np.testing.assert_array_equal(np.flatnonzero(result.cluster == label), group)
    assert result.sizes[-1] == 1 and result.cluster[-1] == len(groups) - 1
    assert system.with_columns(result).table["cluster"].to_list() == list(
        result.cluster
    )


def test_clusters_across_the_boundary_and_per_pair():
    # two halves of a chain joined only through the periodic boundary
    chain = mdapy.System(
        [[0.5, 5, 5], [9.5, 5, 5], [3.0, 5, 5]], mdapy.Box(10.0), elements="Ar"
    )
    np.testing.assert_array_equal(
        ClusterAnalysis(cutoff=1.5).compute(chain).cluster, [0, 0, 1]
    )
    system = b2()
    like = ClusterAnalysis(cutoffs={("Cu", "Cu"): 3.0}).compute(system)
    cu = system.table["element"].to_numpy() == "Cu"
    assert like.sizes[0] == cu.sum() and (like.sizes[1:] == 1).all()
    assert (like.cluster[cu] == 0).all()
    assert ClusterAnalysis(cutoff=2.6).compute(system).sizes.tolist() == [len(system)]
    assert "cutoffs" in repr(ClusterAnalysis(cutoffs={("Cu", "Cu"): 3.0}))
    with pytest.raises(MissingProperty, match="element column"):
        ClusterAnalysis(cutoffs={("Cu", "Cu"): 3.0}).compute(
            mdapy.System(system.positions, system.box)
        )


@pytest.mark.parametrize(
    "kind, a, cutoff, expected",
    [
        ("fcc", 3.6, 3.0, {60: 24, 90: 12, 120: 24, 180: 6}),
        ("bcc", 2.87, 2.6, {70.5: 12, 109.5: 12, 180: 4}),
    ],
)
def test_bond_angles_of_crystals(kind, a, cutoff, expected):
    system = crystal(kind, a)
    result = BondAngleDistribution(cutoff, bins=360).compute(system)
    for angle, per_atom in expected.items():
        # exact angles such as 60 degrees lie on edges of the bins
        near = np.abs(result.angle - angle) < 1.0
        assert result.counts[near].sum() == per_atom * len(system)
    assert result.counts.sum() == sum(expected.values()) * len(system)
    np.testing.assert_allclose(result.density.sum() * 0.5, 1.0)


def test_bond_angles_match_numpy():
    system = crystal("fcc", 3.6, jitter=0.15, seed=1)
    result = BondAngleDistribution(3.2, bins=90).compute(system)
    nl = mdapy.NeighborList.build(system, 3.2).within(3.2)
    angles = []
    for i in range(len(system)):
        v = nl.vectors[nl.offsets[i] : nl.offsets[i + 1]]
        u = v / np.linalg.norm(v, axis=1, keepdims=True)
        a, b = np.triu_indices(len(u), 1)
        angles.append(np.degrees(np.arccos(np.clip((u[a] * u[b]).sum(1), -1, 1))))
    counts, _ = np.histogram(np.concatenate(angles), bins=90, range=(0, 180))
    np.testing.assert_array_equal(result.counts, counts)
    with mdapy.num_threads(3):
        np.testing.assert_array_equal(
            BondAngleDistribution(3.2, bins=90).compute(system).counts, counts
        )
    assert result.columns() == {}
    empty = mdapy.System(np.zeros((0, 3)), mdapy.Box(5.0), elements=[])
    assert not BondAngleDistribution(3.0).compute(empty).density.any()
    assert (
        repr(BondAngleDistribution(3.0))
        == "BondAngleDistribution(cutoff=3.0, bins=180)"
    )


def test_warren_cowley_of_ordered_and_random_alloys():
    ordered = WarrenCowley(num_neighbors=8).compute(b2())
    assert ordered.elements == ("Cu", "Zn")
    np.testing.assert_allclose(ordered.alpha, [[1.0, -1.0], [-1.0, 1.0]])
    assert ordered["Cu", "Zn"] == -1.0
    with pytest.raises(InvalidParameter, match="no parameter"):
        ordered["Cu", "Al"]
    by_cutoff = WarrenCowley(cutoff=2.6).compute(b2())
    np.testing.assert_allclose(by_cutoff.alpha, ordered.alpha)
    system = crystal("fcc", 3.6, (10, 10, 10))
    rng = np.random.default_rng(2)
    system = system.with_columns(element=rng.choice(["Cu", "Ni", "Al"], len(system)))
    random = WarrenCowley(cutoff=3.0).compute(system)
    np.testing.assert_allclose(random.alpha, 0.0, atol=0.06)  # 4000 atoms
    nl = mdapy.NeighborList.build(system, 4.0)
    np.testing.assert_array_equal(
        WarrenCowley(num_neighbors=12).compute(system, neighbors=nl).alpha,
        WarrenCowley(num_neighbors=12).compute(system).alpha,
    )


def test_warren_cowley_errors():
    with pytest.raises(MissingProperty, match="element column"):
        WarrenCowley(cutoff=3.0).compute(mdapy.System(np.zeros((2, 3)), mdapy.Box(5.0)))
    assert repr(WarrenCowley(num_neighbors=8)) == "WarrenCowley(num_neighbors=8)"
    assert repr(WarrenCowley(cutoff=3.0)) == "WarrenCowley(cutoff=3.0)"


@pytest.mark.parametrize(
    "make, message",
    [
        (lambda: WarrenCowley(), "either cutoff or num_neighbors"),
        (lambda: WarrenCowley(3.0, num_neighbors=8), "either cutoff or num_neighbors"),
        (lambda: WarrenCowley(-1.0), "cutoff must be positive"),
        (lambda: WarrenCowley(num_neighbors=0), "num_neighbors must be at least 1"),
        (lambda: BondAngleDistribution(0.0), "cutoff must be positive"),
        (lambda: BondAngleDistribution(3.0, bins=0), "bins must be at least 1"),
        (lambda: ClusterAnalysis(), "either cutoff or cutoffs"),
    ],
)
def test_invalid_options(make, message):
    with pytest.raises(InvalidParameter, match=message):
        make()
