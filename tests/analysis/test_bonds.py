# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

import numpy as np
import pytest

import mdapy
from mdapy.analysis import BondAngleDistribution, ClusterAnalysis, WarrenCowley
from mdapy.errors import InvalidParameter, MissingProperty

from lattices import crystal


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


def test_clusters_on_the_grid_and_from_a_list_agree():
    # without a list the bonds are found on the grid
    system = crystal("fcc", 3.6, (4, 4, 4), jitter=0.35, seed=6)
    rng = np.random.default_rng(6)
    system = system.with_columns(element=rng.choice(["Cu", "Ni", "Al"], len(system)))
    nl = mdapy.NeighborList.build(system, 3.0)
    for analysis in (
        ClusterAnalysis(2.1),
        ClusterAnalysis(cutoffs={("Cu", "Ni"): 2.5, ("Al", "Al"): 2.6}),
    ):
        grid, listed = analysis.compute(system), analysis.compute(system, neighbors=nl)
        np.testing.assert_array_equal(grid.cluster, listed.cluster)
        np.testing.assert_array_equal(grid.sizes, listed.sizes)
        assert len(grid.sizes) > 1
        np.testing.assert_array_equal(grid.spanning, listed.spanning)
        np.testing.assert_allclose(grid.masses, listed.masses, rtol=1e-12)
        # the same images, whatever the order of the bonds, for clusters that
        # do not span the box
        closed = ~grid.spanning
        atoms = closed[grid.cluster]
        np.testing.assert_allclose(
            grid.unwrapped[atoms], listed.unwrapped[atoms], rtol=0, atol=1e-12
        )
        for name in ("center_of_mass", "radius_of_gyration", "gyration_tensor"):
            np.testing.assert_allclose(
                getattr(grid, name)[closed],
                getattr(listed, name)[closed],
                rtol=1e-12,
                atol=1e-12,
            )
    assert grid.spanning.any() and not grid.spanning.all()  # both kinds occur


def test_clusters_unwrapped_across_the_boundary():
    # a water molecule split by the boundary at x = 0, and an argon atom
    system = mdapy.System(
        [[9.8, 5, 5], [0.6, 5, 5], [9.5, 5.9, 5], [4, 4, 4]],
        mdapy.Box(10.0),
        elements=["O", "H", "H", "Ar"],
    )
    result = ClusterAnalysis(cutoff=1.2).compute(system)
    np.testing.assert_array_equal(result.cluster, [0, 0, 0, 1])
    whole = np.array([[9.8, 5, 5], [10.6, 5, 5], [9.5, 5.9, 5]])  # O stays in place
    np.testing.assert_allclose(result.unwrapped[:3], whole, atol=1e-12)
    np.testing.assert_array_equal(result.unwrapped[3], [4, 4, 4])
    m = mdapy.elements.atomic_masses(mdapy.elements.atomic_numbers(["O", "H", "H"]))
    center = m @ whole / m.sum()
    delta = whole - center
    gyration = np.einsum("i,ia,ib->ab", m, delta, delta) / m.sum()
    np.testing.assert_allclose(result.masses[0], m.sum(), rtol=1e-12)
    np.testing.assert_allclose(result.center_of_mass[0], center, rtol=1e-12)
    np.testing.assert_allclose(result.gyration_tensor[0], gyration, atol=1e-12)
    np.testing.assert_allclose(
        result.radius_of_gyration[0], np.sqrt(np.trace(gyration)), rtol=1e-12
    )
    np.testing.assert_array_equal(result.center_of_mass[1], [4, 4, 4])
    assert result.radius_of_gyration[1] == 0.0
    # masses: a mass column first, else the elements, else unit masses
    heavy = system.with_columns(mass=[1.0, 1.0, 1.0, 5.0])
    weighted = ClusterAnalysis(cutoff=1.2).compute(heavy)
    np.testing.assert_allclose(weighted.center_of_mass[0], whole.mean(axis=0))
    np.testing.assert_array_equal(weighted.masses, [3.0, 5.0])
    bare = ClusterAnalysis(cutoff=1.2).compute(
        mdapy.System(system.positions, system.box)
    )
    np.testing.assert_array_equal(bare.masses, [3.0, 1.0])


def test_clusters_across_the_boundary_and_per_pair():
    # two halves of a chain joined only through the periodic boundary
    chain = mdapy.System(
        [[0.5, 5, 5], [9.5, 5, 5], [3.0, 5, 5]], mdapy.Box(10.0), elements="Ar"
    )
    result = ClusterAnalysis(cutoff=1.5).compute(chain)
    np.testing.assert_array_equal(result.cluster, [0, 0, 1])
    assert not result.spanning.any()
    # a chain through the box, and an atom bonded to its own image
    wire = mdapy.System([[0.5, 5, 5], [1.5, 5, 5], [2.5, 5, 5]], mdapy.Box(3.0))
    nl = mdapy.NeighborList.build(wire, 1.2)
    for given in (None, nl):
        assert ClusterAnalysis(1.2).compute(
            wire, neighbors=given
        ).spanning.tolist() == [True]
    tiny = mdapy.System([[0.5, 0.5, 0.5]], mdapy.Box([1.0, 9.0, 9.0]))
    for given in (None, mdapy.NeighborList.build(tiny, 1.2)):
        assert ClusterAnalysis(1.2).compute(
            tiny, neighbors=given
        ).spanning.tolist() == [True]
    system = b2()
    like = ClusterAnalysis(cutoffs={("Cu", "Cu"): 3.0}).compute(system)
    cu = system.table["element"].to_numpy() == "Cu"
    assert like.sizes[0] == cu.sum() and (like.sizes[1:] == 1).all()
    assert (like.cluster[cu] == 0).all()
    assert ClusterAnalysis(cutoff=2.6).compute(system).sizes.tolist() == [len(system)]
    assert "cutoffs" in repr(ClusterAnalysis(cutoffs={("Cu", "Cu"): 3.0}))
    plain = mdapy.System(system.positions, system.box)
    with pytest.raises(MissingProperty, match="element column"):
        ClusterAnalysis(cutoffs={("Cu", "Cu"): 3.0}).compute(plain)
    with pytest.raises(MissingProperty, match="element column"):
        ClusterAnalysis(cutoffs={("Cu", "Cu"): 3.0}).compute(
            plain, neighbors=mdapy.NeighborList.build(plain, 3.0)
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
    # by cutoff, counted on the grid or from a list
    np.testing.assert_array_equal(
        WarrenCowley(cutoff=3.0).compute(system, neighbors=nl).alpha, random.alpha
    )


def test_warren_cowley_errors():
    with pytest.raises(MissingProperty, match="element column"):
        WarrenCowley(cutoff=3.0).compute(mdapy.System(np.zeros((2, 3)), mdapy.Box(5.0)))
    assert repr(WarrenCowley(num_neighbors=8)) == "WarrenCowley(num_neighbors=8)"
    assert repr(WarrenCowley(cutoff=3.0)) == "WarrenCowley(cutoff=3.0)"
    assert repr(WarrenCowley(cutoff=(3, 4.5))) == "WarrenCowley(cutoff=[3.0, 4.5])"
    assert (
        repr(WarrenCowley(num_neighbors=(12, 6)))
        == "WarrenCowley(num_neighbors=[12, 6])"
    )
    with pytest.raises(InvalidParameter, match="must increase"):
        WarrenCowley(cutoff=(3.0, 3.0))
    with pytest.raises(InvalidParameter, match="positive"):
        WarrenCowley(cutoff=(0.0, 3.0))
    with pytest.raises(InvalidParameter, match="integers"):
        WarrenCowley(num_neighbors=(12.0, 6.0))
    with pytest.raises(InvalidParameter, match="at least 1"):
        WarrenCowley(num_neighbors=(12, 0))
    with pytest.raises(InvalidParameter, match="one shell"):
        WarrenCowley(num_neighbors=8).compute(b2())["Cu", "Zn", 0]


def test_warren_cowley_of_shells_in_l12():
    """Cu3Au-type order: around Au (the minority here), first the 12
    neighbors are all of the majority, then the 6 next all Au."""
    from mdapy.build import build_crystal

    system = build_crystal("l1_2", 3.75, elements=("Cu", "Au"), repeat=(4, 4, 4))
    by_count = WarrenCowley(num_neighbors=(12, 6)).compute(system)
    assert by_count.elements == ("Cu", "Au") and by_count.alpha.shape == (2, 2, 2)
    expected = [[[1 / 9, -1 / 3], [-1 / 3, 1.0]], [[-1 / 3, 1.0], [1.0, -3.0]]]
    np.testing.assert_allclose(by_count.alpha, expected, atol=1e-12)
    np.testing.assert_allclose(by_count["Au", "Au"], [1.0, -3.0])
    assert by_count["Au", "Cu", 1] == pytest.approx(1.0)
    # the same shells by distance: 2.65 and 3.75 Å
    by_distance = WarrenCowley(cutoff=(3.2, 4.2)).compute(system)
    np.testing.assert_allclose(by_distance.alpha, by_count.alpha, atol=1e-12)
    # one shell given as a sequence
    one = WarrenCowley(cutoff=[3.2]).compute(system)
    np.testing.assert_allclose(
        one.alpha[0], WarrenCowley(cutoff=3.2).compute(system).alpha
    )


@pytest.mark.parametrize("kind", ["cutoff", "num_neighbors"])
def test_warren_cowley_of_shells_match_numpy(kind):
    system = crystal("fcc", 3.6, (5, 5, 5), jitter=0.1, seed=6)
    rng = np.random.default_rng(6)
    elements = rng.choice(["Ni", "Cu", "Al"], len(system), p=[0.5, 0.3, 0.2])
    system = system.with_columns(element=elements)
    if kind == "cutoff":
        analysis = WarrenCowley(cutoff=(3.0, 4.0, 4.7))
    else:
        analysis = WarrenCowley(num_neighbors=(12, 6, 24))
    result = analysis.compute(system)
    # brute force: every pair of atoms, its image nearest, in shells
    positions, box = system.positions, system.box
    vectors = positions[None] - positions[:, None]
    fractional = vectors @ box.inverse
    fractional -= np.round(fractional)
    distances = np.linalg.norm(fractional @ box.matrix, axis=2)
    np.fill_diagonal(distances, np.inf)
    names = result.elements
    index = np.array([names.index(e) for e in elements])
    pairs = np.zeros((3, 3, 3))
    for i in range(len(system)):
        if kind == "cutoff":
            shell = np.searchsorted([3.0, 4.0, 4.7], distances[i], side="right")
            keep = shell < 3
            neighbors, shell = np.flatnonzero(keep), shell[keep]
        else:
            neighbors = np.argsort(distances[i], kind="stable")[:42]
            shell = np.searchsorted([12, 18, 42], np.arange(42), side="right")
        np.add.at(pairs, (shell, index[i], index[neighbors]), 1)
    fractions = np.bincount(index) / len(system)
    expected = 1 - pairs / pairs.sum(axis=2, keepdims=True) / fractions
    np.testing.assert_allclose(result.alpha, expected, atol=1e-12)
    # with a given list
    nl = mdapy.NeighborList.build(system, 5.0)
    np.testing.assert_array_equal(
        analysis.compute(system, neighbors=nl).alpha, result.alpha
    )


def test_bond_angles_by_triplet_of_elements_match_numpy():
    system = crystal("fcc", 3.6, jitter=0.2, seed=4)
    rng = np.random.default_rng(4)
    elements = rng.choice(["Ni", "Al", "Cu"], len(system))
    system = system.with_columns(element=elements)
    cutoffs = {("Ni", "Ni"): 3.0, ("Al", "Ni"): 3.2, ("Cu", "Al"): 2.9}
    result = BondAngleDistribution(bins=36, cutoffs=cutoffs).compute(system)
    # brute force: the bonds of each atom, then every two of them
    limit = {}
    for (a, b), r in cutoffs.items():
        limit[a, b] = limit[b, a] = r
    nl = mdapy.NeighborList.build(system, 3.2)
    order = {"Al": 13, "Ni": 28, "Cu": 29}
    expected = {}
    for i in range(len(system)):
        block = slice(nl.offsets[i], nl.offsets[i + 1])
        js, d, v = nl.j[block], nl.distances[block], nl.vectors[block]
        bonded = np.array(
            [
                d[k] < limit.get((elements[i], elements[j]), 0.0)
                for k, j in enumerate(js)
            ],
            dtype=bool,
        )
        js, v = js[bonded], v[bonded]
        u = v / np.linalg.norm(v, axis=1, keepdims=True)
        for a, b in zip(*np.triu_indices(len(u), 1)):
            pair = sorted((elements[js[a]], elements[js[b]]), key=order.get)
            angle = np.degrees(np.arccos(np.clip(u[a] @ u[b], -1, 1)))
            expected.setdefault((elements[i], *pair), []).append(angle)
    assert set(result.partial) == {
        (a, b, c) for a in order for b in order for c in order if order[b] <= order[c]
    }
    for key, counts in result.partial.items():
        reference, _ = np.histogram(expected.get(key, []), bins=36, range=(0, 180))
        np.testing.assert_array_equal(counts, reference)
    np.testing.assert_array_equal(sum(result.partial.values()), result.counts)
    np.testing.assert_array_equal(
        result.triplet("Ni", "Cu", "Al"), result.partial["Ni", "Al", "Cu"]
    )
    with mdapy.num_threads(3):
        again = BondAngleDistribution(bins=36, cutoffs=cutoffs).compute(system)
    np.testing.assert_array_equal(again.counts, result.counts)


def test_bond_angles_on_the_grid_and_from_a_list_agree():
    system = crystal("fcc", 3.6, (4, 4, 4), jitter=0.2, seed=8)
    rng = np.random.default_rng(8)
    system = system.with_columns(element=rng.choice(["Cu", "Ni", "Al"], len(system)))
    nl = mdapy.NeighborList.build(system, 3.5)
    for angles in (
        BondAngleDistribution(3.0, bins=90),
        BondAngleDistribution(bins=60, cutoffs={("Cu", "Ni"): 2.9, ("Al", "Al"): 3.1}),
    ):
        grid, listed = angles.compute(system), angles.compute(system, neighbors=nl)
        np.testing.assert_array_equal(grid.counts, listed.counts)
        assert grid.partial.keys() == listed.partial.keys()
        for key in grid.partial:
            np.testing.assert_array_equal(grid.partial[key], listed.partial[key])


def test_bond_angle_of_water():
    angle = np.radians(104.5)
    oxygen = np.array([5.0, 5.0, 5.0])
    hydrogens = oxygen + 0.96 * np.array(
        [[1.0, 0.0, 0.0], [np.cos(angle), np.sin(angle), 0.0]]
    )
    water = mdapy.System(
        np.vstack([oxygen, hydrogens]), mdapy.Box(10.0), elements=["O", "H", "H"]
    )
    result = BondAngleDistribution(bins=180, cutoffs={("O", "H"): 1.2}).compute(water)
    hoh = result.triplet("O", "H", "H")
    assert hoh.sum() == 1 and hoh[104] == 1
    assert result.triplet("H", "O", "O").sum() == 0
    assert result.counts.sum() == 1
    with pytest.raises(InvalidParameter, match="no partial distribution"):
        result.triplet("O", "C", "H")
    plain = BondAngleDistribution(1.2).compute(mdapy.System(water.positions, water.box))
    assert plain.partial == {} and plain.counts.sum() == 1


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
