# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import pytest

import mdapy
from mdapy.analysis import RingStatistics
from mdapy.build import build_crystal
from mdapy.errors import Cancelled, InvalidParameter, MissingProperty
from mdapy.transform import replicate


def cluster(points):
    return mdapy.System(np.asarray(points, float) + 10, mdapy.Box(40.0, pbc=False))


def theta(arms):
    """Two atoms joined by paths of the given numbers of bonds, in the
    topology, the atoms scattered."""
    bonds, atoms = [], 2
    for length in arms:
        chain = [0, *range(atoms, atoms + length - 1), 1]
        bonds += [[a, b] for a, b in zip(chain[:-1], chain[1:])]
        atoms += length - 1
    points = np.random.default_rng(0).random((atoms, 3)) * 20
    return cluster(points).with_topology(mdapy.Topology(bonds=bonds))


def ring_sets(result):
    return {frozenset(a[a >= 0].tolist()) for a in result.atoms}


@pytest.mark.parametrize(
    "arms, expected",
    [
        ((3, 3, 3), {6: 3}),  # three equal rings, each without a shortcut
        ((2, 3, 3), {5: 2}),  # the ring of the two long arms has a shortcut
    ],
)
def test_theta_graphs(arms, expected):
    result = RingStatistics(8).compute(theta(arms))
    assert {k: int(v) for k, v in enumerate(result.counts) if v} == expected


def test_a_cube_and_a_square_with_a_diagonal():
    """The cube: its six faces and the four hexagons about its body
    diagonals, whose opposite corners are three bonds apart either way; a
    square with one diagonal: two triangles, the square cut short."""
    cube = cluster([[i, j, k] for i in (0, 2) for j in (0, 2) for k in (0, 2)])
    result = RingStatistics(8, cutoff=2.1).compute(cube)
    assert result.counts.tolist() == [0, 0, 0, 0, 6, 0, 4, 0, 0]
    np.testing.assert_array_equal(result.atoms[0], [0, 1, 3, 2, -1, -1, -1, -1])
    np.testing.assert_array_equal(result.per_atom, np.full(8, 3 + 3))
    square = cluster([[0, 0, 0], [1.4, 0, 0], [1.4, 1.4, 0], [0, 1.4, 0]])
    square = square.with_topology(
        mdapy.Topology(bonds=[[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]])
    )
    result = RingStatistics(6).compute(square)
    assert ring_sets(result) == {frozenset({0, 1, 2}), frozenset({0, 2, 3})}
    assert RingStatistics(6, min_size=4).compute(square).counts.sum() == 0


@pytest.mark.parametrize("repeat", [2, 3, 4])
def test_diamond_and_graphene(repeat):
    """Every atom of diamond is in twelve six-rings and of graphene in
    three, the only rings up to twelve atoms; paths around the periodic box
    are no rings, however small it is."""
    diamond = build_crystal("diamond", 3.567, elements="C", repeat=(repeat,) * 3)
    result = RingStatistics(12, cutoff=1.7).compute(diamond)
    assert result.counts.sum() == result.counts[6] == 2 * len(diamond)
    np.testing.assert_array_equal(result.per_atom, 12)
    graphene = build_crystal(
        "graphene", 2.46, c=10.0, elements="C", repeat=(repeat, repeat, 1)
    )
    result = RingStatistics(12, cutoff=1.6).compute(graphene)
    assert result.counts.sum() == result.counts[6] == len(graphene) // 2
    np.testing.assert_array_equal(result.per_atom, 3)


def disordered(seed=1, noise=0.3):
    """A network of bonds of shaken diamond in a periodic box, rings of many
    sizes."""
    crystal = build_crystal("diamond", 3.567, elements="C", repeat=(3, 3, 3))
    rng = np.random.default_rng(seed)
    return crystal.with_columns(
        positions=crystal["positions"] + rng.normal(0, noise, (len(crystal), 3))
    )


def test_against_all_cycles():
    """Every cycle of the bonds of a disordered cluster, up to nine bonds,
    whose atoms are no closer through the network than along it."""
    nx = pytest.importorskip("networkx")
    crystal = disordered()
    middle = crystal.box.matrix.sum(axis=0) / 2
    keep = np.linalg.norm(crystal["positions"] - middle, axis=1) < 8
    system = mdapy.System(crystal["positions"][keep], mdapy.Box(60.0, pbc=False))
    result = RingStatistics(9, cutoff=1.8).compute(system)
    pairs = mdapy.NeighborList.build(system, 1.8)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(system)))
    owner = np.repeat(np.arange(len(system)), np.diff(pairs.offsets))
    graph.add_edges_from(zip(owner.tolist(), pairs.j.tolist()))
    distance = dict(nx.all_pairs_shortest_path_length(graph, cutoff=9))
    expected = set()
    for cycle in nx.simple_cycles(graph, length_bound=9):
        n = len(cycle)
        if n >= 3 and all(
            distance[cycle[i]].get(cycle[j], n) == min(j - i, n - j + i)
            for i in range(n)
            for j in range(i + 1, n)
        ):
            expected.add(frozenset(cycle))
    assert ring_sets(result) == expected
    assert len(set(result.sizes.tolist())) >= 4


def test_periodic_invariance():
    """The rings of a periodic network do not depend on where the box cuts
    it, and a box of copies has as many more of them."""
    system = disordered(seed=2, noise=0.25)
    result = RingStatistics(10, cutoff=1.8).compute(system)
    moved = system.with_columns(positions=system["positions"] + [4.0, -3.0, 7.5])
    again = RingStatistics(10, cutoff=1.8).compute(mdapy.transform.wrap(moved))
    assert ring_sets(again) == ring_sets(result)
    np.testing.assert_array_equal(again.counts, result.counts)
    copies = RingStatistics(10, cutoff=1.8).compute(replicate(system, 2))
    np.testing.assert_array_equal(copies.counts, 8 * result.counts)
    # each ring whole: the steps from atom to atom are bonds
    box = system.box
    for atoms, shifts, n in zip(result.atoms, result.shifts, result.sizes):
        points = system["positions"][atoms[:n]] + shifts[:n] @ box.matrix
        steps = np.linalg.norm(np.diff(points[np.r_[0:n, 0]], axis=0), axis=1)
        assert steps.max() < 1.8


def test_bonds_from_the_topology_and_per_pair():
    """The bonds of the topology give the rings of the same bonds found by
    a cutoff; bonds per pair of elements, of boron nitride."""
    system = disordered(seed=3, noise=0.2)
    by_cutoff = RingStatistics(8, cutoff=1.8).compute(system)
    pairs = mdapy.NeighborList.build(system, 1.8)
    owner = np.repeat(np.arange(len(system)), np.diff(pairs.offsets))
    half = owner < pairs.j
    bonds = np.stack([owner[half], pairs.j[half]], axis=1)
    with_bonds = system.with_topology(mdapy.Topology(bonds=bonds))
    by_topology = RingStatistics(8).compute(with_bonds)
    assert ring_sets(by_topology) == ring_sets(by_cutoff)
    np.testing.assert_array_equal(by_topology.shifts, by_cutoff.shifts)
    # cubic boron nitride: bonds only between boron and nitrogen
    boron_nitride = build_crystal(
        "zincblende", 3.615, elements=("B", "N"), repeat=(3, 3, 3)
    )
    result = RingStatistics(8, cutoffs={("B", "N"): 1.7}).compute(boron_nitride)
    assert result.counts.sum() == result.counts[6] == 2 * len(boron_nitride)
    none = RingStatistics(8, cutoffs={("B", "B"): 1.7}).compute(boron_nitride)
    assert none.counts.sum() == 0


def test_polygons_and_results():
    """A fan of triangles about each ring, the hexagons of graphene of their
    area; the number of rings of each atom as a column."""
    sheet = build_crystal("graphene", 2.46, c=10.0, elements="C", repeat=(3, 3, 1))
    result = RingStatistics(8, cutoff=1.6).compute(sheet)
    mesh, ring = result.polygons(sheet)
    assert len(mesh) == 6 * result.counts[6] and len(ring) == len(mesh)
    bond = 2.46 / np.sqrt(3)
    assert mesh.area == pytest.approx(result.counts[6] * 3 * np.sqrt(3) / 2 * bond**2)
    np.testing.assert_array_equal(np.bincount(ring), 6)
    assert sheet.with_columns(result)["rings"].tolist() == result.per_atom.tolist()
    empty, ring = (
        RingStatistics(8, min_size=7, cutoff=1.6).compute(sheet).polygons(sheet)
    )
    assert len(empty) == 0 and len(ring) == 0
    assert (
        repr(RingStatistics(8, cutoff=1.6))
        == "RingStatistics(8, min_size=3, cutoff=1.6)"
    )
    assert repr(RingStatistics(6)) == "RingStatistics(6, min_size=3)"


def test_threads_progress_and_cancel():
    system = disordered()
    found = []
    for threads in (1, 4):
        with mdapy.num_threads(threads):
            found.append(RingStatistics(9, cutoff=1.8).compute(system))
    np.testing.assert_array_equal(found[0].atoms, found[1].atoms)
    np.testing.assert_array_equal(found[0].shifts, found[1].shifts)
    calls = []
    RingStatistics(8, cutoff=1.8).compute(
        system, progress=lambda done, total: calls.append(done)
    )
    assert calls and calls[-1] == len(system)
    token = mdapy.CancelToken()
    token.cancel()
    with pytest.raises(Cancelled):
        RingStatistics(8, cutoff=1.8).compute(system, cancel=token)


def test_errors():
    for kwargs in (
        {"max_size": 2},
        {"max_size": 37},
        {"max_size": 6, "min_size": 7},
        {"max_size": 6.5},
        {"max_size": True},
    ):
        with pytest.raises(InvalidParameter):
            RingStatistics(**kwargs)
    with pytest.raises(InvalidParameter, match="either cutoff or cutoffs"):
        RingStatistics(8, cutoff=1.5, cutoffs={("C", "C"): 1.5})
    bare = cluster([[0, 0, 0], [1, 0, 0]])
    with pytest.raises(MissingProperty, match="no bonds"):
        RingStatistics(8).compute(bare)
    with pytest.raises(MissingProperty, match="element"):
        RingStatistics(8, cutoffs={("C", "C"): 1.5}).compute(bare)
