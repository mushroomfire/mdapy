# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import pytest
from numpy.polynomial import legendre

import mdapy
from mdapy.analysis import SteinhardtParameters
from mdapy.errors import InvalidParameter

from lattices import crystal, icosahedron


def unit_vectors(system, k):
    nl = mdapy.NeighborList.build(system, num_neighbors=k)
    v = nl.vectors.reshape(len(system), k, 3)
    return v / np.linalg.norm(v, axis=2, keepdims=True), nl.j.reshape(len(system), k)


def reference(system, degree, k, average=False):
    """q_l from the addition theorem: (4 pi / (2l + 1)) sum_m q_lm(a) q_lm(b)*
    is the mean of P_l over the bonds of a and b."""
    u, j = unit_vectors(system, k)

    def overlap(a, b):
        return legendre.legval(u[a] @ u[b].T, [0] * degree + [1]).mean()

    q = []
    for i in range(len(system)):
        members = [i, *j[i]] if average else [i]
        total = sum(overlap(a, b) for a in members for b in members)
        q.append(np.sqrt(total) / len(members))
    return np.array(q)


@pytest.mark.parametrize(
    "kind, a, k, q4, q6",
    [
        ("fcc", 3.6, 12, 0.190941, 0.574524),
        ("hcp", 2.5, 12, 0.097222, 0.484762),
        ("bcc", 2.87, 8, 0.509175, 0.628539),
        ("bcc", 2.87, 14, 0.036369, 0.510688),
    ],
)
def test_perfect_crystals(kind, a, k, q4, q6):
    """Values of the literature, to their six digits."""
    result = SteinhardtParameters(num_neighbors=k).compute(crystal(kind, a))
    np.testing.assert_allclose(result[4], q4, atol=1e-6)
    np.testing.assert_allclose(result[6], q6, atol=1e-6)
    averaged = SteinhardtParameters(num_neighbors=k, average=True).compute(
        crystal(kind, a)
    )
    np.testing.assert_allclose(averaged.q, result.q, rtol=1e-12)  # all atoms alike


def test_icosahedron_and_cutoff():
    center = SteinhardtParameters((6,), num_neighbors=12).compute(icosahedron())[6][0]
    assert center == pytest.approx(np.sqrt(11) / 5, abs=1e-12)  # 0.663325
    fcc = crystal("fcc", 3.6)
    by_cutoff = SteinhardtParameters(cutoff=3.0).compute(fcc)
    np.testing.assert_allclose(
        by_cutoff.q, SteinhardtParameters().compute(fcc).q, rtol=1e-12
    )


@pytest.mark.parametrize("average", [False, True])
def test_matches_the_addition_theorem(average):
    system = crystal("fcc", 3.6, jitter=0.2, seed=1)
    degrees = tuple(range(13))
    result = SteinhardtParameters(degrees, average=average).compute(system)
    np.testing.assert_allclose(result[0], 1.0, rtol=1e-12)
    for degree in (1, 2, 4, 6, 8, 12):
        np.testing.assert_allclose(
            result[degree],
            reference(system, degree, 12, average),
            rtol=1e-10,
            atol=1e-12,
        )


def test_rotation_threads_and_columns():
    system = crystal("bcc", 2.87, jitter=0.1, seed=2)
    q, _ = np.linalg.qr(np.random.default_rng(3).normal(size=(3, 3)))
    turned = mdapy.System(
        system["positions"] @ q.T, mdapy.Box(system.box.matrix @ q.T), elements="Cu"
    )
    parameters = SteinhardtParameters((4, 6, 8), num_neighbors=8, average=True)
    result = parameters.compute(system)
    np.testing.assert_allclose(parameters.compute(turned).q, result.q, rtol=1e-10)
    with mdapy.num_threads(3):
        np.testing.assert_array_equal(parameters.compute(system).q, result.q)
    nl = mdapy.NeighborList.build(system, 4.0)
    np.testing.assert_array_equal(parameters.compute(system, neighbors=nl).q, result.q)
    for k in (8, 10):  # lists of nearest neighbors, used as they are or cut
        nearest = mdapy.NeighborList.build(system, num_neighbors=k)
        np.testing.assert_array_equal(
            parameters.compute(system, neighbors=nearest).q, result.q
        )
    table = system.with_columns(result).table
    assert {"q4_average", "q6_average", "q8_average"} <= set(table.columns)
    assert list(SteinhardtParameters((6,)).compute(system).columns()) == ["q6"]
    assert repr(parameters) == (
        "SteinhardtParameters(degrees=(4, 6, 8), num_neighbors=8, average=True)"
    )
    assert "cutoff=3.0" in repr(SteinhardtParameters(cutoff=3.0))
    with pytest.raises(InvalidParameter, match="degree 2 was not computed"):
        result[2]
    with pytest.raises(InvalidParameter, match="fewer than 8"):
        parameters.compute(system, neighbors=nl.within(2.0))


def test_atoms_without_neighbors():
    pair = mdapy.System(
        [[0.0, 0, 0], [5.0, 0, 0]], mdapy.Box(20.0, pbc=False), elements="Ar"
    )
    result = SteinhardtParameters((0, 6), cutoff=2.0).compute(pair)
    np.testing.assert_array_equal(result.q, 0.0)


@pytest.mark.parametrize(
    "options, message",
    [
        ({"degrees": ()}, "degrees must be from 0 to 20"),
        ({"degrees": (21,)}, "degrees must be from 0 to 20"),
        ({"num_neighbors": 4, "cutoff": 3.0}, "give one of"),
        ({"voronoi": True, "cutoff": 3.0}, "give one of"),
        ({"weighted": True}, "weighted needs voronoi"),
        ({"cutoff": 0.0}, "cutoff must be positive"),
        ({"num_neighbors": 0}, "num_neighbors must be at least 1"),
    ],
)
def test_invalid_options(options, message):
    with pytest.raises(InvalidParameter, match=message):
        SteinhardtParameters(**options)


def weighted_reference(nl, degree, weights, average=False):
    """q_l of every atom from the addition theorem over the pairs of a list,
    each pair weighted: (4 pi / (2l + 1)) sum_m q_lm(a) q_lm(b)* is the
    weighted mean of P_l over the pairs of a and b."""
    u = nl.vectors / nl.distances[:, None]
    offsets = nl.offsets

    def overlap(a, b):
        sa, sb = slice(offsets[a], offsets[a + 1]), slice(offsets[b], offsets[b + 1])
        if sa.start == sa.stop or sb.start == sb.stop:
            return 0.0
        p = legendre.legval(u[sa] @ u[sb].T, [0] * degree + [1])
        return weights[sa] @ p @ weights[sb] / (weights[sa].sum() * weights[sb].sum())

    q = []
    for i in range(len(nl.counts)):
        members = [i, *nl.j[offsets[i] : offsets[i + 1]]] if average else [i]
        total = sum(overlap(a, b) for a in members for b in members)
        q.append(np.sqrt(total) / len(members))
    return np.array(q)


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("average", [False, True])
def test_voronoi_neighbors(weighted, average):
    system = crystal("bcc", 2.87, jitter=0.15, seed=4)
    nl = mdapy.NeighborList.build(system, voronoi=True)
    parameters = SteinhardtParameters(
        (2, 4, 6, 8), voronoi=True, weighted=weighted, average=average
    )
    result = parameters.compute(system)
    np.testing.assert_array_equal(parameters.compute(system, neighbors=nl).q, result.q)
    weights = nl.areas if weighted else np.ones(len(nl))
    for degree in (2, 4, 6, 8):
        np.testing.assert_allclose(
            result[degree],
            weighted_reference(nl, degree, weights, average),
            rtol=1e-10,
            atol=1e-12,
        )
    # the weights matter where faces differ
    other = SteinhardtParameters((6,), voronoi=True, weighted=not weighted)
    assert not np.allclose(other.compute(system)[6], result[6])


@pytest.mark.parametrize("kind, a, k", [("fcc", 3.6, 12), ("bcc", 2.87, 14)])
def test_voronoi_neighbors_of_perfect_crystals(kind, a, k):
    """fcc: twelve equal faces, the twelve nearest neighbors; bcc: the 8 + 6
    nearest, on faces of two sizes."""
    system = crystal(kind, a)
    nearest = SteinhardtParameters((4, 6), num_neighbors=k).compute(system)
    voronoi = SteinhardtParameters((4, 6), voronoi=True).compute(system)
    np.testing.assert_allclose(voronoi.q, nearest.q, rtol=1e-12)
    weighted = SteinhardtParameters((4, 6), voronoi=True, weighted=True).compute(system)
    if kind == "fcc":
        np.testing.assert_allclose(weighted.q, nearest.q, rtol=1e-12)
    else:
        nl = mdapy.NeighborList.build(system, voronoi=True)
        np.testing.assert_allclose(
            weighted[6], weighted_reference(nl, 6, nl.areas), rtol=1e-12
        )
        np.testing.assert_allclose(weighted[4], 0.224025, atol=1e-6)
        np.testing.assert_allclose(weighted[6], 0.566940, atol=1e-6)
    assert repr(SteinhardtParameters((6,), voronoi=True, weighted=True)) == (
        "SteinhardtParameters(degrees=(6,), voronoi=True, weighted=True, average=False)"
    )


def test_neighbor_lists_that_do_not_fit():
    system = crystal("fcc", 3.6)
    voronoi = mdapy.NeighborList.build(system, voronoi=True)
    cutoff = mdapy.NeighborList.build(system, 3.0)
    with pytest.raises(InvalidParameter, match="does not hold Voronoi neighbors"):
        SteinhardtParameters(voronoi=True).compute(system, neighbors=cutoff)
    with pytest.raises(InvalidParameter, match="Voronoi neighbors does not hold every"):
        SteinhardtParameters(cutoff=3.0).compute(system, neighbors=voronoi)
    with pytest.raises(InvalidParameter, match="does not hold the nearest"):
        SteinhardtParameters().compute(system, neighbors=voronoi)
    # the same crystal made again serves; a strained one does not
    again = crystal("fcc", 3.6)
    SteinhardtParameters(voronoi=True).compute(again, neighbors=voronoi)
    other = crystal("fcc", 3.61)
    with pytest.raises(InvalidParameter, match="other positions or another box"):
        SteinhardtParameters(voronoi=True).compute(other, neighbors=voronoi)


@pytest.mark.parametrize("box", ["periodic", "triclinic", "cluster"])
def test_nearest_neighbors_found_as_they_go_give_what_a_list_gives(box):
    system = crystal("fcc", 3.6, (3, 3, 3), jitter=0.15, seed=7)
    if box == "triclinic":
        tilted = system.box.matrix + [[0, 0, 0], [1.3, 0, 0], [0.4, -0.9, 0]]
        system = mdapy.System(system["positions"], mdapy.Box(tilted), elements="Cu")
    elif box == "cluster":
        system = system.with_box(mdapy.Box(system.box.matrix, pbc=False))
    for k, degrees in [(8, (4, 6)), (12, (2, 3, 4, 5, 6, 7, 8)), (16, (6, 12))]:
        parameters = SteinhardtParameters(degrees, num_neighbors=k)
        nl = mdapy.NeighborList.build(system, num_neighbors=k)
        np.testing.assert_array_equal(
            parameters.compute(system).q,
            parameters.compute(system, neighbors=nl).q,
        )
    small = mdapy.System(system["positions"][:10], mdapy.Box(20.0, pbc=False))
    with pytest.raises(InvalidParameter, match="has only 10 atoms"):
        SteinhardtParameters((6,), num_neighbors=12).compute(small)


def solid_reference(nl, degree, weights, threshold):
    """Solid-like bonds from the addition theorem: the product of the
    harmonics of two atoms is the weighted mean of P_l over their bonds, up to
    a factor that the normalization removes."""
    u = nl.vectors / nl.distances[:, None]
    offsets = nl.offsets

    def overlap(a, b):
        sa, sb = slice(offsets[a], offsets[a + 1]), slice(offsets[b], offsets[b + 1])
        if sa.start == sa.stop or sb.start == sb.stop:
            return 0.0
        p = legendre.legval(u[sa] @ u[sb].T, [0] * degree + [1])
        return weights[sa] @ p @ weights[sb] / (weights[sa].sum() * weights[sb].sum())

    norms = np.sqrt([overlap(i, i) for i in range(len(nl.counts))])
    bonds = np.zeros(len(nl.counts), dtype=int)
    for i in range(len(nl.counts)):
        for j in nl.j[offsets[i] : offsets[i + 1]]:
            if norms[i] > 0 and norms[j] > 0:
                bonds[i] += overlap(i, j) / (norms[i] * norms[j]) > threshold
    return bonds


def half_molten(seed=0):
    """fcc copper, its upper half disordered like a liquid."""
    system = crystal("fcc", 3.6, (5, 5, 5), jitter=0.05, seed=seed)
    rng = np.random.default_rng(seed)
    upper = system["positions"][:, 2] > 9.0
    shake = rng.normal(0, 0.6, (len(system), 3)) * upper[:, None]
    return system.with_columns(positions=system["positions"] + shake), upper


@pytest.mark.parametrize(
    "neighbors",
    [
        {"num_neighbors": 12},
        {"cutoff": 3.1},
        {"voronoi": True},
        {"voronoi": True, "weighted": True},
    ],
)
def test_solid_bonds_match_the_addition_theorem(neighbors):
    system, _ = half_molten()
    result = SteinhardtParameters(solid=True, **neighbors).compute(system)
    if neighbors.get("voronoi"):
        nl = mdapy.NeighborList.build(system, voronoi=True)
    elif "cutoff" in neighbors:
        nl = mdapy.NeighborList.build(system, neighbors["cutoff"])
    else:
        nl = mdapy.NeighborList.build(system, num_neighbors=12)
    weights = nl.areas if neighbors.get("weighted") else np.ones(len(nl))
    np.testing.assert_array_equal(
        result.solid_bonds, solid_reference(nl, 6, weights, 0.7)
    )
    np.testing.assert_array_equal(result.solid, result.solid_bonds >= 7)
    # the same from a given list
    again = SteinhardtParameters(solid=True, **neighbors).compute(system, neighbors=nl)
    np.testing.assert_array_equal(again.solid_bonds, result.solid_bonds)


def test_solid_and_liquid_halves():
    system, upper = half_molten(seed=3)
    result = SteinhardtParameters(solid=True).compute(system)
    assert result.solid[~upper].mean() > 0.95
    assert result.solid[upper].mean() < 0.05
    assert result.solid_bonds.dtype == np.int32
    columns = result.columns()
    assert set(columns) == {"q4", "q6", "solid", "solid_bonds"}
    np.testing.assert_array_equal(system.with_columns(result)["solid"], result.solid)
    # other degree, threshold and count of bonds
    loose = SteinhardtParameters(
        solid=True, solid_degree=4, bond_threshold=0.5, min_solid_bonds=3
    ).compute(system)
    assert loose.solid.sum() >= result.solid.sum()
    np.testing.assert_array_equal(loose.solid, loose.solid_bonds >= 3)


@pytest.mark.parametrize(
    "kind, a, k", [("fcc", 3.6, 12), ("bcc", 2.87, 8), ("hcp", 3.2, 12)]
)
def test_perfect_crystals_are_solid(kind, a, k):
    system = crystal(kind, a)
    result = SteinhardtParameters(num_neighbors=k, solid=True).compute(system)
    np.testing.assert_array_equal(result.solid_bonds, k)
    assert result.solid.all()


def test_solid_bonds_ignore_rotation_and_threads():
    system, _ = half_molten(seed=5)
    parameters = SteinhardtParameters(solid=True, average=True)
    result = parameters.compute(system)
    rotation = np.linalg.qr(np.random.default_rng(6).normal(size=(3, 3)))[0]
    turned = mdapy.System(
        system["positions"] @ rotation.T,
        mdapy.Box(system.box.matrix @ rotation.T),
        elements="Cu",
    )
    np.testing.assert_array_equal(
        parameters.compute(turned).solid_bonds, result.solid_bonds
    )
    with mdapy.num_threads(1):
        np.testing.assert_array_equal(
            parameters.compute(system).solid_bonds, result.solid_bonds
        )
    # the bonds are those of the harmonics before averaging
    plain = SteinhardtParameters(solid=True).compute(system)
    np.testing.assert_array_equal(plain.solid_bonds, result.solid_bonds)
    assert repr(parameters) == (
        "SteinhardtParameters(degrees=(4, 6), num_neighbors=12, average=True, solid=True, "
        "solid_degree=6, bond_threshold=0.7, min_solid_bonds=7)"
    )


def test_atoms_without_neighbors_have_no_solid_bonds():
    pair = mdapy.System(
        [[0.0, 0, 0], [5.0, 0, 0]], mdapy.Box(20.0, pbc=False), elements="Ar"
    )
    result = SteinhardtParameters(
        (6,), cutoff=2.0, solid=True, min_solid_bonds=0
    ).compute(pair)
    np.testing.assert_array_equal(result.solid_bonds, 0)
    assert result.solid.all()  # at least 0 bonds


@pytest.mark.parametrize(
    "options, message",
    [
        ({"solid_degree": 21}, "solid_degree must be from 0 to 20"),
        ({"solid_degree": 6.5}, "solid_degree must be from 0 to 20"),
        ({"bond_threshold": 1.5}, "bond_threshold must be from -1 to 1"),
        ({"min_solid_bonds": -1}, "min_solid_bonds must be an integer"),
        ({"min_solid_bonds": 2.5}, "min_solid_bonds must be an integer"),
    ],
)
def test_invalid_solid_options(options, message):
    with pytest.raises(InvalidParameter, match=message):
        SteinhardtParameters(solid=True, **options)
