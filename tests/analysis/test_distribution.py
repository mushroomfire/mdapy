# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

import numpy as np
import pytest
from mdapy._core import analysis as _analysis

import mdapy
from mdapy.analysis import CoordinationNumber, RadialDistributionFunction
from mdapy.errors import InvalidParameter, MissingProperty

from lattices import crystal


def alloy(repeat=(3, 3, 3), seed=0, jitter=0.1):
    system = crystal("fcc", 3.6, repeat, jitter=jitter, seed=seed)
    elements = np.random.default_rng(seed).choice(["Cu", "Ni", "Al"], size=len(system))
    return system.with_columns(element=elements)


def b2(a=2.95):
    """CsCl order on a bcc lattice: Cu at the corners, Zn at the centers."""
    system = crystal("bcc", a, (3, 3, 3))
    return system.with_columns(element=["Cu", "Zn"] * (len(system) // 2))


def test_fcc_shells():
    a = 3.6
    result = RadialDistributionFunction(7.0, bins=700).compute(
        crystal("fcc", a, (4, 4, 4))
    )
    # neighbors closer than a point between each shell and the next
    for r, count in [
        (2.6, 12),
        (3.7, 18),
        (4.5, 42),
        (5.2, 54),
        (5.8, 78),
        (6.3, 86),
        (6.8, 134),
    ]:
        assert result.coordination[np.searchsorted(result.edges, r) - 1] == count
    np.testing.assert_allclose(result.r, np.arange(700) * 0.01 + 0.005)
    first = np.argmax(result.g)
    assert result.edges[first] <= a / np.sqrt(2) < result.edges[first + 1]


def test_normalization_and_sum_rule():
    system = alloy()
    result = RadialDistributionFunction(6.0, bins=60).compute(system)
    n, volume = len(system), system.box.volume
    shells = 4 / 3 * np.pi * np.diff(result.edges**3)
    pairs = len(mdapy.NeighborList.build(system, 6.0).within(6.0))
    np.testing.assert_allclose((result.g * n * n / volume * shells).sum(), pairs)
    np.testing.assert_allclose(result.coordination[-1], pairs / n)


def test_ideal_gas_is_one():
    rng = np.random.default_rng(1)
    gas = mdapy.System(
        rng.uniform(0, 30, size=(20000, 3)), mdapy.Box(30.0), elements="Ar"
    )
    result = RadialDistributionFunction(8.0, bins=16).compute(gas)
    np.testing.assert_allclose(result.g[2:], 1.0, atol=0.02)


def test_partial_functions():
    system = alloy(seed=2)
    result = RadialDistributionFunction(6.0, bins=50).compute(system)
    # by atomic number: Al 13, Ni 28, Cu 29
    assert list(result.partial) == [
        ("Al", "Al"),
        ("Al", "Ni"),
        ("Al", "Cu"),
        ("Ni", "Ni"),
        ("Ni", "Cu"),
        ("Cu", "Cu"),
    ]
    fractions = {
        e: np.mean(system.table["element"].to_numpy() == e) for e in ("Al", "Cu", "Ni")
    }
    total = sum(
        fractions[a] * fractions[b] * result.pair(a, b)
        for a in fractions
        for b in fractions
    )
    np.testing.assert_allclose(total, result.g, atol=1e-12)
    assert result.pair("Ni", "Cu") is result.pair("Cu", "Ni")
    with pytest.raises(InvalidParameter, match="no partial function"):
        result.pair("Cu", "Zn")


def test_a_supercell_has_the_same_function():
    system = alloy(seed=3)
    a = system.box.matrix[0]
    positions = np.vstack([system["positions"], system["positions"] + a])
    double = mdapy.System(
        positions,
        mdapy.Box(system.box.matrix * [[2], [1], [1]]),
        elements=np.tile(system.table["element"].to_numpy(), 2),
    )
    one = RadialDistributionFunction(6.0, bins=40).compute(system)
    two = RadialDistributionFunction(6.0, bins=40).compute(double)
    np.testing.assert_allclose(two.g, one.g, rtol=1e-12)
    np.testing.assert_allclose(two.pair("Cu", "Ni"), one.pair("Cu", "Ni"), rtol=1e-12)


def test_rdf_options_and_errors():
    system = crystal("fcc", 3.6)
    nl = mdapy.NeighborList.build(system, 8.0)
    rdf = RadialDistributionFunction(6.0, bins=30)
    np.testing.assert_array_equal(
        rdf.compute(system, neighbors=nl).g, rdf.compute(system).g
    )
    mixed = alloy(seed=2, jitter=0.2)  # every pair of elements, from a list
    listed = rdf.compute(mixed, neighbors=mdapy.NeighborList.build(mixed, 6.0))
    searched = rdf.compute(mixed)
    np.testing.assert_array_equal(listed.coordination, searched.coordination)
    for pair in searched.partial:
        np.testing.assert_array_equal(listed.partial[pair], searched.partial[pair])
    plain = mdapy.System(system["positions"], system.box)
    assert rdf.compute(plain).partial == {}
    empty = mdapy.System(np.zeros((0, 3)), mdapy.Box(5.0), elements=[])
    assert not rdf.compute(empty).g.any() and rdf.compute(empty).partial == {}
    assert repr(rdf) == "RadialDistributionFunction(r_max=6.0, bins=30)"
    assert rdf.compute(system).columns() == {}
    with pytest.raises(InvalidParameter, match="below the required"):
        rdf.compute(system, neighbors=mdapy.NeighborList.build(system, 3.0))
    with pytest.raises(InvalidParameter, match="r_max must be positive"):
        RadialDistributionFunction(0.0)
    with pytest.raises(InvalidParameter, match="bins must be at least 1"):
        RadialDistributionFunction(5.0, bins=0)


def test_coordination():
    system = alloy(seed=4, jitter=0.0)
    result = CoordinationNumber(3.0).compute(system)
    np.testing.assert_array_equal(result.coordination, 12)
    np.testing.assert_array_equal(sum(result.partial.values()), result.coordination)
    table = system.with_columns(result).table
    assert {
        "coordination",
        "coordination_Al",
        "coordination_Cu",
        "coordination_Ni",
    } <= set(table.columns)
    plain = mdapy.System(system["positions"], system.box)
    assert CoordinationNumber(3.0).compute(plain).partial == {}


def test_coordination_per_pair_of_elements():
    system = b2()
    cu = system.table["element"].to_numpy() == "Cu"
    unlike = CoordinationNumber(cutoffs={("Zn", "Cu"): 2.7}).compute(system)
    np.testing.assert_array_equal(unlike.coordination, 8)
    np.testing.assert_array_equal(unlike.partial["Zn"][cu], 8)
    np.testing.assert_array_equal(unlike.partial["Cu"][cu], 0)
    like = CoordinationNumber(cutoffs={("Cu", "Cu"): 3.0, ("Zn", "Zn"): 3.0}).compute(
        system
    )
    np.testing.assert_array_equal(like.coordination, 6)
    both = CoordinationNumber(cutoffs={("Cu", "Zn"): 2.7, ("Cu", "Cu"): 3.0})
    np.testing.assert_array_equal(
        both.compute(system).coordination, np.where(cu, 14, 8)
    )
    assert (
        "cutoffs" in repr(both)
        and repr(CoordinationNumber(2.0)) == "CoordinationNumber(cutoff=2.0)"
    )


def test_kernels_of_lists_check_their_arguments():
    offsets, j, d = np.array([0, 1, 2]), np.array([1, 0], np.int32), np.ones(2)
    types = np.zeros(2, np.int32)
    histogram = _analysis.pair_histogram_list(offsets, j, d, 2.0, 4, types, 1)
    np.testing.assert_array_equal(histogram.ravel(), [0, 0, 2, 0])
    counts = _analysis.count_neighbors_list(
        offsets, j, d, types, 1, np.full((1, 1), 2.0)
    )
    np.testing.assert_array_equal(counts.ravel(), [1, 1])
    for call, message in [
        (
            lambda: _analysis.pair_histogram_list(offsets, j, d[:1], 2.0, 4, types, 1),
            "fit",
        ),
        (
            lambda: _analysis.pair_histogram_list(offsets, j, d, 2.0, 0, types, 1),
            "positive",
        ),
        (
            lambda: _analysis.pair_histogram_list(offsets, j, d, 2.0, 4, types + 1, 1),
            "range",
        ),
        (
            lambda: _analysis.count_neighbors_list(
                offsets, j, d, types, 1, np.ones((2, 2))
            ),
            "limits",
        ),
    ]:
        with pytest.raises(ValueError, match=message):
            call()


def test_coordination_on_the_grid_and_from_a_list_agree():
    # without a list the neighbors are counted on the grid
    system = alloy(seed=5, jitter=0.25)
    nl = mdapy.NeighborList.build(system, 4.0)
    for counter in (
        CoordinationNumber(3.0),
        CoordinationNumber(cutoffs={("Cu", "Ni"): 2.9, ("Al", "Al"): 3.1}),
    ):
        grid, listed = counter.compute(system), counter.compute(system, neighbors=nl)
        np.testing.assert_array_equal(grid.coordination, listed.coordination)
        for element in grid.partial:
            np.testing.assert_array_equal(
                grid.partial[element], listed.partial[element]
            )
    plain = mdapy.System(system["positions"], system.box)
    np.testing.assert_array_equal(
        CoordinationNumber(3.0).compute(plain).coordination,
        CoordinationNumber(3.0)
        .compute(plain, neighbors=mdapy.NeighborList.build(plain, 3.5))
        .coordination,
    )


@pytest.mark.parametrize(
    "options, message",
    [
        ({}, "either cutoff or cutoffs"),
        ({"cutoff": 2.0, "cutoffs": {}}, "either cutoff or cutoffs"),
        ({"cutoff": -1.0}, "cutoff must be positive"),
        ({"cutoffs": {("Cu", "Qq"): 2.0}}, "unknown element 'Qq'"),
        ({"cutoffs": {("Cu", "Cu"): 0.0}}, "must be positive"),
    ],
)
def test_coordination_errors(options, message):
    with pytest.raises(InvalidParameter, match=message):
        CoordinationNumber(**options)


def test_coordination_needs_elements_for_pairs():
    system = b2()
    with pytest.raises(MissingProperty, match="element column"):
        CoordinationNumber(cutoffs={("Cu", "Zn"): 2.7}).compute(
            mdapy.System(system["positions"], system.box)
        )


def test_counting_during_the_search_equals_counting_a_list():
    system = alloy(seed=8)
    direct = RadialDistributionFunction(6.0, bins=60).compute(system)
    listed = RadialDistributionFunction(6.0, bins=60).compute(
        system, neighbors=mdapy.NeighborList.build(system, 7.0)
    )
    np.testing.assert_array_equal(direct.g, listed.g)
    for pair in direct.partial:
        np.testing.assert_array_equal(direct.partial[pair], listed.partial[pair])
    with mdapy.num_threads(3):
        np.testing.assert_array_equal(
            RadialDistributionFunction(6.0, bins=60).compute(system).g, direct.g
        )
