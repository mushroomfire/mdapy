# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

import numpy as np
import pytest

import mdapy
from mdapy.errors import InvalidParameter
from mdapy.identify import (
    AcklandJonesAnalysis,
    CentroSymmetryParameter,
    ChillPlus,
    CommonNeighborAnalysis,
    DiamondStructure,
)

from .references import (
    ackland_jones,
    adaptive_cna,
    centro_symmetry,
    chill_plus,
    diamond,
)
from lattices import crystal, icosahedron

CRYSTALS = [("fcc", 3.6), ("hcp", 2.5), ("bcc", 2.87)]
# a cutoff between the shells that conventional CNA uses for each crystal
FIXED = {"fcc": 3.6 * 0.854, "hcp": 2.5 * 1.207, "bcc": 2.87 * 1.207}


def rotated(system, seed=0):
    """The system rotated and moved as a whole."""
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    box = mdapy.Box(system.box.matrix @ q.T, origin=[1.0, -2.0, 3.0])
    return mdapy.System(
        system.positions @ q.T + [1.0, -2.0, 3.0], box, elements=system.table["element"]
    )


@pytest.mark.parametrize("kind, a", CRYSTALS)
def test_perfect_crystals(kind, a):
    system = crystal(kind, a, (4, 4, 4))
    for cna in (CommonNeighborAnalysis(), CommonNeighborAnalysis(cutoff=FIXED[kind])):
        result = cna.compute(system)
        assert result.counts()[kind] == len(system)
        assert result.structure.dtype == np.int8
        np.testing.assert_array_equal(
            cna.compute(rotated(system)).structure, result.structure
        )


def test_other_structures():
    assert (
        CommonNeighborAnalysis().compute(crystal("diamond", 5.43)).counts()["other"]
        == 216
    )
    cluster = CommonNeighborAnalysis().compute(icosahedron())
    assert cluster.structure[0] == cluster.names.index("ico")
    assert cluster.counts() == {"other": 12, "fcc": 0, "hcp": 0, "bcc": 0, "ico": 1}
    assert (
        CommonNeighborAnalysis(cutoff=3.0).compute(icosahedron()).counts()["ico"] == 1
    )


def test_a_vacancy_changes_its_twelve_neighbors():
    perfect = crystal("fcc", 3.6, (4, 4, 4))
    vacancy = perfect.select(np.arange(1, len(perfect)))
    assert CommonNeighborAnalysis().compute(vacancy).counts() == {
        "other": 12,
        "fcc": len(vacancy) - 12,
        "hcp": 0,
        "bcc": 0,
        "ico": 0,
    }


@pytest.mark.parametrize(
    "kind, a, jitter",
    [("fcc", 3.6, 0.15), ("fcc", 3.6, 0.25), ("hcp", 2.5, 0.12), ("bcc", 2.87, 0.15)],
)
def test_matches_a_python_reference(kind, a, jitter):
    system = crystal(kind, a, (3, 3, 3), jitter=jitter, seed=1)
    result = CommonNeighborAnalysis().compute(system)
    names = [result.names[t] for t in result.structure]
    assert names == adaptive_cna(system)
    assert 0 < result.counts()[kind]


def test_small_boxes_and_small_systems():
    primitive = mdapy.System(
        [[0.0, 0.0, 0.0]],
        mdapy.Box([[0, 1.8, 1.8], [1.8, 0, 1.8], [1.8, 1.8, 0]]),
        elements="Cu",
    )
    assert CommonNeighborAnalysis().compute(primitive).counts()["fcc"] == 1
    assert CommonNeighborAnalysis(cutoff=3.0).compute(primitive).counts()["fcc"] == 1
    empty = mdapy.System(np.zeros((0, 3)), mdapy.Box(5.0, pbc=False), elements=[])
    assert len(CommonNeighborAnalysis().compute(empty).structure) == 0
    assert len(CommonNeighborAnalysis(3.0).compute(empty).structure) == 0
    lonely = mdapy.System([[1.0, 1, 1]], mdapy.Box(5.0, pbc=False), elements="Cu")
    np.testing.assert_array_equal(
        CommonNeighborAnalysis().compute(lonely).structure, [0]
    )


def test_given_neighbors_and_threads():
    system = crystal("fcc", 3.6, (4, 4, 4), jitter=0.2, seed=3)
    expected = CommonNeighborAnalysis().compute(system).structure
    nl = mdapy.NeighborList.build(system, 5.0)
    np.testing.assert_array_equal(
        CommonNeighborAnalysis().compute(system, neighbors=nl).structure, expected
    )
    fixed = CommonNeighborAnalysis(cutoff=3.05)
    np.testing.assert_array_equal(
        fixed.compute(system, neighbors=nl).structure, fixed.compute(system).structure
    )
    for threads in (1, 3):
        with mdapy.num_threads(threads):
            np.testing.assert_array_equal(
                CommonNeighborAnalysis().compute(system).structure, expected
            )
    with pytest.raises(InvalidParameter, match="below the required"):
        fixed.compute(system, neighbors=mdapy.NeighborList.build(system, 2.0))
    with pytest.raises(InvalidParameter, match="other positions or another box"):
        fixed.compute(crystal("fcc", 3.6), neighbors=nl)


@pytest.mark.parametrize(
    "kind, a, cutoff", [("fcc", 3.6, 3.07), ("bcc", 2.87, 3.46), ("hcp", 2.5, 3.02)]
)
@pytest.mark.parametrize("pbc", [True, False])
def test_a_cutoff_on_the_grid_gives_what_a_list_gives(kind, a, cutoff, pbc):
    system = crystal(kind, a, (4, 4, 4), jitter=0.12, seed=4)
    if not pbc:
        system = system.with_box(mdapy.Box(system.box.matrix, pbc=False))
    fixed = CommonNeighborAnalysis(cutoff)
    nl = mdapy.NeighborList.build(system, cutoff)
    found = fixed.compute(system).structure
    np.testing.assert_array_equal(found, fixed.compute(system, neighbors=nl).structure)
    assert np.count_nonzero(found) > 0  # a fair test: structures are found


@pytest.mark.parametrize("kind, a", [("fcc", 3.6), ("bcc", 2.87), ("hcp", 2.5)])
@pytest.mark.parametrize("box", ["periodic", "triclinic", "cluster"])
def test_nearest_neighbors_found_as_they_go_give_what_a_list_gives(kind, a, box):
    system = crystal(kind, a, (3, 3, 3), jitter=0.15, seed=6)
    if box == "triclinic":
        tilted = system.box.matrix + [[0, 0, 0], [1.3, 0, 0], [0.4, -0.9, 0]]
        system = mdapy.System(system.positions, mdapy.Box(tilted), elements="Cu")
    elif box == "cluster":
        system = system.with_box(mdapy.Box(system.box.matrix, pbc=False))
    nl = mdapy.NeighborList.build(system, num_neighbors=14)
    for analysis in (
        CommonNeighborAnalysis(),
        AcklandJonesAnalysis(),
        CentroSymmetryParameter(12),
        CentroSymmetryParameter(8),
    ):
        found = analysis.compute(system)
        listed = analysis.compute(system, neighbors=nl)
        for name, value in vars(found).items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, vars(listed)[name])
    small = mdapy.System(system.positions[:10], mdapy.Box(20.0, pbc=False))
    with pytest.raises(InvalidParameter, match="has only 10 atoms"):
        CentroSymmetryParameter(12).compute(small)
    assert AcklandJonesAnalysis().compute(small).counts()["other"] == 10


def test_results_become_columns():
    system = crystal("bcc", 2.87, (2, 2, 2))
    result = CommonNeighborAnalysis().compute(system)
    table = system.with_columns(result).table
    np.testing.assert_array_equal(table["structure"].to_numpy(), result.structure)
    assert len(result.colors) == len(result.names) == 5
    assert not result.structure.flags.writeable
    assert (
        repr(CommonNeighborAnalysis(cutoff=3.0)) == "CommonNeighborAnalysis(cutoff=3.0)"
    )
    with pytest.raises(InvalidParameter, match="cutoff must be positive"):
        CommonNeighborAnalysis(cutoff=0.0)


@pytest.mark.parametrize(
    "kind, a, k", [("fcc", 3.6, 12), ("bcc", 2.87, 8), ("hcp", 2.5, 12)]
)
def test_centro_symmetry(kind, a, k):
    perfect = CentroSymmetryParameter(k).compute(crystal(kind, a, (3, 3, 3))).csp
    if kind == "hcp":
        # pairs of the six neighbors above and below the basal plane are a apart
        np.testing.assert_allclose(perfect, a * a, rtol=1e-12)
    else:
        np.testing.assert_allclose(perfect, 0.0, atol=1e-20)
    system = crystal(kind, a, (3, 3, 3), jitter=0.1, seed=2)
    csp = CentroSymmetryParameter(k).compute(system).csp
    np.testing.assert_allclose(csp, centro_symmetry(system, k), rtol=1e-12)
    np.testing.assert_allclose(
        CentroSymmetryParameter(k).compute(rotated(system)).csp, csp, rtol=1e-10
    )


def test_centro_symmetry_options():
    system = crystal("fcc", 3.6, (3, 3, 3), jitter=0.1, seed=4)
    csp = CentroSymmetryParameter().compute(system)
    nl = mdapy.NeighborList.build(system, 4.5)
    np.testing.assert_array_equal(
        CentroSymmetryParameter().compute(system, neighbors=nl).csp, csp.csp
    )
    with mdapy.num_threads(3):
        np.testing.assert_array_equal(
            CentroSymmetryParameter().compute(system).csp, csp.csp
        )
    np.testing.assert_array_equal(
        system.with_columns(csp).table["csp"].to_numpy(), csp.csp
    )
    assert (
        repr(CentroSymmetryParameter(8)) == "CentroSymmetryParameter(num_neighbors=8)"
    )
    with pytest.raises(InvalidParameter, match="fewer than 12"):
        CentroSymmetryParameter().compute(system, neighbors=nl.within(3.0))
    for bad in (0, 3):
        with pytest.raises(InvalidParameter, match="even and at least 2"):
            CentroSymmetryParameter(bad)


@pytest.mark.parametrize(
    "kind, a, name",
    [("diamond", 5.43, "cubic_diamond"), ("lonsdaleite", 2.51, "hexagonal_diamond")],
)
def test_diamond_structures(kind, a, name):
    system = crystal(kind, a, (3, 3, 3))
    result = DiamondStructure().compute(system)
    assert result.counts()[name] == len(system)
    np.testing.assert_array_equal(
        DiamondStructure().compute(rotated(system)).structure, result.structure
    )
    for other in (crystal("fcc", 3.6), crystal("bcc", 2.87)):
        assert DiamondStructure().compute(other).counts()["other"] == len(other)


@pytest.mark.parametrize("kind, a", [("diamond", 5.43), ("lonsdaleite", 2.51)])
def test_diamond_matches_a_python_reference(kind, a):
    system = crystal(kind, a, (3, 3, 3), jitter=0.22, seed=4)
    result = DiamondStructure().compute(system)
    assert [result.names[t] for t in result.structure] == diamond(system)
    assert 0 < result.counts()["other"] < len(system)  # a mixture


@pytest.mark.parametrize(
    "kind, a, repeat, seed",
    [("diamond", 5.43, (4, 4, 4), 2), ("lonsdaleite", 3.84, (5, 5, 4), 7)],
)
def test_diamond_needs_the_bond_back_from_every_neighbor(kind, a, repeat, seed):
    """An atom that is not among the four nearest of one of its neighbors is
    other, whatever its second neighbors look like; in these systems two such
    atoms have second neighbors of diamond."""
    system = crystal(kind, a, repeat, jitter=0.3, seed=seed)
    nl = mdapy.NeighborList.build(system, num_neighbors=4)
    first = nl.j.reshape(-1, 4)
    one_way = np.array(
        [any(i not in first[j] for j in first[i]) for i in range(len(system))]
    )
    assert one_way.sum() > 10
    result = DiamondStructure().compute(system)
    assert np.all(result.structure[one_way] == 0)
    assert result.counts()["other"] < len(system)  # some atoms are diamond
    assert [result.names[t] for t in result.structure] == diamond(system)


def test_diamond_options():
    system = crystal("diamond", 5.43, (2, 2, 2), jitter=0.1, seed=5)
    expected = DiamondStructure().compute(system).structure
    nl = mdapy.NeighborList.build(system, 3.0)
    np.testing.assert_array_equal(
        DiamondStructure().compute(system, neighbors=nl).structure, expected
    )
    with mdapy.num_threads(3):
        np.testing.assert_array_equal(
            DiamondStructure().compute(system).structure, expected
        )
    assert repr(DiamondStructure()) == "DiamondStructure()"
    with pytest.raises(InvalidParameter, match="fewer than 4"):
        DiamondStructure().compute(system, neighbors=nl.within(1.0))


@pytest.mark.parametrize(
    "kind, a, name",
    [("diamond", 6.36, "cubic_ice"), ("lonsdaleite", 4.5, "hexagonal_ice")],
)
def test_chill_plus_ice(kind, a, name):
    ice = crystal(kind, a, (3, 3, 3), element="O")
    assert ChillPlus().compute(ice).counts()[name] == len(ice)
    hot = crystal(kind, a, (3, 3, 3), element="O", jitter=0.25, seed=1)
    result = ChillPlus().compute(hot)
    assert [result.names[t] for t in result.structure] == chill_plus(hot, 3.5)
    assert 0 < result.counts()["other"] + result.counts()["interfacial_ice"] < len(hot)


def test_chill_plus_liquid_and_options():
    rng = np.random.default_rng(0)
    liquid = mdapy.System(rng.random((1000, 3)) * 31, mdapy.Box(31.0), elements="O")
    result = ChillPlus().compute(liquid)
    assert [result.names[t] for t in result.structure] == chill_plus(liquid, 3.5)
    counts = result.counts()
    assert (
        counts["other"] > 900
        and counts["hydrate"] > 0
        and counts["interfacial_hydrate"] > 0
    )
    nl = mdapy.NeighborList.build(liquid, 4.0)
    np.testing.assert_array_equal(
        ChillPlus().compute(liquid, neighbors=nl).structure, result.structure
    )
    with mdapy.num_threads(3):
        np.testing.assert_array_equal(
            ChillPlus().compute(liquid).structure, result.structure
        )
    assert repr(ChillPlus(3.2)) == "ChillPlus(cutoff=3.2)"
    with pytest.raises(InvalidParameter, match="cutoff must be positive"):
        ChillPlus(0.0)


@pytest.mark.parametrize("kind, a", CRYSTALS)
def test_ackland_jones_perfect_crystals(kind, a):
    result = AcklandJonesAnalysis().compute(crystal(kind, a, (3, 3, 3)))
    assert result.counts()[kind] == len(result.structure)
    center = AcklandJonesAnalysis().compute(icosahedron()).structure
    np.testing.assert_array_equal(center, [4] + [0] * 12)
    assert repr(AcklandJonesAnalysis()) == "AcklandJonesAnalysis()"


@pytest.mark.parametrize(
    "kind, a, jitter", [("fcc", 3.6, 0.2), ("hcp", 2.5, 0.15), ("bcc", 2.87, 0.15)]
)
def test_ackland_jones_matches_a_python_reference(kind, a, jitter):
    """The reference takes every neighbor within 1.55 r0^2; mdapy the 14
    nearest, which are all of them here."""
    system = crystal(kind, a, (3, 3, 3), jitter=jitter, seed=11)
    expected, largest = ackland_jones(system)
    assert largest <= 14
    analysis = AcklandJonesAnalysis()
    result = analysis.compute(system).structure
    np.testing.assert_array_equal(result, expected)
    assert len(np.unique(result)) > 2
    for nl in (
        mdapy.NeighborList.build(system, num_neighbors=14),
        mdapy.NeighborList.build(system, num_neighbors=16),
        mdapy.NeighborList.build(system, 5.0),
    ):
        np.testing.assert_array_equal(
            analysis.compute(system, neighbors=nl).structure, result
        )
    with mdapy.num_threads(3):
        np.testing.assert_array_equal(analysis.compute(system).structure, result)


def test_ackland_jones_small_systems_and_short_lists():
    analysis = AcklandJonesAnalysis()
    few = mdapy.System(np.eye(3) * 2.5, mdapy.Box(10.0, pbc=False), elements="Cu")
    np.testing.assert_array_equal(analysis.compute(few).structure, 0)
    empty = mdapy.System(np.zeros((0, 3)), mdapy.Box(5.0, pbc=False), elements=[])
    assert len(analysis.compute(empty).structure) == 0
    # a cluster: each atom takes the 14 nearest of its 31 others
    cluster = crystal("fcc", 3.6, (2, 2, 2)).with_box(mdapy.Box(30.0, pbc=False))
    expected, largest = ackland_jones(cluster)
    assert largest <= 14
    result = analysis.compute(cluster).structure
    np.testing.assert_array_equal(result, expected)
    assert 0 < (result == 1).sum() < len(cluster)  # inner atoms are fcc
    system = crystal("fcc", 3.6, (3, 3, 3))
    with pytest.raises(InvalidParameter, match="fewer than 14"):
        analysis.compute(system, neighbors=mdapy.NeighborList.build(system, 3.0))
