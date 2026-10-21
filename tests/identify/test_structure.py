# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

import numpy as np
import pytest
from lattices import crystal, icosahedron

import mdapy
from mdapy.errors import InvalidParameter
from mdapy.identify import CentroSymmetryParameter, CommonNeighborAnalysis

from .references import adaptive_cna, centro_symmetry

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
    with pytest.raises(InvalidParameter, match="another system"):
        fixed.compute(crystal("fcc", 3.6), neighbors=nl)


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
