# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Structure identification against OVITO 3.17.0, atom by atom, for every
method OVITO has. Skipped where OVITO is not installed."""

import numpy as np
import pytest

import mdapy
from mdapy.identify import (
    AcklandJonesAnalysis,
    CentroSymmetryParameter,
    ChillPlus,
    CommonNeighborAnalysis,
    DiamondStructure,
)

from lattices import crystal

ovito = pytest.importorskip("ovito")
from ovito.modifiers import (  # noqa: E402
    AcklandJonesModifier,
    CentroSymmetryModifier,
    ChillPlusModifier,
    CommonNeighborAnalysisModifier,
    IdentifyDiamondModifier,
)

from ovito_reference import ovito_compute  # noqa: E402

LATTICES = {"fcc": 3.6, "hcp": 2.5, "bcc": 2.87}
# cutoffs between the first and second shells, for conventional CNA
FIXED = {"fcc": 0.854, "hcp": 1.207, "bcc": 1.207}


def ovito_particles(system, modifier):
    """The particles of system after modifier, as OVITO computes them."""
    return ovito_compute(system, modifier).particles


def disordered(kind, a, box, jitter, seed):
    """A crystal with random displacements, in an orthogonal periodic box, a
    rotated triclinic box or no box at all."""
    system = crystal(kind, a, (5, 5, 5), jitter=jitter, seed=seed)
    if box == "rotated":
        q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
        q[:, 0] *= np.sign(np.linalg.det(q))
        return mdapy.System(
            system["positions"] @ q.T + 1.5,
            mdapy.Box(system.box.matrix @ q.T, origin=[1.5, 1.5, 1.5]),
            elements=system.table["element"].to_list(),
        )
    if box == "cluster":
        return system.with_box(mdapy.Box(system.box.matrix, pbc=False))
    return system


def check_types(ours, theirs):
    np.testing.assert_array_equal(ours, theirs)
    # a fair test: some atoms keep their structure and some lose it
    assert 0 < np.count_nonzero(ours) < len(ours)


@pytest.mark.parametrize("box", ["periodic", "rotated", "cluster"])
@pytest.mark.parametrize("kind", LATTICES)
@pytest.mark.parametrize("jitter", [0.16, 0.22])
def test_common_neighbor_analysis(kind, box, jitter):
    system = disordered(kind, LATTICES[kind], box, jitter, seed=1)
    mode = CommonNeighborAnalysisModifier.Mode
    adaptive = ovito_particles(
        system, CommonNeighborAnalysisModifier(mode=mode.AdaptiveCutoff)
    )
    check_types(
        CommonNeighborAnalysis().compute(system).structure,
        np.asarray(adaptive["Structure Type"]),
    )
    cutoff = FIXED[kind] * LATTICES[kind]
    fixed = ovito_particles(
        system, CommonNeighborAnalysisModifier(mode=mode.FixedCutoff, cutoff=cutoff)
    )
    check_types(
        CommonNeighborAnalysis(cutoff).compute(system).structure,
        np.asarray(fixed["Structure Type"]),
    )


@pytest.mark.parametrize("box", ["periodic", "rotated", "cluster"])
@pytest.mark.parametrize("kind", LATTICES)
def test_ackland_jones(kind, box):
    system = disordered(kind, LATTICES[kind], box, 0.15, seed=2)
    theirs = ovito_particles(system, AcklandJonesModifier())
    check_types(
        AcklandJonesAnalysis().compute(system).structure,
        np.asarray(theirs["Structure Type"]),
    )


@pytest.mark.parametrize("box", ["periodic", "rotated", "cluster"])
@pytest.mark.parametrize("kind", LATTICES)
def test_centro_symmetry(kind, box):
    system = disordered(kind, LATTICES[kind], box, 0.15, seed=3)
    k = 8 if kind == "bcc" else 12
    theirs = ovito_particles(system, CentroSymmetryModifier(num_neighbors=k))
    # sums of a dozen squared lengths of a few Å: rounding only
    np.testing.assert_allclose(
        CentroSymmetryParameter(k).compute(system).csp,
        np.asarray(theirs["Centrosymmetry"]),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("box", ["periodic", "rotated", "cluster"])
@pytest.mark.parametrize("kind, a", [("diamond", 5.43), ("lonsdaleite", 3.84)])
def test_diamond_structure(kind, a, box):
    system = disordered(kind, a, box, 0.25, seed=4)
    theirs = np.asarray(
        ovito_particles(system, IdentifyDiamondModifier())["Structure Type"]
    )
    # OVITO also marks the first and second neighbors of diamond atoms, which
    # are other here
    as_ours = np.array([0, 1, 0, 0, 2, 0, 0])[theirs]
    check_types(DiamondStructure().compute(system).structure, as_ours)


@pytest.mark.parametrize(
    "kind, a", [("diamond", 6.36), ("lonsdaleite", 4.5), ("liquid", 31.0)]
)
def test_chill_plus(kind, a):
    if kind == "liquid":
        rng = np.random.default_rng(0)
        system = mdapy.System(rng.random((1000, 3)) * a, mdapy.Box(a), elements="O")
    else:
        system = crystal(kind, a, (3, 3, 3), element="O", jitter=0.25, seed=1)
    theirs = ovito_particles(system, ChillPlusModifier(cutoff=3.5))
    check_types(
        ChillPlus().compute(system).structure, np.asarray(theirs["Structure Type"])
    )


@pytest.mark.parametrize("repeat", [(1, 1, 1), (2, 1, 1), (2, 2, 2)])
@pytest.mark.parametrize("kind", LATTICES)
def test_boxes_smaller_than_the_neighbor_shells(kind, repeat):
    """Atoms see their own periodic images among their neighbors."""
    system = crystal(kind, LATTICES[kind], repeat, jitter=0.1, seed=5)
    for ours, modifier in (
        (CommonNeighborAnalysis(), CommonNeighborAnalysisModifier()),
        (AcklandJonesAnalysis(), AcklandJonesModifier()),
    ):
        np.testing.assert_array_equal(
            ours.compute(system).structure,
            np.asarray(ovito_particles(system, modifier)["Structure Type"]),
        )
    k = 8 if kind == "bcc" else 12
    np.testing.assert_allclose(
        CentroSymmetryParameter(k).compute(system).csp,
        np.asarray(
            ovito_particles(system, CentroSymmetryModifier(num_neighbors=k))[
                "Centrosymmetry"
            ]
        ),
        rtol=1e-12,
        atol=1e-12,
    )
