# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Distributions and order parameters against OVITO 3.17.0, for every
analysis OVITO has. Skipped where OVITO is not installed."""

import numpy as np
import pytest

import mdapy
from mdapy.analysis import (
    BondAngleDistribution,
    CoordinationNumber,
    RadialDistributionFunction,
    SteinhardtParameters,
)

from lattices import crystal

ovito = pytest.importorskip("ovito")
from ovito.modifiers import (  # noqa: E402
    BondAngleDistributionModifier,
    CoordinationAnalysisModifier,
    CreateBondsModifier,
    SteinhardtParametersModifier,
)

from ovito_reference import ovito_compute  # noqa: E402


def alloy(box, seed=3):
    """A disordered fcc alloy of Cu and Ni, in an orthogonal periodic box, a
    rotated triclinic box or no box at all."""
    system = crystal("fcc", 3.6, (5, 5, 5), jitter=0.15, seed=seed)
    elements = np.random.default_rng(seed).choice(["Cu", "Ni"], len(system))
    system = system.with_columns(element=list(elements))
    if box == "rotated":
        q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
        q[:, 0] *= np.sign(np.linalg.det(q))
        return mdapy.System(
            system.positions @ q.T + 1.5,
            mdapy.Box(system.box.matrix @ q.T, origin=[1.5, 1.5, 1.5]),
            elements=list(elements),
        )
    if box == "cluster":
        return system.with_box(mdapy.Box(system.box.matrix, pbc=False))
    return system


@pytest.mark.parametrize("box", ["periodic", "rotated"])
def test_radial_distribution_function(box):
    system = alloy(box)
    for partial in (False, True):
        table = ovito_compute(
            system,
            CoordinationAnalysisModifier(
                cutoff=7.5, number_of_bins=75, partial=partial
            ),
        ).tables["coordination-rdf"]
        xy = table.xy()
        ours = RadialDistributionFunction(7.5, bins=75).compute(system)
        np.testing.assert_allclose(ours.r, xy[:, 0], rtol=1e-12)
        if not partial:
            # counts of pairs over volumes: rounding only
            np.testing.assert_allclose(ours.g, xy[:, 1], rtol=1e-12, atol=1e-12)
            continue
        for k, name in enumerate(table.y.component_names):
            a, b = name.split("-")
            np.testing.assert_allclose(
                ours.pair(a, b), xy[:, 1 + k], rtol=1e-12, atol=1e-12
            )


@pytest.mark.parametrize("box", ["periodic", "rotated", "cluster"])
def test_coordination_numbers(box):
    system = alloy(box)
    particles = ovito_compute(
        system, CoordinationAnalysisModifier(cutoff=3.0, partial=True)
    ).particles
    ours = CoordinationNumber(3.0).compute(system)
    np.testing.assert_array_equal(ours.coordination, particles["Coordination"])
    per_type = np.asarray(particles["Per Type Coordination"])
    for k, element in enumerate(("Ni", "Cu")):  # OVITO's types, by atomic number
        np.testing.assert_array_equal(ours.partial[element], per_type[:, k])


@pytest.mark.parametrize("box", ["periodic", "rotated", "cluster"])
def test_bond_angle_distribution(box):
    system = alloy(box)
    partition = BondAngleDistributionModifier.Partition
    table = ovito_compute(
        system,
        CreateBondsModifier(cutoff=3.0),
        BondAngleDistributionModifier(bins=180, partition=partition.ByParticleType),
    ).tables["bond-angle-distr"]
    xy = table.xy()
    ours = BondAngleDistribution(cutoff=3.0, bins=180).compute(system)
    np.testing.assert_allclose(ours.angle, xy[:, 0], rtol=1e-12)
    np.testing.assert_array_equal(ours.counts, xy[:, 1:].sum(axis=1))
    for k, name in enumerate(table.y.component_names):
        b, vertex, c = name.split("-")  # OVITO names the vertex in the middle
        np.testing.assert_array_equal(ours.triplet(vertex, b, c), xy[:, 1 + k])


@pytest.mark.parametrize("box", ["periodic", "rotated", "cluster"])
@pytest.mark.parametrize("average", [False, True])
@pytest.mark.parametrize("neighbors", ["nearest", "cutoff"])
def test_steinhardt_parameters(box, average, neighbors):
    system = alloy(box)
    mode = SteinhardtParametersModifier.NeighborMode
    if neighbors == "nearest":
        modifier = SteinhardtParametersModifier(
            neighbor_mode=mode.NearestNeighbors, num_neighbors=12
        )
        options = {"num_neighbors": 12}
    else:
        modifier = SteinhardtParametersModifier(neighbor_mode=mode.Cutoff, cutoff=3.0)
        options = {"cutoff": 3.0}
    modifier.max_l = 10
    modifier.compute_averaged = average
    particles = ovito_compute(system, modifier).particles
    theirs = np.asarray(particles["Smooth Steinhardt" if average else "Steinhardt"])
    ours = SteinhardtParameters(tuple(range(2, 11)), average=average, **options)
    result = ours.compute(system)
    for degree in range(2, 11):
        # sums of a dozen spherical harmonics: rounding only
        np.testing.assert_allclose(
            result[degree], theirs[:, degree - 2], rtol=1e-12, atol=1e-13
        )
