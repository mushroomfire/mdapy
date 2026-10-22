# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Analyses against OVITO 3.17.0, for every analysis OVITO has. Skipped
where OVITO is not installed."""

import numpy as np
import pytest

import mdapy
from mdapy.analysis import (
    AtomicStrain,
    BondAngleDistribution,
    ClusterAnalysis,
    CoordinationNumber,
    MeanSquaredDisplacement,
    RadialDistributionFunction,
    SteinhardtParameters,
    WignerSeitz,
)

from lattices import crystal

ovito = pytest.importorskip("ovito")
from ovito.io import import_file  # noqa: E402
from ovito.modifiers import (  # noqa: E402
    AtomicStrainModifier,
    BondAngleDistributionModifier,
    CalculateDisplacementsModifier,
    ClusterAnalysisModifier,
    CoordinationAnalysisModifier,
    CreateBondsModifier,
    SteinhardtParametersModifier,
    UnwrapTrajectoriesModifier,
    WignerSeitzAnalysisModifier,
)
from ovito.pipeline import ReferenceConfigurationModifier  # noqa: E402

from ovito_reference import ovito_compute, ovito_source  # noqa: E402


def placed(system, box, seed=3):
    """The system as it is, rotated with its box into a triclinic box, or
    without periodic directions."""
    if box == "rotated":
        q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
        q[:, 0] *= np.sign(np.linalg.det(q))
        rotated = mdapy.System(
            system.positions @ q.T + 1.5,
            mdapy.Box(system.box.matrix @ q.T, origin=[1.5, 1.5, 1.5]),
            elements=system.table["element"].to_list(),
        )
        if "mass" in system.table.columns:
            rotated = rotated.with_columns(mass=system.table["mass"].to_numpy())
        return rotated
    if box == "cluster":
        return system.with_box(mdapy.Box(system.box.matrix, pbc=False))
    return system


def alloy(box, seed=3):
    """A disordered fcc alloy of Cu and Ni."""
    system = crystal("fcc", 3.6, (5, 5, 5), jitter=0.15, seed=seed)
    elements = np.random.default_rng(seed).choice(["Cu", "Ni"], len(system))
    return placed(system.with_columns(element=list(elements)), box, seed)


def mapping(affine):
    affine_mapping = ReferenceConfigurationModifier.AffineMapping
    return affine_mapping.ToReference if affine else affine_mapping.Off


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


@pytest.mark.parametrize("box", ["periodic", "rotated", "cluster"])
def test_cluster_analysis(box):
    rng = np.random.default_rng(1)
    gas = mdapy.System(rng.random((400, 3)) * 20, mdapy.Box(20.0), elements="Ar")
    system = placed(gas, box)
    ours = ClusterAnalysis(cutoff=2.6).compute(system)
    modifier = ClusterAnalysisModifier(cutoff=2.6, sort_by_size=True)
    theirs = np.asarray(ovito_compute(system, modifier).particles["Cluster"])
    # the same clusters of the same sizes; OVITO orders equal sizes in no
    # particular order, mdapy by their first atom
    np.testing.assert_array_equal(ours.sizes, np.sort(np.bincount(theirs))[::-1])
    assert len(set(zip(ours.cluster, theirs))) == len(ours.sizes)
    unique = np.flatnonzero(np.bincount(ours.sizes)[ours.sizes] == 1)
    for cluster in unique:
        np.testing.assert_array_equal(theirs[ours.cluster == cluster], cluster)
    assert len(unique) < len(ours.sizes)  # a fair test: some sizes repeat


def defective(box, seed=1):
    """A perfect fcc reference and the same crystal sheared by 3 %, its atoms
    displaced, with 3 vacancies and 5 interstitials."""
    rng = np.random.default_rng(seed)
    reference = crystal("fcc", 3.6, (4, 4, 4), element="Cu")
    moved = reference.positions + rng.normal(0, 0.2, reference.positions.shape)
    moved = np.vstack(
        [np.delete(moved, [3, 50, 77], axis=0), rng.random((5, 3)) * 14.4]
    )
    shear = np.eye(3) + np.array([[0, 0.03, 0], [0, 0, 0], [0, 0, 0]])
    current = mdapy.System(
        moved @ shear.T, mdapy.Box(reference.box.matrix @ shear.T), elements="Cu"
    )
    return placed(reference, box), placed(current, box)


@pytest.mark.parametrize("box", ["periodic", "rotated", "cluster"])
@pytest.mark.parametrize("affine", [False, True])
def test_wigner_seitz(box, affine):
    reference, current = defective(box)
    ours = WignerSeitz(reference, affine=affine).compute(current)
    sites = WignerSeitzAnalysisModifier(
        output_displaced=True, affine_mapping=mapping(affine)
    )
    sites.reference = ovito_source(reference)
    np.testing.assert_array_equal(
        ours.site, ovito_compute(current, sites).particles["Site Index"]
    )
    occupancy = WignerSeitzAnalysisModifier(affine_mapping=mapping(affine))
    occupancy.reference = ovito_source(reference)
    output = ovito_compute(current, occupancy)
    np.testing.assert_array_equal(ours.occupancy, output.particles["Occupancy"])
    assert ours.vacancy_count == output.attributes["WignerSeitz.vacancy_count"] == 3
    assert (
        ours.interstitial_count == output.attributes["WignerSeitz.interstitial_count"]
    )


@pytest.mark.parametrize("box", ["periodic", "rotated", "cluster"])
@pytest.mark.parametrize("affine", [False, True])
def test_atomic_strain(box, affine):
    rng = np.random.default_rng(2)
    reference = crystal("fcc", 3.6, (5, 5, 5), element="Cu")
    deformation = np.eye(3) + np.array(
        [[0.02, 0.03, 0], [0, -0.01, 0.01], [0, 0, 0.01]]
    )
    moved = reference.positions + rng.normal(0, 0.05, reference.positions.shape)
    current = mdapy.System(
        moved @ deformation.T,
        mdapy.Box(reference.box.matrix @ deformation.T),
        elements="Cu",
    )
    reference, current = placed(reference, box), placed(current, box)
    ours = AtomicStrain(reference, 3.0, affine=affine).compute(current)
    modifier = AtomicStrainModifier(
        cutoff=3.0,
        output_strain_tensors=True,
        output_nonaffine_squared_displacements=True,
        affine_mapping=mapping(affine),
    )
    modifier.reference = ovito_source(reference)
    theirs = ovito_compute(current, modifier).particles
    e = ours.strain
    voigt = np.stack(
        [e[:, 0, 0], e[:, 1, 1], e[:, 2, 2], e[:, 0, 1], e[:, 0, 2], e[:, 1, 2]], axis=1
    )
    # sums over a dozen neighbor vectors and a 3 x 3 inverse: rounding only
    tolerance = {"rtol": 1e-10, "atol": 1e-12}
    np.testing.assert_allclose(voigt, theirs["Strain Tensor"], **tolerance)
    np.testing.assert_allclose(ours.shear, theirs["Shear Strain"], **tolerance)
    np.testing.assert_allclose(
        ours.volumetric, theirs["Volumetric Strain"], **tolerance
    )
    np.testing.assert_allclose(
        ours.d2min, theirs["Nonaffine Squared Displacement"], **tolerance
    )


def molecules(box, seed=4):
    """Thirty chains of 2 to 8 atoms 1.4 Å apart, scattered in a 25 Å box
    and wrapped into it, so that boundaries cut many of them; each atom has a
    mass of its own."""
    rng = np.random.default_rng(seed)
    chains = []
    for _ in range(30):
        steps = rng.normal(size=(rng.integers(2, 9), 3))
        steps *= 1.4 / np.linalg.norm(steps, axis=1)[:, None]
        steps[0] = rng.random(3) * 25.0
        chains.append(np.cumsum(steps, axis=0))
    positions = np.vstack(chains) % 25.0
    system = mdapy.System(positions, mdapy.Box(25.0), elements="Ar")
    system = system.with_columns(mass=rng.uniform(1.0, 60.0, len(positions)))
    return placed(system, box)


@pytest.mark.parametrize("box", ["periodic", "rotated", "cluster"])
def test_cluster_geometry(box):
    system = molecules(box)
    ours = ClusterAnalysis(cutoff=1.6).compute(system)
    output = ovito_compute(
        system,
        ClusterAnalysisModifier(
            cutoff=1.6,
            sort_by_size=True,
            compute_com=True,
            compute_gyration=True,
            unwrap_particles=True,
            output_table=True,
        ),
    )
    assert not ours.spanning.any()
    # the first atom of each cluster stays in place, the others follow it
    np.testing.assert_allclose(
        ours.unwrapped, output.particles["Position"], rtol=0, atol=1e-10
    )
    if box != "cluster":
        assert np.abs(ours.unwrapped - system.positions).max() > 10  # some were cut
    table = output.tables["clusters"]
    row = {int(c): r for r, c in enumerate(table["Cluster Identifier"])}
    theirs = np.asarray(output.particles["Cluster"])
    first = [np.flatnonzero(ours.cluster == c)[0] for c in range(len(ours.sizes))]
    rows = [row[theirs[i]] for i in first]
    np.testing.assert_allclose(
        ours.center_of_mass, np.asarray(table["Center of Mass"])[rows], atol=1e-10
    )
    np.testing.assert_allclose(
        ours.radius_of_gyration,
        np.asarray(table["Radius of Gyration"])[rows],
        rtol=1e-10,
        atol=1e-10,
    )
    g = ours.gyration_tensor
    voigt = np.stack(
        [g[:, 0, 0], g[:, 1, 1], g[:, 2, 2], g[:, 0, 1], g[:, 0, 2], g[:, 1, 2]], axis=1
    )
    np.testing.assert_allclose(
        voigt, np.asarray(table["Gyration Tensor"])[rows], rtol=1e-10, atol=1e-10
    )


@pytest.mark.parametrize("step, flags", [(4.0, True), (0.4, False)])
def test_displacement_from_the_first_frame(tmp_path, step, flags):
    """A random walk written as a dump, the atoms of each frame in another
    order: OVITO matches them by their ids, unwraps them by the image flags,
    or without them by following the jumps through the boundaries, and
    measures the displacements from frame 0."""
    box = mdapy.Box([[12.0, 0, 0], [2.0, 11.0, 0], [-1.0, 1.5, 13.0]])
    rng = np.random.default_rng(7)
    start = box.to_cartesian(rng.random((40, 3)))
    walk = start + np.cumsum(rng.normal(scale=step, size=(12, 40, 3)), axis=0)
    frames = []
    for t, positions in enumerate(walk):
        wrapped, image = box.wrap(positions)
        frame = mdapy.System(wrapped, box, elements="Ar").with_columns(
            id=np.arange(1, 41)
        )
        if flags:
            frame = frame.with_columns(image=image)
        # each frame in another order of the atoms, matched by their ids
        frame = frame.with_table(frame.table[rng.permutation(40)])
        frames.append(frame.with_info(timestep=t))
    mdapy.write(tmp_path / "walk.dump", frames)
    pipeline = import_file(str(tmp_path / "walk.dump"))
    pipeline.modifiers.append(UnwrapTrajectoriesModifier())
    pipeline.modifiers.append(
        CalculateDisplacementsModifier(minimum_image_convention=False)
    )
    expected = np.array(
        [
            (pipeline.compute(t).particles["Displacement"][...] ** 2).mean(axis=0)
            for t in range(len(frames))
        ]
    )
    result = MeanSquaredDisplacement(window=False).compute(frames)
    np.testing.assert_allclose(result.components, expected, rtol=1e-12, atol=1e-12)
    assert box.wrap(walk[-1])[1].any()  # atoms crossed the boundaries
