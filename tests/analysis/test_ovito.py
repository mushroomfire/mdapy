# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Analyses against OVITO 3.17.1, for every analysis OVITO has. Skipped
where OVITO is not installed."""

import numpy as np
import pytest

import mdapy
from mdapy import transform
from mdapy.analysis import (
    AtomicDisplacement,
    AtomicStrain,
    BondAngleDistribution,
    ChemicalSpecies,
    ClusterAnalysis,
    CoordinationNumber,
    MeanSquaredDisplacement,
    RadialDistributionFunction,
    SpatialBinning,
    SteinhardtParameters,
    StructureFactor,
    SurfaceMesh,
    TimeAverage,
    VoronoiAnalysis,
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
    ConstructSurfaceModifier,
    CoordinationAnalysisModifier,
    CreateBondsModifier,
    DiffusionModifier,
    ExpressionSelectionModifier,
    GenerateTrajectoryLinesModifier,
    SliceModifier,
    SpatialBinningModifier,
    SteinhardtParametersModifier,
    StructureFactorModifier,
    TimeAveragingModifier,
    UnwrapTrajectoriesModifier,
    VoronoiAnalysisModifier,
    WignerSeitzAnalysisModifier,
)
from ovito.pipeline import Pipeline, ReferenceConfigurationModifier  # noqa: E402

from mdapy.elements import vdw_radii  # noqa: E402

from ovito_reference import ovito_compute, ovito_source  # noqa: E402


def placed(system, box, seed=3):
    """The system as it is, rotated with its box into a triclinic box, or
    without periodic directions."""
    if box == "rotated":
        q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
        q[:, 0] *= np.sign(np.linalg.det(q))
        rotated = mdapy.System(
            system["positions"] @ q.T + 1.5,
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


@pytest.mark.parametrize("box", ["periodic", "rotated", "cluster"])
@pytest.mark.parametrize("hydrogen", [False, True])
def test_chemical_species_by_van_der_waals_radii(box, hydrogen):
    # a random mixture: molecules of every size, some only joined by two
    # hydrogen atoms
    rng = np.random.default_rng(1)
    elements = rng.choice(["H", "C", "N", "O"], 300).tolist()
    mixture = mdapy.System(
        rng.random((300, 3)) * 16, mdapy.Box(16.0), elements=elements
    )
    system = placed(mixture, box)
    cutoffs = {("H", "H"): 0.6 * 2 * 1.2} if hydrogen else None
    ours = ChemicalSpecies("vdw", cutoffs=cutoffs).compute(system)
    # OVITO with the same radii
    source = ovito_source(system)
    types = source.data.particles_.particle_types_
    for t in types.types:
        types.make_mutable(t).vdw_radius = float(vdw_radii([t.name])[0])
    pipeline = Pipeline(source=source)
    pipeline.modifiers.append(
        CreateBondsModifier(
            mode=CreateBondsModifier.Mode.VdWRadius, prevent_hh_bonds=not hydrogen
        )
    )
    pipeline.modifiers.append(
        ClusterAnalysisModifier(
            neighbor_mode=ClusterAnalysisModifier.NeighborMode.Bonding,
            sort_by_size=True,
        )
    )
    theirs = np.asarray(pipeline.compute().particles["Cluster"])
    np.testing.assert_array_equal(ours.sizes, np.sort(np.bincount(theirs))[::-1])
    assert len(set(zip(ours.molecule, theirs))) == len(ours.sizes) > 50
    # bonds of two hydrogen atoms join some molecules
    other = ChemicalSpecies("vdw", cutoffs=None if hydrogen else {("H", "H"): 1.44})
    assert len(other.compute(system).sizes) != len(ours.sizes)


def defective(box, seed=1):
    """A perfect fcc reference and the same crystal sheared by 3 %, its atoms
    displaced, with 3 vacancies and 5 interstitials."""
    rng = np.random.default_rng(seed)
    reference = crystal("fcc", 3.6, (4, 4, 4), element="Cu")
    moved = reference["positions"] + rng.normal(0, 0.2, reference["positions"].shape)
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
    moved = reference["positions"] + rng.normal(0, 0.05, reference["positions"].shape)
    current = mdapy.System(
        moved @ deformation.T,
        mdapy.Box(reference.box.matrix @ deformation.T),
        elements="Cu",
    )
    reference, current = placed(reference, box), placed(current, box)
    ours = AtomicStrain(
        reference, 3.0, affine=affine, strain_tensor=True, d2min=True
    ).compute(current)
    modifier = AtomicStrainModifier(
        cutoff=3.0,
        output_strain_tensors=True,
        output_nonaffine_squared_displacements=True,
        affine_mapping=mapping(affine),
    )
    modifier.reference = ovito_source(reference)
    theirs = ovito_compute(current, modifier).particles
    e = ours.strain_tensor
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
    ours = ClusterAnalysis(
        cutoff=1.6, unwrap=True, center_of_mass=True, gyration=True
    ).compute(system)
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
        assert np.abs(ours.unwrapped - system["positions"]).max() > 10  # some were cut
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


def random_walk(path, step, flags, seed=7):
    """A random walk of 40 atoms in 12 frames written as a dump, wrapped
    into a triclinic box, the atoms of each frame in another order."""
    box = mdapy.Box([[12.0, 0, 0], [2.0, 11.0, 0], [-1.0, 1.5, 13.0]])
    rng = np.random.default_rng(seed)
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
    mdapy.write(path, frames)
    return frames, walk


@pytest.mark.parametrize("step, flags", [(4.0, True), (0.4, False)])
def test_displacement_from_the_first_frame(tmp_path, step, flags):
    """A random walk written as a dump, the atoms of each frame in another
    order: OVITO matches them by their ids, unwraps them by the image flags,
    or without them by following the jumps through the boundaries, and
    measures the displacements from frame 0."""
    frames, walk = random_walk(tmp_path / "walk.dump", step, flags)
    box = frames[0].box
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


@pytest.mark.parametrize("window", [True, False])
@pytest.mark.parametrize("step, flags", [(4.0, True), (0.4, False)])
def test_mean_squared_displacement(tmp_path, window, step, flags):
    """The MSD of OVITO's diffusion modifier, from every time origin or from
    frame 0, along each axis and in total, after OVITO unwraps the
    trajectory by the image flags or by following the jumps."""
    frames, _ = random_walk(tmp_path / "walk.dump", step, flags)
    pipeline = import_file(str(tmp_path / "walk.dump"))
    pipeline.modifiers.append(UnwrapTrajectoriesModifier())
    origins = DiffusionModifier.ReferenceMode
    pipeline.modifiers.append(
        DiffusionModifier(
            time_source=DiffusionModifier.TimeSource.AnimationFrames,
            reference_mode=origins.MultipleOrigins if window else origins.FixedOrigin,
            partition_by_direction=True,
        )
    )
    table = pipeline.compute(len(frames) - 1).tables["msd"]
    names = list(table.y.component_names)
    values = np.asarray(table.y)
    lags = np.asarray(table.x).astype(int)
    ours = MeanSquaredDisplacement(window=window).compute(frames)
    np.testing.assert_array_equal(lags, ours.lag[1:])
    np.testing.assert_allclose(
        ours.msd[1:], values[:, names.index("Total")], rtol=1e-10, atol=1e-12
    )
    for k, axis in enumerate(("XX", "YY", "ZZ")):
        np.testing.assert_allclose(
            ours.components[1:, k], values[:, names.index(axis)], rtol=1e-10, atol=1e-12
        )


@pytest.mark.parametrize("step, flags", [(4.0, True), (0.4, False)])
def test_time_average(tmp_path, step, flags):
    """OVITO's time averages of the unwrapped positions, matched by id, of
    the radial distribution function and of the timestep."""
    frames, walk = random_walk(tmp_path / "walk.dump", step, flags)
    pipeline = import_file(str(tmp_path / "walk.dump"), sort_particles=True)
    pipeline.modifiers.append(UnwrapTrajectoriesModifier())
    pipeline.modifiers.append(
        CoordinationAnalysisModifier(cutoff=5.0, number_of_bins=50)
    )
    pipeline.modifiers.append(
        TimeAveragingModifier(
            operate_on=[
                "property:particles/Position",
                "table:coordination-rdf",
                "attribute:Timestep",
            ]
        )
    )
    data = pipeline.compute(0)
    positions = TimeAverage("positions").compute(frames)
    first = np.asarray(frames[0]["id"])
    np.testing.assert_allclose(
        positions.value[np.argsort(first)],
        np.asarray(data.particles["Position Average"]),
        rtol=0,
        atol=1e-9,
    )
    rdf = TimeAverage(
        lambda frame: RadialDistributionFunction(5.0, bins=50).compute(frame).g
    ).compute(frames)
    table = data.tables["coordination-rdf[average]"]
    np.testing.assert_allclose(rdf.value, np.asarray(table.y), rtol=1e-10, atol=1e-12)
    timestep = TimeAverage(lambda frame: frame.info["timestep"]).compute(frames)
    assert timestep.value == data.attributes["Timestep (Average)"] == 5.5


@pytest.mark.parametrize("every", [1, 2])
def test_trajectory_lines(tmp_path, every):
    """OVITO's lines of the paths of the atoms, every frame or every other
    one. OVITO follows the atoms by the nearest image only, without the
    image flags; mdapy uses the flags, which give the true path of atoms
    that jump far."""
    from mdapy.analysis import trajectory_lines

    frames, _ = random_walk(tmp_path / "walk.dump", 0.4, False)
    pipeline = import_file(str(tmp_path / "walk.dump"), sort_particles=True)
    pipeline.modifiers.append(
        GenerateTrajectoryLinesModifier(only_selected=False, sampling_frequency=every)
    )
    lines = next(o for o in pipeline.compute(0).objects if type(o).__name__ == "Lines")
    sections = np.asarray(lines["Section"])
    points = np.asarray(lines["Position"])
    ours = trajectory_lines(frames[::every])
    ids = np.asarray(frames[0]["id"])
    order = np.argsort(ids)  # a section of OVITO for each id
    theirs = np.stack([points[sections == i] for i in ids[order]])
    np.testing.assert_allclose(ours[order], theirs, rtol=0, atol=1e-9)
    # long jumps, followed by the image flags
    frames, walk = random_walk(tmp_path / "far.dump", 4.0, True)
    true = walk.transpose(1, 0, 2)[np.asarray(frames[0]["id"]) - 1]
    ours = trajectory_lines(frames)
    np.testing.assert_allclose(ours - ours[:, :1], true - true[:, :1], atol=1e-9)


def scattered(pbc, seed=0):
    """Random atoms strictly inside a box, triclinic or not, as OVITO needs
    in the directions that are not periodic."""
    if pbc == "orthogonal":
        box = mdapy.Box([12.0, 11.0, 13.0], pbc=(True, False, True))
    else:
        box = mdapy.Box(
            [[12.0, 0, 0], [2.0, 11.0, 0], [-1.5, 1.0, 13.0]],
            origin=[1, -2, 0.5],
            pbc=pbc,
        )
    rng = np.random.default_rng(seed)
    fractional = rng.random((300, 3)) * 0.98 + 0.01
    return mdapy.System(box.to_cartesian(fractional), box, elements="Ar")


@pytest.mark.parametrize(
    "system",
    [
        pytest.param(scattered(True), id="periodic"),
        pytest.param(scattered((True, True, False)), id="slab"),
        pytest.param(scattered("orthogonal"), id="orthogonal-slab"),
        pytest.param(crystal("fcc", 3.6, jitter=0.03, seed=6), id="fcc"),
    ],
)
@pytest.mark.parametrize(
    "thresholds",
    [
        {},
        {"face_threshold": 0.5},
        {"relative_face_threshold": 0.02},
        {"edge_threshold": 0.3},
        {"face_threshold": 0.1, "relative_face_threshold": 0.01, "edge_threshold": 0.2},
    ],
    ids=["none", "face", "relative", "edge", "all"],
)
def test_voronoi_cells(system, thresholds):
    """OVITO counts the walls of the box among the faces of a cell, in the
    coordination and the Voronoi index; mdapy counts faces shared with atoms."""
    particles = ovito_compute(
        system, VoronoiAnalysisModifier(compute_indices=True, **thresholds)
    ).particles
    result = VoronoiAnalysis(faces=True, **thresholds).compute(system)
    np.testing.assert_allclose(result.volume, particles["Atomic Volume"], atol=1e-12)
    np.testing.assert_allclose(result.radius, particles["Cavity Radius"], atol=1e-12)
    n = len(system)
    atom = np.repeat(np.arange(n), np.diff(result.offsets))
    walls = np.bincount(atom[result.j < 0], minlength=n)
    np.testing.assert_array_equal(
        result.coordination + walls, particles["Coordination"]
    )
    # OVITO's index counts the faces with 1, 2, ... edges, walls included
    expected = np.asarray(particles["Voronoi Index"])
    index = np.zeros_like(expected)
    np.add.at(index, (atom, result.edges - 1), 1)
    np.testing.assert_array_equal(index, expected)
    # index() counts the faces shared with atoms only
    shared = np.zeros_like(expected)
    atoms = result.j >= 0
    np.add.at(shared, (atom[atoms], result.edges[atoms] - 1), 1)
    np.testing.assert_array_equal(result.index(expected.shape[1]), shared[:, 2:])


def bond(i, j, shift):
    """A pair as an unordered bond: the same for (i, j, s) and (j, i, -s)."""
    shift = tuple(int(t) for t in shift)
    back = tuple(-t for t in shift)
    return min((int(i), int(j), shift), (int(j), int(i), back))


@pytest.mark.parametrize(
    "system",
    [
        pytest.param(scattered(True), id="periodic"),
        pytest.param(scattered((True, True, False)), id="slab"),
        pytest.param(crystal("fcc", 3.6, jitter=0.03, seed=6), id="fcc"),
    ],
)
@pytest.mark.parametrize(
    "thresholds",
    [{}, {"face_threshold": 0.5}, {"edge_threshold": 0.3}],
    ids=["none", "face", "edge"],
)
def test_voronoi_neighbors_as_bonds(system, thresholds):
    """OVITO makes a bond for each face that counts; the thresholds here
    treat both atoms of a face alike."""
    bonds = ovito_compute(
        system, VoronoiAnalysisModifier(generate_bonds=True, **thresholds)
    ).particles.bonds
    expected = {
        bond(a, b, shift)
        for (a, b), shift in zip(bonds.topology[...], bonds.pbc_vectors[...])
    }
    nl = mdapy.NeighborList.build(system, voronoi=True, **thresholds)
    pairs = [bond(i, j, shift) for i, j, shift in zip(nl.i, nl.j, nl.shifts)]
    assert set(pairs) == expected
    assert len(pairs) == 2 * len(expected)  # each bond from both of its atoms


def with_voids(box, seed=6):
    """fcc copper, moved at random, with two voids, one across a corner of
    the box."""
    system = crystal("fcc", 3.615, (9, 9, 9), jitter=0.05, seed=seed)
    keep = np.ones(len(system), dtype=bool)
    for center, radius in [((0.0, 0.0, 0.0), 6.0), ((0.5, 0.55, 0.5), 7.0)]:
        d = system.box.minimum_image(
            system["positions"] - np.array(center) @ system.box.matrix
        )
        keep &= np.linalg.norm(d, axis=1) > radius
    return placed(system.filter(keep), box)


@pytest.mark.parametrize("box", ["periodic", "rotated", "cluster"])
@pytest.mark.parametrize("smoothing", [0, 8])
def test_surface_mesh(box, smoothing):
    system = with_voids(box)
    ours = SurfaceMesh(smoothing=smoothing, regions=True, surface=True).compute(system)
    output = ovito_compute(
        system,
        ConstructSurfaceModifier(
            method=ConstructSurfaceModifier.Method.AlphaShape,
            radius=4.0,
            smoothing_level=smoothing,
            identify_regions=True,
            select_surface_particles=True,
        ),
    )
    attributes = output.attributes
    assert ours.area == pytest.approx(
        attributes["ConstructSurfaceMesh.surface_area"], rel=1e-9
    )
    assert ours.solid_volume == pytest.approx(
        attributes["ConstructSurfaceMesh.filled_volume"], rel=1e-9
    )
    np.testing.assert_array_equal(ours.surface, output.particles["Selection"] != 0)
    # the voids: the empty regions OVITO does not call exterior
    regions = output.surfaces["surface"].regions
    voids = (np.asarray(regions["Filled"]) == 0) & (
        np.asarray(regions["Exterior"]) == 0
    )
    assert attributes["ConstructSurfaceMesh.void_region_count"] == len(
        ours.regions.voids
    )
    for name, ours_values in [
        ("Volume", ours.regions.volume),
        ("Surface Area", ours.regions.area),
    ]:
        np.testing.assert_allclose(
            np.sort(ours_values[ours.regions.voids]),
            np.sort(np.asarray(regions[name])[voids]),
            rtol=1e-7,  # sums of thousands of tetrahedra in another order
        )


def tilted(n=300, seed=8, pbc=True):
    """Atoms at random in a triclinic box, some outside it, with ids."""
    rng = np.random.default_rng(seed)
    box = mdapy.Box(
        [[10.0, 0, 0], [2.0, 9.0, 0], [1.0, 1.0, 8.0]], origin=[1, 2, 3], pbc=pbc
    )
    positions = box.to_cartesian(rng.uniform(-0.2, 1.2, (n, 3)))
    system = mdapy.System(positions, box, elements="Cu")
    return system.with_columns(id=np.arange(1, n + 1), charge=rng.normal(size=n))


@pytest.mark.parametrize("width", [0.0, 3.0])
@pytest.mark.parametrize("invert", [False, True])
@pytest.mark.parametrize(
    "normal, distance, miller",
    [((1.0, 2.0, 0.5), 6.0, False), ((1, -2, 1), 0.3, True), ((2, 1, 0), -0.4, True)],
)
def test_slice(width, invert, normal, distance, miller):
    system = tilted()
    ours = transform.slice(
        system, normal, distance, width=width, invert=invert, miller=miller
    )
    theirs = ovito_compute(
        system,
        SliceModifier(
            normal=normal,
            distance=distance,
            slab_width=width,
            inverse=invert,
            miller=miller,
        ),
    ).particles
    np.testing.assert_array_equal(
        np.sort(ours["id"]), np.sort(theirs["Particle Identifier"])
    )


@pytest.mark.parametrize("invert", [False, True])
def test_slice_of_selected_atoms_and_the_mask(invert):
    """OVITO's only_selected is atoms=, and its select mode the atoms
    slice_mask leaves out."""
    system = tilted()
    chosen = system["id"] % 3 == 0
    ours = transform.slice(
        system, (1, 2, 0.5), 6.0, width=2.0, atoms=chosen, invert=invert
    )
    theirs = ovito_compute(
        system,
        ExpressionSelectionModifier(expression="fmod(ParticleIdentifier,3)==0"),
        SliceModifier(
            normal=(1, 2, 0.5),
            distance=6.0,
            slab_width=2.0,
            only_selected=True,
            inverse=invert,
        ),
    ).particles
    np.testing.assert_array_equal(
        np.sort(ours["id"]), np.sort(theirs["Particle Identifier"])
    )
    kept = transform.slice_mask(system, (1, 2, 0.5), 6.0, invert=invert)
    selected = ovito_compute(
        system,
        SliceModifier(normal=(1, 2, 0.5), distance=6.0, inverse=invert, select=True),
    ).particles["Selection"]
    np.testing.assert_array_equal(~kept, np.asarray(selected).astype(bool))


@pytest.mark.parametrize("pbc", [True, (True, True, False)])
@pytest.mark.parametrize("reduction", ["mean", "sum", "density", "min", "max"])
@pytest.mark.parametrize(
    "direction, axes, bins",
    [("Z", (2,), (6,)), ("XY", (0, 1), (3, 4)), ("XYZ", (0, 1, 2), (2, 3, 2))],
)
def test_spatial_binning(pbc, reduction, direction, axes, bins):
    system = tilted(pbc=pbc)
    operation = {
        "mean": "Mean",
        "sum": "Sum",
        "density": "SumVol",
        "min": "Min",
        "max": "Max",
    }
    modifier = SpatialBinningModifier(
        property="Position.X",
        direction=getattr(SpatialBinningModifier.Direction, direction),
        bin_count=bins if len(bins) > 1 else bins[0],
        reduction_operation=getattr(
            SpatialBinningModifier.Operation, operation[reduction]
        ),
    )
    output = ovito_compute(system, modifier)
    if len(axes) == 1:
        theirs = np.asarray(output.tables["binning"].y[...])
    else:
        grid = output.grids["binning"]
        # the first axis fastest
        theirs = np.asarray(grid["Position.X"]).reshape(bins[::-1]).T
    ours = SpatialBinning("positions.x", bins, axes=axes, reduction=reduction).compute(
        system
    )
    filled = ours.counts > 0  # OVITO gives 0 for an empty bin
    np.testing.assert_allclose(ours.values[filled], theirs[filled], rtol=1e-12)
    if reduction in ("sum", "density"):
        np.testing.assert_allclose(ours.values[~filled], theirs[~filled])


@pytest.mark.parametrize("minimum_image", [True, False])
@pytest.mark.parametrize("affine", [False, True])
def test_atomic_displacement(minimum_image, affine):
    reference = tilted(100)
    rng = np.random.default_rng(9)
    strain = np.eye(3) + [[0.03, 0.02, 0], [0, -0.01, 0.01], [0, 0, 0.02]]
    box = mdapy.Box(reference.box.matrix @ strain, origin=reference.box.origin)
    positions = box.to_cartesian(reference.box.to_fractional(reference["positions"]))
    current = mdapy.System(positions + rng.normal(0, 0.8, (100, 3)), box, elements="Cu")
    current = current.with_columns(id=reference["id"]).gather(rng.permutation(100))
    modifier = CalculateDisplacementsModifier(
        minimum_image_convention=minimum_image, affine_mapping=mapping(affine)
    )
    modifier.reference = ovito_source(reference)
    theirs = ovito_compute(current, modifier).particles
    ours = AtomicDisplacement(
        reference, minimum_image=minimum_image, affine=affine
    ).compute(current)
    np.testing.assert_allclose(ours.displacement, theirs["Displacement"], atol=1e-12)
    np.testing.assert_allclose(
        ours.magnitude, theirs["Displacement Magnitude"], atol=1e-12
    )


def ovito_structure_factor(system, mode, q_max, bins):
    xy = np.asarray(
        ovito_compute(
            system, StructureFactorModifier(k_max=int(q_max), k_bins=bins, mode=mode)
        )
        .tables["structure-factor"]
        .xy()
    )
    return xy[:, 0], xy[:, 1]


def test_structure_factor_octant_of_ovito():
    """OVITO's direct method averages over the wave vectors of one octant,
    h, k, l >= 0, which weights a family of equivalent vectors by how many
    of them have zero components; mdapy over all of them. Each is NumPy's
    average over its set."""
    rng = np.random.default_rng(2)
    box = mdapy.Box(14.0)
    system = mdapy.System(box.to_cartesian(rng.random((300, 3))), box, elements="Cu")
    q, expected = ovito_structure_factor(
        system, StructureFactorModifier.Mode.Direct, 6.0, 60
    )
    m = int(6.0 * 14.0 / (2 * np.pi)) + 1
    hkl = np.array(
        [
            (h, k, l)
            for h in range(-m, m + 1)
            for k in range(-m, m + 1)
            for l in range(-m, m + 1)  # noqa: E741
            if (h, k, l) != (0, 0, 0)
        ]
    )
    vectors = 2 * np.pi * hkl / 14.0
    length = np.linalg.norm(vectors, axis=1)
    keep = length <= 6.0
    hkl, vectors, length = hkl[keep], vectors[keep], length[keep]
    f = np.exp(1j * system["positions"] @ vectors.T).sum(axis=0) / np.sqrt(len(system))
    each = (f.conj() * f).real
    binned = np.minimum((length / 0.1).astype(int), 59)

    def mean(mask):
        return np.array(
            [
                each[mask & (binned == b)].mean()
                if (mask & (binned == b)).any()
                else np.nan
                for b in range(60)
            ]
        )

    octant = mean((hkl >= 0).all(axis=1))
    found = np.isfinite(octant)
    np.testing.assert_allclose(expected[found], octant[found], rtol=1e-10, atol=1e-12)
    ours = StructureFactor(6.0, 60).compute(system)
    everything = mean(np.ones(len(hkl), dtype=bool))
    np.testing.assert_allclose(ours.s[found], everything[found], rtol=1e-10, atol=1e-12)
    assert np.abs(octant[found] - everything[found]).max() > 0.1  # not the same


@pytest.mark.parametrize("lengths", [14.0, (14.0, 13.0, 15.0)])
def test_structure_factor_debye(lengths):
    """OVITO's Debye method transforms g(r) itself, 1 plus the transform of
    the uniform density over mdapy's, which transforms g(r) - 1; both with
    the window sinc(2 r / L) up to half the longest cell vector L."""
    rng = np.random.default_rng(2)
    box = mdapy.Box(lengths)
    system = mdapy.System(box.to_cartesian(rng.random((300, 3))), box, elements="Cu")
    q, expected = ovito_structure_factor(
        system, StructureFactorModifier.Mode.Debye, 6.0, 60
    )
    ours = StructureFactor(6.0, 60, method="debye").compute(system)
    r_max = 0.5 * np.linalg.norm(box.matrix, axis=1).max()
    edges = np.linspace(0.0, r_max, int(np.ceil(r_max / 0.01)) + 1)
    r, dr = (edges[1:] + edges[:-1]) / 2, edges[1] - edges[0]
    density = len(system) / box.volume
    uniform = (
        4
        * np.pi
        * density
        * dr
        * np.sin(np.outer(q, r))
        * r
        * np.sinc(r / r_max)
        / q[:, None]
    ).sum(axis=1)
    # the rest: OVITO's own bins of g(r)
    np.testing.assert_allclose(ours.s + uniform, expected, rtol=0, atol=2e-3)
