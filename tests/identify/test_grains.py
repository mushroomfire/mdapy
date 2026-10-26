# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Grain segmentation, against OVITO's Grain segmentation atom by atom where
OVITO is installed, and by its invariances."""

import dataclasses

import numpy as np
import pytest

import mdapy as mp
from mdapy.build import build_crystal, build_polycrystal, twin_lamellae
from mdapy.errors import Cancelled, InvalidParameter
from mdapy.identify import GrainSegmentation, PolyhedralTemplateMatching
from mdapy.transform import replicate


def shaken(system, amplitude, seed):
    """The atoms displaced at random, so that no two choices are equal."""
    noise = np.random.default_rng(seed).normal(0, amplitude, (len(system), 3))
    return system.with_columns(positions=system["positions"] + noise)


def copper_grains(size=60.0, grains=8, seed=2):
    copper = build_crystal("fcc", 3.615, elements="Cu")
    return shaken(
        build_polycrystal(copper, mp.Box(size), grains, seed=seed), 0.04, seed
    )


def twinned_grains():
    layers = build_crystal(
        "fcc", 3.615, elements="Cu", orientation=[(1, 1, -2), (-1, 1, 0), (1, 1, 1)]
    )
    system = build_polycrystal(twin_lamellae(layers, 6), mp.Box(60.0), 8, seed=2)
    return shaken(system, 0.04, 3)


def iron_grains():
    iron = build_crystal("bcc", 2.866, elements="Fe")
    return shaken(build_polycrystal(iron, mp.Box(55.0), 8, seed=5), 0.04, 5)


def silicon_grains():
    silicon = build_crystal("diamond", 5.431, elements="Si")
    return shaken(build_polycrystal(silicon, mp.Box(60.0), 6, seed=6), 0.03, 6)


def open_copper_grains():
    system = copper_grains()
    return system.with_box(mp.Box(system.box.matrix, pbc=False))


SYSTEMS = {
    "copper": (copper_grains, ("fcc", "hcp", "bcc")),
    "open copper": (open_copper_grains, ("fcc", "hcp", "bcc")),
    "twinned": (twinned_grains, ("fcc", "hcp", "bcc")),
    "iron": (iron_grains, ("fcc", "hcp", "bcc")),
    "silicon": (silicon_grains, ("cubic_diamond", "hexagonal_diamond")),
}

# mdapy options and the OVITO settings that match them
OPTIONS = {
    "default": ({}, {}),
    "no orphans": ({"adopt_orphans": False}, {"orphan_adoption": False}),
    "no stacking faults": (
        {"stacking_faults": False},
        {"handle_stacking_faults": False},
    ),
    "manual": ({"threshold": 4.5}, {"algorithm": "manual", "merging_threshold": 4.5}),
    "spanning tree": (
        {"method": "spanning_tree", "threshold": 2.5, "min_grain_size": 200},
        {"algorithm": "tree", "merging_threshold": 2.5, "min_grain_size": 200},
    ),
}


@pytest.mark.parametrize("option", OPTIONS)
@pytest.mark.parametrize("name", SYSTEMS)
def test_against_ovito(name, option):
    pytest.importorskip("ovito")
    from ovito.modifiers import (
        GrainSegmentationModifier,
        PolyhedralTemplateMatchingModifier,
    )

    from ovito_reference import ovito_compute

    build, structures = SYSTEMS[name]
    system = build()
    ours_options, theirs_options = OPTIONS[option]
    ours = GrainSegmentation(structures=structures, **ours_options).compute(system)

    ptm = PolyhedralTemplateMatchingModifier(output_orientation=True)
    for t in ptm.structures:
        t.enabled = t.id in [
            PolyhedralTemplateMatching.NAMES.index(s) for s in structures
        ]
    theirs_options = dict(theirs_options)
    algorithm = theirs_options.pop("algorithm", None)
    if algorithm == "manual":
        theirs_options["algorithm"] = (
            GrainSegmentationModifier.Algorithm.GraphClusteringManual
        )
    elif algorithm == "tree":
        theirs_options["algorithm"] = (
            GrainSegmentationModifier.Algorithm.MinimumSpanningTree
        )
    data = ovito_compute(system, ptm, GrainSegmentationModifier(**theirs_options))
    table = data.tables["grains"]
    np.testing.assert_array_equal(ours.grain, data.particles["Grain"])
    np.testing.assert_array_equal(ours.size, table["Grain Size"])
    np.testing.assert_array_equal(ours.structure, table["Structure Type"])
    # OVITO keeps orientations in single precision
    np.testing.assert_allclose(ours.orientation, table["Orientation"], atol=1e-6)
    # the center of a grain that wraps around a periodic box is that of
    # whichever spanning tree of its bonds is unwrapped: compared in an open
    # box
    if not system.box.pbc.any():
        np.testing.assert_allclose(ours.center, table["Center of Mass"], atol=1e-6)
        np.testing.assert_allclose(ours.radius, table["Radius"], atol=1e-6)
        np.testing.assert_allclose(
            ours.radius_of_gyration, table["Radius of Gyration"], atol=1e-6
        )
    if option == "default":
        assert ours.count > 1


def test_single_crystal_is_one_grain():
    crystal = shaken(
        replicate(build_crystal("fcc", 3.615, elements="Cu"), (6, 6, 6)), 0.03, 1
    )
    result = GrainSegmentation().compute(crystal)
    assert result.count == 1
    assert result.size.tolist() == [len(crystal)]
    np.testing.assert_array_equal(result.grain, 0)
    assert result.names[result.structure[0]] == "fcc"
    assert abs(np.linalg.norm(result.orientation[0]) - 1) < 1e-12
    assert result.columns() == {"grain": result.grain}
    assert result.colors.shape == (1, 3)


def test_grains_follow_the_atoms():
    system = copper_grains(40.0, 3, seed=4)
    result = GrainSegmentation().compute(system)
    assert result.count == 3
    assert result.size.sum() == len(system)  # every atom adopted
    # the same grains wherever the origin; nearly the same in another order
    # of the atoms, which decides which pairs of disordered atoms are bonded
    order = np.random.default_rng(0).permutation(len(system))
    shuffled = GrainSegmentation().compute(system.gather(order))
    assert shuffled.count == result.count
    agree = 0
    for g in range(result.count):
        agree += np.bincount(shuffled.grain[result.grain[order] == g]).max()
    assert agree > 0.995 * len(system)
    moved = system.with_columns(positions=system["positions"] + [3.0, -2.0, 1.0])
    np.testing.assert_array_equal(
        GrainSegmentation().compute(moved).grain, result.grain
    )
    # the grains of the polycrystal: each mostly the atoms of one seed
    for g in range(result.count):
        _, counts = np.unique(system["grain"][result.grain == g], return_counts=True)
        assert counts.max() > 0.95 * counts.sum()


def test_orphans_and_sizes():
    system = copper_grains(40.0, 3, seed=4)
    free = GrainSegmentation(adopt_orphans=False).compute(system)
    assert (free.grain < 0).any()
    assert np.array_equal(np.bincount(free.grain[free.grain >= 0]), free.size)
    assert np.all(np.diff(free.size) <= 0)
    none = GrainSegmentation(min_grain_size=10**6).compute(system)
    assert none.count == 0 and (none.grain == -1).all()
    assert none.orientation.shape == (0, 4) and none.center.shape == (0, 3)
    np.testing.assert_array_equal(none.atom_colors(), 0.8)
    colors = free.atom_colors(none=(0, 0, 0))
    np.testing.assert_array_equal(colors[free.grain < 0], 0)
    np.testing.assert_array_equal(
        colors[free.grain >= 0], free.colors[free.grain[free.grain >= 0]]
    )


def test_geometry_of_a_cluster():
    # a grain away from the boundaries of the box: its center and radii as
    # NumPy computes them, weighted by the masses
    crystal = build_crystal("fcc", 3.615, elements="Cu", repeat=(8, 8, 8))
    center = crystal.box.matrix.diagonal() / 2
    ball = crystal.filter(np.linalg.norm(crystal["positions"] - center, axis=1) < 9.0)
    ball = shaken(ball.with_box(mp.Box(30.0, pbc=False)), 0.02, 2)
    masses = np.linspace(1.0, 2.0, len(ball))
    ball = ball.with_columns(mass=masses)
    result = GrainSegmentation(min_grain_size=10).compute(ball)
    assert result.count == 1
    x = ball["positions"]
    mean = (masses[:, None] * x).sum(0) / masses.sum()
    np.testing.assert_allclose(result.center[0], mean, atol=1e-10)
    r2 = ((x - mean) ** 2).sum(1)
    np.testing.assert_allclose(result.radius[0], np.sqrt(r2.max()), atol=1e-10)
    np.testing.assert_allclose(
        result.radius_of_gyration[0],
        np.sqrt((masses * r2).sum() / masses.sum()),
        atol=1e-10,
    )


def test_grain_across_the_boundary():
    # a ball of crystal centered at a corner of a periodic box, cut by its
    # faces: unwrapped, its center is the corner
    crystal = build_crystal("fcc", 3.615, elements="Cu", repeat=(10, 10, 10))
    x = crystal["positions"]
    lengths = crystal.box.matrix.diagonal()
    d = x - lengths * np.round(x / lengths)  # from the nearest corner
    ball = shaken(crystal.filter(np.linalg.norm(d, axis=1) < 10.0), 0.02, 4)
    result = GrainSegmentation(min_grain_size=10).compute(ball)
    assert result.count == 1
    d = ball["positions"] - lengths * np.round(ball["positions"] / lengths)
    np.testing.assert_allclose(result.center[0], d.mean(0) % lengths, atol=1e-10)
    np.testing.assert_allclose(
        result.radius_of_gyration[0],
        np.sqrt(((d - d.mean(0)) ** 2).sum(1).mean()),
        atol=1e-10,
    )


def test_merges_and_threshold():
    system = copper_grains(40.0, 3, seed=4)
    result = GrainSegmentation().compute(system)
    assert len(result.merge_distance) == len(result.merge_size) > 0
    assert np.all(np.diff(result.merge_distance) >= 0)
    assert np.all(result.merge_size >= 20)
    # the automatic threshold, given by hand, gives the same grains
    manual = GrainSegmentation(result.threshold).compute(system)
    np.testing.assert_array_equal(manual.grain, result.grain)
    tree = GrainSegmentation(3.0, method="spanning_tree").compute(system)
    assert tree.threshold == 3.0
    assert tree.count == result.count


def test_given_template_matching():
    system = copper_grains(40.0, 3, seed=4)
    ptm = PolyhedralTemplateMatching(template_neighbors=True).compute(system)
    np.testing.assert_array_equal(
        GrainSegmentation().compute(system, ptm=ptm).grain,
        GrainSegmentation().compute(system).grain,
    )
    with pytest.raises(InvalidParameter, match="template_neighbors"):
        GrainSegmentation().compute(
            system, ptm=dataclasses.replace(ptm, neighbors=None)
        )
    with pytest.raises(InvalidParameter, match="template_neighbors"):
        GrainSegmentation().compute(system.gather(np.arange(10)), ptm=ptm)


def test_progress_and_cancel():
    system = copper_grains(40.0, 3, seed=4)
    calls = []
    GrainSegmentation().compute(
        system, progress=lambda done, total: calls.append((done, total))
    )
    assert calls and calls[-1][0] == calls[-1][1]
    token = mp.CancelToken()
    token.cancel()
    with pytest.raises(Cancelled):
        GrainSegmentation().compute(system, cancel=token)


def test_invalid_arguments():
    with pytest.raises(InvalidParameter, match="method"):
        GrainSegmentation(method="kmeans")
    with pytest.raises(InvalidParameter, match="spanning tree"):
        GrainSegmentation(method="spanning_tree")
    with pytest.raises(InvalidParameter, match="threshold"):
        GrainSegmentation(float("nan"))
    with pytest.raises(InvalidParameter, match="min_grain_size"):
        GrainSegmentation(min_grain_size=0)
    with pytest.raises(InvalidParameter, match="min_grain_size"):
        GrainSegmentation(min_grain_size=True)
    with pytest.raises(InvalidParameter, match="too short"):
        GrainSegmentation().compute(build_crystal("fcc", 3.615, elements="Cu"))
    assert repr(GrainSegmentation()) == (
        "GrainSegmentation(None, method='graph', min_grain_size=100, "
        "stacking_faults=True, adopt_orphans=True)"
    )
