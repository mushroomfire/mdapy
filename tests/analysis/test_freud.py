# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Analyses against freud, run live. freud computes in single precision, so
values agree to about 1e-6. Skipped where freud is not installed."""

import subprocess
import sys

import numpy as np
import pytest

import mdapy
from mdapy.analysis import (
    ClusterAnalysis,
    MeanSquaredDisplacement,
    RadialDistributionFunction,
    SteinhardtParameters,
    VoronoiAnalysis,
)
from mdapy.build import build_crystal
from mdapy.neighbor import NeighborList

freud = pytest.importorskip("freud")


def copper(seed=0, repeat=6):
    """fcc copper with its atoms displaced at random, in a cubic box at the
    origin."""
    rng = np.random.default_rng(seed)
    crystal = build_crystal("fcc", 3.615, elements="Cu", repeat=(repeat,) * 3)
    return crystal.with_positions(
        crystal.positions + rng.normal(0, 0.1, (len(crystal), 3))
    )


def freud_input(system):
    """The cubic box of the system as freud's box, centered on the origin,
    and the atoms in it."""
    side = system.box.matrix[0, 0]
    box = freud.box.Box.cube(side)
    return box, box.wrap((system.positions - side / 2).astype(np.float32))


def test_radial_distribution_function():
    system = copper()
    ours = RadialDistributionFunction(6.0, bins=60).compute(system)
    theirs = freud.density.RDF(bins=60, r_max=6.0).compute(freud_input(system))
    np.testing.assert_allclose(ours.r, theirs.bin_centers, atol=1e-6)
    np.testing.assert_allclose(ours.g, theirs.rdf, atol=1e-4)
    np.testing.assert_allclose(ours.coordination, theirs.n_r, atol=1e-4)


@pytest.mark.parametrize(
    "neighbors, query",
    [
        ({"num_neighbors": 12}, {"num_neighbors": 12, "exclude_ii": True}),
        ({"cutoff": 3.0}, {"r_max": 3.0, "exclude_ii": True}),
    ],
)
@pytest.mark.parametrize("average", [False, True])
def test_steinhardt_parameters(neighbors, query, average):
    system = copper(seed=1)
    degrees = (2, 4, 6, 8)
    ours = SteinhardtParameters(degrees, average=average, **neighbors).compute(system)
    theirs = freud.order.Steinhardt(list(degrees), average=average)
    theirs.compute(freud_input(system), neighbors=query)
    np.testing.assert_allclose(ours.q, theirs.particle_order, atol=1e-5)


def test_clusters():
    rng = np.random.default_rng(2)
    side = (3000 / 0.0847) ** (1 / 3)
    gas = mdapy.System(rng.random((3000, 3)) * side, mdapy.Box(side))
    # no pair within 1e-4 Å of the cutoff, where single precision may differ
    pairs = NeighborList.build(gas, 2.0 + 1e-4)
    kept = np.ones(len(gas), dtype=bool)
    kept[pairs.i[np.abs(pairs.distances - 2.0) < 1e-4]] = False
    gas = gas.filter(kept)
    ours = ClusterAnalysis(cutoff=2.0).compute(gas)
    theirs = freud.cluster.Cluster().compute(freud_input(gas), neighbors={"r_max": 2.0})
    np.testing.assert_array_equal(
        ours.sizes, np.sort(np.bincount(theirs.cluster_idx))[::-1]
    )
    # the same atoms together
    for cluster in range(min(20, len(ours.sizes))):
        members = theirs.cluster_idx[ours.cluster == cluster]
        assert len(set(members.tolist())) == 1


@pytest.mark.parametrize("window, mode", [(True, "window"), (False, "direct")])
def test_mean_squared_displacement(window, mode):
    rng = np.random.default_rng(3)
    walk = np.cumsum(rng.normal(0, 0.3, (60, 80, 3)), axis=0)
    frames = [mdapy.System(r, mdapy.Box(1000.0, pbc=False)) for r in walk]
    ours = MeanSquaredDisplacement(window=window, unwrap=False).compute(frames)
    theirs = freud.msd.MSD(mode=mode).compute(walk)
    # freud's transforms leave some 1e-6 Å² at lag 0
    np.testing.assert_allclose(ours.msd, theirs.msd, rtol=1e-5, atol=1e-5)


# freud's Voronoi tessellation crashes in a process that has loaded OVITO, as
# the test session has; it runs in a process of its own
VORONOI = """
import sys
import freud
import numpy as np

side, points = float(sys.argv[2]), np.load(sys.argv[1])
box = freud.box.Box.cube(side)
voronoi = freud.locality.Voronoi().compute((box, points))
steinhardt = freud.order.Steinhardt([4, 6], weighted=True)
steinhardt.compute((box, points), neighbors=voronoi.nlist)
nlist = voronoi.nlist
np.savez(
    sys.argv[1] + ".out.npz",
    volumes=voronoi.volumes,
    i=nlist.query_point_indices,
    j=nlist.point_indices,
    areas=nlist.weights,
    q=steinhardt.particle_order,
)
"""


def freud_voronoi(system, tmp_path):
    box, points = freud_input(system)
    path = tmp_path / "points.npy"
    np.save(path, points)
    side = str(system.box.matrix[0, 0])
    subprocess.run([sys.executable, "-c", VORONOI, str(path), side], check=True)
    return np.load(str(path) + ".out.npz")


def test_voronoi_cells_and_their_steinhardt_parameters(tmp_path):
    system = copper(seed=4)
    theirs = freud_voronoi(system, tmp_path)
    ours = VoronoiAnalysis().compute(system)
    np.testing.assert_allclose(ours.volume, theirs["volumes"], rtol=1e-5)
    # every face above 1e-2 Å² of one is a face of the other: below it, faces
    # come and go in single precision
    n = len(system)
    mine = np.repeat(np.arange(n), np.diff(ours.offsets)) * n + ours.j
    found = theirs["i"].astype(np.int64) * n + theirs["j"]
    assert np.isin(mine[ours.areas > 1e-2], found).all()
    assert np.isin(found[theirs["areas"] > 1e-2], mine).all()
    weighted = SteinhardtParameters((4, 6), voronoi=True, weighted=True).compute(system)
    np.testing.assert_allclose(weighted.q, theirs["q"], atol=1e-5)
