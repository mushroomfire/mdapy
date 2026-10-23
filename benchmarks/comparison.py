# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""mdapy against OVITO and freud: each case computes the same quantity
with mdapy and with the programs that have it, on the same system, and a
program is timed only where it gives the result of mdapy.

Every measurement runs in a fresh process, so that the time and the memory
of one program do not include another, and a process that grows beyond the
memory limit is stopped. Sizes grow eightfold; a size is skipped where the
previous one predicts more memory than the limit.

    python benchmarks/comparison.py run                  # every case
    python benchmarks/comparison.py run cna rdf --sizes 4000 32000 --threads 1
    python benchmarks/comparison.py table results.json   # Markdown
    python benchmarks/comparison.py plot results.json timings.png
"""

import argparse
import gc
import json
import os
import platform
import resource
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

SIZES = (4_000, 32_000, 256_000, 2_048_000)
WARM_UP = 500  # atoms of the system each process computes once before timing
PROGRAMS = {"ovito": "OVITO", "freud": "freud"}


@dataclass(frozen=True)
class Other:
    """The side of another program: ``prepare(inputs)`` does everything that
    is not timed and returns the timed call; ``agree(ours, theirs, inputs)``
    raises AssertionError where its result differs from that of mdapy."""

    prepare: Callable[[dict], Callable[[], Any]]
    agree: Callable[[Any, Any, dict], None]


@dataclass(frozen=True)
class Case:
    """One quantity computed by mdapy and other programs.

    ``inputs(atoms)`` makes the systems, the same in every process.
    ``mdapy(inputs)`` prepares everything that is not timed and returns the
    timed call; ``others`` holds the programs that compute the same, by name.
    """

    title: str
    inputs: Callable[[int], dict]
    mdapy: Callable[[dict], Callable[[], Any]]
    others: dict[str, Other]


# ------------------------------------------------------------ systems


def crystal(structure: str, a: float, atoms: int, sigma: float, seed: int = 1):
    """A cubic periodic crystal of about that many atoms, each displaced at
    random by sigma Å along each axis."""
    from mdapy.build import build_crystal

    per_cell = len(build_crystal(structure, a))
    n = max(1, round((atoms / per_cell) ** (1 / 3)))
    perfect = build_crystal(structure, a, repeat=(n, n, n))
    rng = np.random.default_rng(seed)
    return perfect.with_positions(
        perfect.positions + rng.normal(0.0, sigma, (len(perfect), 3))
    )


def copper(atoms: int) -> dict:
    """fcc copper at about room temperature."""
    return {"system": crystal("fcc", 3.615, atoms, 0.1)}


def silicon(atoms: int) -> dict:
    return {"system": crystal("diamond", 5.431, atoms, 0.05)}


def gas(atoms: int) -> dict:
    """Random points at the density of copper; with a cutoff of 2 Å, close
    to percolation, so clusters of every size. Points with a neighbor within
    1e-4 Å of 2 Å are left out, so that the bonds are the same in single
    precision."""
    import mdapy
    from mdapy.neighbor import NeighborList

    side = (atoms / 0.0847) ** (1 / 3)
    rng = np.random.default_rng(1)
    system = mdapy.System(rng.random((atoms, 3)) * side, mdapy.Box(side))
    pairs = NeighborList.build(system, 2.0 + 1e-4)
    close = np.abs(pairs.distances - 2.0) < 1e-4
    kept = np.ones(len(system), dtype=bool)
    kept[pairs.i[close]] = False
    return {"system": system.select(kept)}


def deformed(atoms: int) -> dict:
    """A perfect fcc reference and the same crystal strained, its atoms
    displaced."""
    import mdapy

    reference = crystal("fcc", 3.615, atoms, 0.0)
    strain = np.eye(3) + [[0.02, 0.03, 0.0], [0.0, -0.01, 0.01], [0.0, 0.0, 0.01]]
    rng = np.random.default_rng(2)
    moved = reference.positions + rng.normal(0.0, 0.05, (len(reference), 3))
    current = mdapy.System(moved @ strain.T, mdapy.Box(reference.box.matrix @ strain.T))
    return {"system": current, "reference": reference}


def damaged(atoms: int) -> dict:
    """A perfect fcc reference and the same crystal with its atoms displaced
    and 0.5 % of them gone."""
    reference = crystal("fcc", 3.615, atoms, 0.0)
    current = crystal("fcc", 3.615, atoms, 0.2)
    rng = np.random.default_rng(3)
    kept = rng.random(len(current)) > 0.005
    return {"system": current.select(kept), "reference": reference}


# ------------------------------------------------------------ OVITO


def ovito_data(system):
    """The system as an OVITO data collection."""
    from ovito.data import DataCollection

    data = DataCollection()
    cell = np.zeros((3, 4))
    cell[:, :3] = system.box.matrix.T  # OVITO keeps the cell vectors as columns
    cell[:, 3] = system.box.origin
    data.create_cell(cell, pbc=tuple(bool(p) for p in system.box.pbc))
    particles = data.create_particles(count=len(system))
    particles.create_property("Position", data=system.positions)
    return data


def pipeline(modifiers: Callable[[dict], list]) -> Callable[[dict], Callable]:
    """The OVITO side of a case: a pipeline of the modifiers, new for each
    call so that nothing is cached."""

    def prepare(inputs: dict) -> Callable:
        from ovito.pipeline import Pipeline, StaticSource

        data = ovito_data(inputs["system"])
        references = {"reference": None}
        if "reference" in inputs:
            references["reference"] = StaticSource(data=ovito_data(inputs["reference"]))

        def run():
            p = Pipeline(source=StaticSource(data=data))
            for modifier in modifiers(references):
                p.modifiers.append(modifier)
            return p.compute()

        return run

    return prepare


class Member(str):
    """A member of an enumeration of the modifier, such as "Mode.FixedCutoff",
    looked up only in the process that runs OVITO."""


def modifier(name: str, **options) -> Callable[[dict], list]:
    def make(references: dict) -> list:
        import ovito.modifiers

        kind = getattr(ovito.modifiers, name)
        for key, value in options.items():
            if isinstance(value, Member):
                enumeration, member = value.split(".")
                options[key] = getattr(getattr(kind, enumeration), member)
        made = kind(**options)
        if references["reference"] is not None:
            made.reference = references["reference"]
        return [made]

    return make


def particles(name: str):
    return lambda data: np.asarray(data.particles[name])


def ovito(prepare: Callable[[dict], Callable], agree: Callable[[Any, Any], None]):
    """The OVITO side of a case; OVITO computes in double precision, as
    mdapy, so its results are compared without the inputs."""
    return Other(prepare, lambda ours, theirs, inputs: agree(ours, theirs))


# ------------------------------------------------------------ freud


def freud_side(compute: Callable[..., Any]) -> Callable[[dict], Callable]:
    """The freud side of a case: compute(box, points) with the system in
    freud's box, converted before the timing."""

    def prepare(inputs: dict) -> Callable:
        from mdapy.interop import to_freud

        box, points = to_freud(inputs["system"])
        return lambda: compute(box, points)

    return prepare


def _freud_neighbors(query: dict) -> Callable[[dict], Callable]:
    def compute(box, points):
        import freud

        found = freud.locality.AABBQuery(box, points).query(points, query)
        return found.toNeighborList()

    return freud_side(compute)


def _freud_voronoi_steinhardt(box, points):
    """Steinhardt q2 to q6 over the Voronoi faces, weighted by their areas."""
    import freud

    voronoi = freud.locality.Voronoi().compute((box, points))
    steinhardt = freud.order.Steinhardt(list(range(2, 7)), weighted=True)
    return steinhardt.compute((box, points), neighbors=voronoi.nlist)


def _freud(name: str, *args, **options) -> Callable[[dict], Callable]:
    """freud's compute of the class at that dotted name, made with these
    arguments, for the box and points of the system."""
    compute_options = options.pop("compute", {})

    def compute(box, points):
        import freud

        module, cls = name.split(".")
        made = getattr(getattr(freud, module), cls)(*args, **options)
        return made.compute((box, points), **compute_options)

    return freud_side(compute)


# ------------------------------------------------------------ agreement


def equal(ours, theirs) -> None:
    np.testing.assert_array_equal(ours, theirs)


def close(ours, theirs, tolerance: float = 1e-10) -> None:
    np.testing.assert_allclose(ours, theirs, rtol=tolerance, atol=tolerance)


# freud computes in single precision: positions in a box of 300 Å are
# rounded by some 1e-5 Å, so a pair this close to a cutoff may fall on either
# side of it, and values agree to this much
SINGLE = 1e-4


def structures(result, data, mapping=None) -> None:
    theirs = np.asarray(data.particles["Structure Type"])
    equal(result.structure, theirs if mapping is None else np.asarray(mapping)[theirs])


def analysis(make: Callable[[dict], Any]) -> Callable[[dict], Callable]:
    """The mdapy side of a case: make(inputs) is the analysis, computed for
    the system."""

    def prepare(inputs: dict) -> Callable:
        return lambda: make(inputs).compute(inputs["system"])

    return prepare


def together(make: Callable[[dict], list], cutoff: float) -> Callable[[dict], Callable]:
    """The mdapy side of analyses together: one list within cutoff, built
    inside the timing, given to each of make(inputs)."""

    def prepare(inputs: dict) -> Callable:
        from mdapy.neighbor import NeighborList

        analyses, system = make(inputs), inputs["system"]

        def run():
            nl = NeighborList.build(system, cutoff)
            return [a.compute(system, neighbors=nl) for a in analyses]

        return run

    return prepare


def _freud_steinhardt_rdf(box, points):
    """Steinhardt q2 to q6 of the 12 nearest and the radial distribution
    function within 5 Å, with one tree of freud for both queries; given a
    list of the pairs instead, freud's radial distribution function takes
    fifty times as long."""
    import freud

    query = freud.locality.AABBQuery(box, points)
    rdf = freud.density.RDF(bins=200, r_max=5.0).compute(query)
    nearest = {"num_neighbors": 12, "exclude_ii": True}
    steinhardt = freud.order.Steinhardt(list(range(2, 7))).compute(
        query, neighbors=nearest
    )
    return steinhardt, rdf


# ------------------------------------------------------------ cases


def _neighbors_ovito(inputs: dict) -> Callable:
    from ovito.data import CutoffNeighborFinder

    data = ovito_data(inputs["system"])
    return lambda: CutoffNeighborFinder(5.0, data).find_all()


def _neighbors_mdapy(inputs: dict) -> Callable:
    from mdapy.neighbor import NeighborList

    return lambda: NeighborList.build(inputs["system"], 5.0)


def _neighbors_agree(ours, theirs) -> None:
    pairs, vectors = theirs
    n = len(ours.counts)
    equal(ours.counts, np.bincount(pairs[:, 0], minlength=n))
    summed = np.bincount(pairs[:, 0], np.linalg.norm(vectors, axis=1), minlength=n)
    close(np.add.reduceat(ours.distances, ours.offsets[:-1]), summed)


def _nearest_ovito(inputs: dict) -> Callable:
    from ovito.data import NearestNeighborFinder

    data = ovito_data(inputs["system"])
    return lambda: NearestNeighborFinder(12, data).find_all()


def _nearest_mdapy(inputs: dict) -> Callable:
    from mdapy.neighbor import NeighborList

    return lambda: NeighborList.build(inputs["system"], num_neighbors=12)


def _nearest_agree(ours, theirs) -> None:
    _, vectors = theirs
    distances = np.sort(np.linalg.norm(vectors, axis=2), axis=1)
    close(ours.distances.reshape(-1, 12), distances, 1e-12)


def _cases() -> dict[str, Case]:
    from mdapy.analysis import (
        AtomicStrain,
        BondAngleDistribution,
        ClusterAnalysis,
        RadialDistributionFunction,
        SteinhardtParameters,
        VoronoiAnalysis,
        WignerSeitz,
    )
    from mdapy.identify import (
        AcklandJonesAnalysis,
        CentroSymmetryParameter,
        CommonNeighborAnalysis,
        DiamondStructure,
        PolyhedralTemplateMatching,
    )

    degrees = tuple(range(2, 7))

    def steinhardt(ours, data, tolerance=1e-10):
        theirs = np.asarray(data.particles["Steinhardt"])
        for degree in degrees:
            close(ours[degree], theirs[:, degree - 2], tolerance)

    def steinhardt_voronoi(ours, data):
        # the areas of faces come from two tessellations, with different
        # rounding, and weight the sums
        steinhardt(ours, data, 1e-8)

    def steinhardt_freud(ours, theirs, inputs, tolerance=SINGLE):
        close(ours.q, theirs.particle_order, tolerance)

    def steinhardt_voronoi_freud(ours, theirs, inputs):
        # freud leaves out some faces of distant neighbors below some 2e-3 Å²,
        # which have that weight in the sums
        steinhardt_freud(ours, theirs, inputs, 1e-3)

    def neighbors_freud(ours, theirs, inputs):
        from mdapy.neighbor import NeighborList

        n = len(ours.counts)
        counts = np.bincount(theirs.query_point_indices, minlength=n)
        wider = NeighborList.build(inputs["system"], 5.0 + SINGLE)
        edge = np.bincount(ours.i[ours.distances > 5.0 - SINGLE], minlength=n)
        assert np.all(counts <= wider.counts) and np.all(counts >= ours.counts - edge)
        same = counts == ours.counts
        summed = np.bincount(theirs.query_point_indices, theirs.distances, minlength=n)
        ours_summed = np.add.reduceat(ours.distances, ours.offsets[:-1])
        close(ours_summed[same], summed[same], SINGLE)

    def nearest_freud(ours, theirs, inputs):
        order = np.lexsort((theirs.distances, theirs.query_point_indices))
        close(
            ours.distances.reshape(-1, 12),
            theirs.distances[order].reshape(-1, 12),
            SINGLE,
        )

    def rdf_freud(ours, theirs, inputs):
        from mdapy.neighbor import NeighborList

        # a pair this close to the edge of a bin may fall into the next one in
        # single precision: each bin may differ by the pairs at its edges
        reach = ours.edges[-1] + SINGLE
        distances = NeighborList.build(inputs["system"], reach).distances
        width = ours.edges[1] - ours.edges[0]
        edge = np.rint(distances / width).astype(np.int64)
        near = np.abs(distances - edge * width) < SINGLE
        slack = np.bincount(edge[near], minlength=len(ours.edges))
        counts = np.histogram(distances, ours.edges)[0]
        per_pair = np.divide(
            ours.g, counts, out=np.zeros_like(ours.g), where=counts > 0
        )
        allowed = (slack[:-1] + slack[1:]) * per_pair.max()
        # and freud normalizes in single precision
        assert np.all(np.abs(ours.g - theirs.rdf) <= allowed + SINGLE * ours.g + 1e-6)

    def voronoi_freud(ours, theirs, inputs):
        close(ours.volume, theirs.volumes, SINGLE)
        # freud leaves out some faces of distant neighbors below some 2e-3 Å²:
        # every face above 1e-2 Å² of one program is a face of the other
        n = len(ours.volume)
        mine = np.repeat(np.arange(n), np.diff(ours.offsets)) * n + ours.j
        nlist = theirs.nlist
        found = nlist.query_point_indices.astype(np.int64) * n + nlist.point_indices
        assert np.isin(mine[ours.areas > 1e-2], found).all()
        assert np.isin(found[nlist.weights > 1e-2], mine).all()

    def metal(ours, data):
        # Ackland-Jones analysis runs last of the two and leaves its types
        structures(ours[1], data)
        close(ours[2].csp, particles("Centrosymmetry")(data))
        steinhardt(ours[3], data)
        rdf(ours[4], data)
        bond_angles(ours[5], data)

    def steinhardt_rdf(ours, data):
        steinhardt(ours[0], data)
        rdf(ours[1], data)

    def steinhardt_rdf_freud(ours, theirs, inputs):
        steinhardt_freud(ours[0], theirs[0], inputs)
        rdf_freud(ours[1], theirs[1], inputs)

    def clusters_freud(ours, theirs, inputs):
        equal(ours.sizes, np.sort(np.bincount(theirs.cluster_idx))[::-1])

    def rdf(ours, data):
        close(ours.g, data.tables["coordination-rdf"].xy()[:, 1])

    def bond_angles(ours, data):
        equal(ours.counts, data.tables["bond-angle-distr"].xy()[:, 1])

    def voronoi(ours, data):
        close(ours.volume, particles("Atomic Volume")(data))
        equal(ours.coordination, particles("Coordination")(data))

    def clusters(ours, data):
        theirs = np.bincount(particles("Cluster")(data))
        equal(ours.sizes, np.sort(theirs)[::-1])

    def wigner_seitz(ours, data):
        equal(ours.occupancy, particles("Occupancy")(data))

    def strain(ours, data):
        close(ours.shear, particles("Shear Strain")(data))
        close(ours.volumetric, particles("Volumetric Strain")(data))

    return {
        "neighbors": Case(
            "Neighbor list, 5 Å",
            copper,
            _neighbors_mdapy,
            {
                "ovito": ovito(_neighbors_ovito, _neighbors_agree),
                "freud": Other(
                    _freud_neighbors({"r_max": 5.0, "exclude_ii": True}),
                    neighbors_freud,
                ),
            },
        ),
        "nearest": Case(
            "12 nearest neighbors",
            copper,
            _nearest_mdapy,
            {
                "ovito": ovito(_nearest_ovito, _nearest_agree),
                "freud": Other(
                    _freud_neighbors({"num_neighbors": 12, "exclude_ii": True}),
                    nearest_freud,
                ),
            },
        ),
        "cna": Case(
            "Adaptive common neighbor analysis",
            copper,
            analysis(lambda inputs: CommonNeighborAnalysis()),
            {
                "ovito": ovito(
                    pipeline(
                        modifier(
                            "CommonNeighborAnalysisModifier",
                            mode=Member("Mode.AdaptiveCutoff"),
                        )
                    ),
                    structures,
                )
            },
        ),
        "cna-fixed": Case(
            "Common neighbor analysis, cutoff 3.09 Å",
            copper,
            analysis(lambda inputs: CommonNeighborAnalysis(3.086)),
            {
                "ovito": ovito(
                    pipeline(
                        modifier(
                            "CommonNeighborAnalysisModifier",
                            mode=Member("Mode.FixedCutoff"),
                            cutoff=3.086,
                        )
                    ),
                    structures,
                )
            },
        ),
        "ackland-jones": Case(
            "Ackland–Jones analysis",
            copper,
            analysis(lambda inputs: AcklandJonesAnalysis()),
            {"ovito": ovito(pipeline(modifier("AcklandJonesModifier")), structures)},
        ),
        "ptm": Case(
            "Polyhedral template matching",
            copper,
            analysis(lambda inputs: PolyhedralTemplateMatching()),
            {
                "ovito": ovito(
                    pipeline(modifier("PolyhedralTemplateMatchingModifier")), structures
                )
            },
        ),
        "csp": Case(
            "Centro-symmetry parameter, 12 neighbors",
            copper,
            analysis(lambda inputs: CentroSymmetryParameter(12)),
            {
                "ovito": ovito(
                    pipeline(modifier("CentroSymmetryModifier", num_neighbors=12)),
                    lambda ours, data: close(
                        ours.csp, particles("Centrosymmetry")(data)
                    ),
                )
            },
        ),
        "diamond": Case(
            "Diamond structure",
            silicon,
            analysis(lambda inputs: DiamondStructure()),
            {
                "ovito": ovito(
                    pipeline(modifier("IdentifyDiamondModifier")),
                    # OVITO also marks the neighbors of diamond atoms
                    lambda ours, data: structures(ours, data, [0, 1, 0, 0, 2, 0, 0]),
                )
            },
        ),
        "steinhardt": Case(
            "Steinhardt q2–q6, 12 neighbors",
            copper,
            analysis(lambda inputs: SteinhardtParameters(degrees, num_neighbors=12)),
            {
                "ovito": ovito(
                    pipeline(
                        modifier(
                            "SteinhardtParametersModifier",
                            neighbor_mode=Member("NeighborMode.NearestNeighbors"),
                            num_neighbors=12,
                            max_l=6,
                        )
                    ),
                    steinhardt,
                ),
                "freud": Other(
                    _freud(
                        "order.Steinhardt",
                        list(degrees),
                        compute={
                            "neighbors": {"num_neighbors": 12, "exclude_ii": True}
                        },
                    ),
                    steinhardt_freud,
                ),
            },
        ),
        "steinhardt-voronoi": Case(
            "Steinhardt q2–q6, Voronoi faces weighted by area",
            copper,
            analysis(
                lambda inputs: SteinhardtParameters(
                    degrees, voronoi=True, weighted=True
                )
            ),
            {
                "ovito": ovito(
                    pipeline(
                        modifier(
                            "SteinhardtParametersModifier",
                            neighbor_mode=Member("NeighborMode.Voronoi"),
                            max_l=6,
                        )
                    ),
                    steinhardt_voronoi,
                ),
                "freud": Other(
                    freud_side(_freud_voronoi_steinhardt), steinhardt_voronoi_freud
                ),
            },
        ),
        "rdf": Case(
            "Radial distribution function, 6 Å, 200 bins",
            copper,
            analysis(lambda inputs: RadialDistributionFunction(6.0, bins=200)),
            {
                "ovito": ovito(
                    pipeline(
                        modifier(
                            "CoordinationAnalysisModifier",
                            cutoff=6.0,
                            number_of_bins=200,
                        )
                    ),
                    rdf,
                ),
                "freud": Other(_freud("density.RDF", bins=200, r_max=6.0), rdf_freud),
            },
        ),
        "bond-angles": Case(
            "Bond angle distribution, 3 Å, 180 bins",
            copper,
            analysis(lambda inputs: BondAngleDistribution(cutoff=3.0, bins=180)),
            {
                "ovito": ovito(
                    pipeline(
                        lambda references: (
                            modifier("CreateBondsModifier", cutoff=3.0)(references)
                            + modifier("BondAngleDistributionModifier", bins=180)(
                                references
                            )
                        )
                    ),
                    bond_angles,
                )
            },
        ),
        "voronoi": Case(
            "Voronoi volumes and coordination",
            copper,
            analysis(lambda inputs: VoronoiAnalysis()),
            {
                "ovito": ovito(pipeline(modifier("VoronoiAnalysisModifier")), voronoi),
                "freud": Other(_freud("locality.Voronoi"), voronoi_freud),
            },
        ),
        "clusters": Case(
            "Cluster analysis, 2 Å",
            gas,
            analysis(lambda inputs: ClusterAnalysis(cutoff=2.0)),
            {
                "ovito": ovito(
                    pipeline(
                        modifier(
                            "ClusterAnalysisModifier", cutoff=2.0, sort_by_size=True
                        )
                    ),
                    clusters,
                ),
                "freud": Other(
                    _freud("cluster.Cluster", compute={"neighbors": {"r_max": 2.0}}),
                    clusters_freud,
                ),
            },
        ),
        "wigner-seitz": Case(
            "Wigner–Seitz occupancies",
            damaged,
            analysis(lambda inputs: WignerSeitz(inputs["reference"])),
            {
                "ovito": ovito(
                    pipeline(modifier("WignerSeitzAnalysisModifier")), wigner_seitz
                )
            },
        ),
        "metal": Case(
            "Six analyses of copper, one neighbor search",
            copper,
            together(
                lambda inputs: [
                    CommonNeighborAnalysis(),
                    AcklandJonesAnalysis(),
                    CentroSymmetryParameter(12),
                    SteinhardtParameters(degrees, num_neighbors=12),
                    RadialDistributionFunction(5.0, bins=200),
                    BondAngleDistribution(cutoff=3.0, bins=180),
                ],
                5.0,
            ),
            {
                "ovito": ovito(
                    pipeline(
                        lambda references: [
                            m
                            for make in (
                                modifier(
                                    "CommonNeighborAnalysisModifier",
                                    mode=Member("Mode.AdaptiveCutoff"),
                                ),
                                modifier("AcklandJonesModifier"),
                                modifier("CentroSymmetryModifier", num_neighbors=12),
                                modifier(
                                    "SteinhardtParametersModifier",
                                    neighbor_mode=Member(
                                        "NeighborMode.NearestNeighbors"
                                    ),
                                    num_neighbors=12,
                                    max_l=6,
                                ),
                                modifier(
                                    "CoordinationAnalysisModifier",
                                    cutoff=5.0,
                                    number_of_bins=200,
                                ),
                                modifier("CreateBondsModifier", cutoff=3.0),
                                modifier("BondAngleDistributionModifier", bins=180),
                            )
                            for m in make(references)
                        ]
                    ),
                    metal,
                )
            },
        ),
        "steinhardt-rdf": Case(
            "Steinhardt q2–q6 and the RDF, one neighbor search",
            copper,
            together(
                lambda inputs: [
                    SteinhardtParameters(degrees, num_neighbors=12),
                    RadialDistributionFunction(5.0, bins=200),
                ],
                5.0,
            ),
            {
                "ovito": ovito(
                    pipeline(
                        lambda references: (
                            modifier(
                                "SteinhardtParametersModifier",
                                neighbor_mode=Member("NeighborMode.NearestNeighbors"),
                                num_neighbors=12,
                                max_l=6,
                            )(references)
                            + modifier(
                                "CoordinationAnalysisModifier",
                                cutoff=5.0,
                                number_of_bins=200,
                            )(references)
                        )
                    ),
                    steinhardt_rdf,
                ),
                "freud": Other(freud_side(_freud_steinhardt_rdf), steinhardt_rdf_freud),
            },
        ),
        "atomic-strain": Case(
            "Atomic strain, 3 Å",
            deformed,
            analysis(lambda inputs: AtomicStrain(inputs["reference"], 3.0)),
            {
                "ovito": ovito(
                    pipeline(modifier("AtomicStrainModifier", cutoff=3.0)), strain
                )
            },
        ),
    }


# ------------------------------------------------------------ one process


def peak_memory() -> int:
    """The largest resident memory of this process so far, in bytes."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def _load(program: str) -> None:
    """Imports a program before any system, so that its memory counts as
    that of the libraries, and gives it the threads of the run."""
    if program == "ovito":
        import ovito.data  # noqa: F401
        import ovito.modifiers  # noqa: F401
        import ovito.pipeline  # noqa: F401
    elif program == "freud":
        import freud

        freud.parallel.set_num_threads(int(os.environ.get("MDAPY_NUM_THREADS", "0")))


def measure(name: str, atoms: int, side: str, repeat: int) -> dict:
    """In a process of its own: whether a program agrees with mdapy (side
    "check-<program>"), or the times of one program (side "mdapy" or the
    program)."""
    case = _cases()[name]
    program = side.removeprefix("check-")
    if program != "mdapy":
        _load(program)
    base = peak_memory()  # the libraries, before any system
    if side.startswith("check-"):
        inputs = case.inputs(atoms)
        other = case.others[program]
        try:
            other.agree(case.mdapy(inputs)(), other.prepare(inputs)(), inputs)
        except AssertionError as error:
            return {"agree": False, "message": str(error)[:2000]}
        size = len(inputs["system"])
        return {"agree": True, "atoms": size, "base": base, "peak": peak_memory()}
    prepare = case.mdapy if side == "mdapy" else case.others[side].prepare
    prepare(case.inputs(WARM_UP))()
    inputs = case.inputs(atoms)
    run = prepare(inputs)
    gc.collect()
    before = peak_memory()
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = run()
        times.append(time.perf_counter() - start)
        del result
    return {"times": times, "base": base, "before": before, "peak": peak_memory()}


# ------------------------------------------------------------ the runs


def resident(pid: int) -> int:
    """The resident memory of a process now, in bytes."""
    out = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True
    ).stdout.strip()
    return int(out) * 1024 if out else 0


def spawn(name, atoms, side, repeat, threads, limit, timeout=None) -> dict:
    """measure() in a new process, stopped above limit bytes or after
    timeout seconds."""
    env = dict(
        os.environ, MDAPY_NUM_THREADS=str(threads), OVITO_THREAD_COUNT=str(threads)
    )
    command = [sys.executable, __file__, "_measure", name, str(atoms), side]
    command.append(str(repeat))
    process = subprocess.Popen(
        command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    largest, start = 0, time.perf_counter()
    while process.poll() is None:
        largest = max(largest, resident(process.pid))
        stopped = None
        if largest > limit:
            stopped = f"more than {limit / 2**30:.1f} GB"
        elif timeout is not None and time.perf_counter() - start > timeout:
            stopped = f"more than {timeout:g} s"
        if stopped is not None:
            process.kill()
            process.communicate()  # waits, and closes the pipes
            return {"stopped": stopped}
        time.sleep(0.05)
    out, err = process.communicate()
    if process.returncode != 0:
        return {"failed": err.strip().splitlines()[-1] if err.strip() else "?"}
    return json.loads(out.strip().splitlines()[-1])


def version(program: str) -> str:
    """The version of a program, asked in a process of its own."""
    code = {
        "mdapy": "import mdapy; print(mdapy.__version__)",
        "ovito": "import ovito; print(ovito.version_string)",
        "freud": "import freud; print(freud.__version__)",
    }[program]
    found = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    return found.stdout.strip() or "not installed"


def run(names, sizes, threads, repeat, limit, output, timeout=None) -> list[dict]:
    """Every case at every size: check each program against mdapy, then
    time mdapy and every program that agrees; results appended to output as
    they come."""
    cases = _cases()
    results = []
    machine = {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "system": platform.platform(),
        "cpus": os.cpu_count(),
        "threads": threads,
        "versions": {p: version(p) for p in ("mdapy", *PROGRAMS)},
    }
    for name in names:
        # side -> atoms, peak memory and memory of the libraries alone
        peaks: dict[str, tuple[int, int, int]] = {}
        stopped: set[str] = set()  # sides stopped at a smaller size
        programs = list(cases[name].others)
        for atoms in sizes:
            entry = {
                "case": name,
                "size": atoms,
                "atoms": atoms,
                **machine,
                "check": {},
            }

            def measured(side: str) -> dict:
                last = peaks.get(side)
                if side in stopped:
                    return {"skipped": "stopped at a smaller size"}
                if last is not None and predicted(*last, atoms) > limit:
                    return {"skipped": "would need too much memory"}
                found = spawn(name, atoms, side, repeat, threads, limit, timeout)
                if "peak" in found:
                    peaks[side] = (atoms, found["peak"], found["base"])
                if "stopped" in found:
                    stopped.add(side)
                return found

            agreed = []
            for program in programs:
                entry["check"][program] = measured(f"check-{program}")
                if entry["check"][program].get("agree"):
                    agreed.append(program)
                    entry["atoms"] = entry["check"][program]["atoms"]
            # timed only where a program agrees with mdapy
            for side in ["mdapy", *agreed] if agreed else []:
                entry[side] = measured(side)
            results.append(entry)
            print(line(entry), flush=True)
            with open(output, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            if not agreed:
                break
    return results


def predicted(atoms: int, peak: int, base: int, more: int) -> float:
    """The peak memory for more atoms: that of the libraries, and the rest
    in proportion to the atoms."""
    return base + (peak - base) * more / atoms


def best(side: dict) -> float | None:
    return min(side["times"]) if "times" in side else None


def line(entry: dict) -> str:
    text = f"{entry['case']:20s} {entry['atoms']:>10,d}"
    ours = best(entry.get("mdapy", {}))
    text += "  mdapy " + (f"{ours:9.4f} s" if ours is not None else "-")
    for program, check in entry["check"].items():
        if not check.get("agree"):
            text += f"  {program} " + json.dumps(check)[:200]
            continue
        theirs = best(entry.get(program, {}))
        if theirs is None:
            text += f"  {program} " + json.dumps(entry.get(program))[:200]
        else:
            text += f"  {program} {theirs:9.4f} s"
            if ours:
                text += f" ×{theirs / ours:.2f}"
    return text


# ------------------------------------------------------------ reports


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(row) for row in f if row.strip()]


def extra(side: dict) -> float:
    """The memory a computation took beyond the systems, in MB."""
    return (side["peak"] - side["before"]) / 2**20


def timed(entries: list[dict], program: str) -> list[dict]:
    """The entries where mdapy and the program were both timed."""
    return [
        e
        for e in entries
        if e["check"].get(program, {}).get("agree")
        and best(e.get("mdapy", {})) is not None
        and best(e.get(program, {})) is not None
    ]


def label(entries: list[dict], program: str) -> str:
    """The program and its version, as the runs recorded it."""
    found = {
        e["versions"][program] for e in entries if program in e.get("versions", {})
    }
    return " ".join([PROGRAMS.get(program, program), *sorted(found)])


def outcome(side: dict | None) -> str:
    """Why a program has no time at a size."""
    if not side:
        return "not run"
    for key in ("stopped", "skipped", "failed"):
        if key in side:
            return f"{key}: {side[key]}"
    return "differs" if side.get("agree") is False else "?"


def table(entries: list[dict]) -> str:
    """The results as Markdown tables, one per program, every case at every
    size: the time of each program, how many times faster mdapy is, and the
    memory each computation took; why a program has no time where it has
    none."""
    titles = {name: case.title for name, case in _cases().items()}
    tables = []
    for program in PROGRAMS:
        mine = [e for e in entries if program in e["check"]]
        if not timed(entries, program):
            continue
        name, other = label(entries, program), PROGRAMS[program]
        rows = [
            f"| Case | Atoms | {label(entries, 'mdapy')} (s) | {name} (s) "
            f"| {other} / mdapy | mdapy (MB) | {other} (MB) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for case in titles:
            for entry in (e for e in mine if e["case"] == case):
                ours = best(entry.get("mdapy", {}))
                theirs = best(entry.get(program, {}))
                cells = [titles[case], f"{entry['atoms']:,d}"]
                cells.append(f"{ours:.3f}" if ours is not None else "")
                if theirs is None:
                    check = entry["check"][program]
                    why = outcome(
                        entry.get(program) or (None if check.get("agree") else check)
                    )
                    cells += [
                        why,
                        "",
                        f"{extra(entry['mdapy']):.0f}" if ours else "",
                        "",
                    ]
                else:
                    cells += [
                        f"{theirs:.3f}",
                        f"{theirs / ours:.2f}",
                        f"{extra(entry['mdapy']):.0f}",
                        f"{extra(entry[program]):.0f}",
                    ]
                rows.append("| " + " | ".join(cells) + " |")
        tables.append("\n".join(rows))
    return "\n\n".join(tables)


def plot(entries: list[dict], path: str) -> None:
    """The ratio of the times of each program and mdapy for each case and
    size, one panel per program."""
    from mdapy.visualization import plot as plots

    titles = {name: case.title for name, case in _cases().items()}
    panels = [(p, timed(entries, p)) for p in PROGRAMS if timed(entries, p)]
    counts = [len({e["case"] for e in done}) for _, done in panels]
    sizes = sorted({e["size"] for _, done in panels for e in done})
    with plots.style():
        fig, axes = plots.figure(
            "double",
            30 * len(panels) + 6.5 * sum(counts),
            nrows=len(panels),
            gridspec_kw={"height_ratios": [c + 4 for c in counts]},
        )
        axes = np.atleast_1d(axes)
        for ax, (program, done) in zip(axes, panels):
            names = [n for n in titles if any(e["case"] == n for e in done)]
            ratios = {
                (e["case"], e["size"]): best(e[program]) / best(e["mdapy"])
                for e in done
            }
            height = 0.8 / len(sizes)
            for k, size in enumerate(sizes):
                ax.barh(
                    np.arange(len(names)) + (k - (len(sizes) - 1) / 2) * height,
                    [ratios.get((name, size), np.nan) for name in names],
                    height,
                    color=plots.COLORS[k % len(plots.COLORS)],
                    label=f"{size:,d} atoms",
                )
            ax.axvline(1.0, color="k", lw=0.5)
            ax.set_xscale("log")
            ticks = (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200)
            low, high = ax.get_xlim()
            ticks = [t for t in ticks if low <= t <= high]
            ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
            ax.minorticks_off()
            ax.set_yticks(np.arange(len(names)), [titles.get(n, n) for n in names])
            ax.invert_yaxis()
            name = PROGRAMS[program]
            ax.set_title(f"{label(entries, 'mdapy')} against {label(entries, program)}")
            ax.set_xlabel(f"time of {name} / time of mdapy")
            # where mdapy is better, and worse, under the axis
            ax.annotate(
                "mdapy better →",
                (1.0, 0.0),
                xycoords="axes fraction",
                xytext=(0, -26),
                textcoords="offset points",
                ha="right",
                va="top",
            )
            ax.annotate(
                "← mdapy worse",
                (0.0, 0.0),
                xycoords="axes fraction",
                xytext=(0, -26),
                textcoords="offset points",
                ha="left",
                va="top",
            )
        axes[0].legend(frameon=False, loc="lower right")  # the same sizes below
        plots.save(fig, path.rsplit(".", 1)[0], formats=(path.rsplit(".", 1)[1],))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    runs = commands.add_parser("run", help="check and time the cases")
    runs.add_argument("cases", nargs="*", help="default: every case")
    runs.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    runs.add_argument("--threads", type=int, default=os.cpu_count())
    runs.add_argument("--repeat", type=int, default=3)
    runs.add_argument("--memory", type=float, default=8.0, help="limit in GB")
    runs.add_argument(
        "--timeout", type=float, default=600.0, help="limit of a process in s"
    )
    runs.add_argument("--output", default="results.json")
    for name in ("table", "plot"):
        report = commands.add_parser(name)
        report.add_argument("results")
        if name == "plot":
            report.add_argument("figure")
    child = commands.add_parser("_measure")
    child.add_argument("case")
    child.add_argument("atoms", type=int)
    child.add_argument("side")
    child.add_argument("repeat", type=int)
    options = parser.parse_args(argv)
    if options.command == "_measure":
        result = measure(options.case, options.atoms, options.side, options.repeat)
        print(json.dumps(result))
    elif options.command == "run":
        names = options.cases or list(_cases())
        unknown = set(names) - set(_cases())
        if unknown:
            parser.error(f"unknown cases {sorted(unknown)}; known {list(_cases())}")
        limit = int(options.memory * 2**30)
        run(
            names,
            options.sizes,
            options.threads,
            options.repeat,
            limit,
            options.output,
            options.timeout,
        )
    elif options.command == "table":
        print(table(load(options.results)))
    else:
        plot(load(options.results), options.figure)


if __name__ == "__main__":
    main()
