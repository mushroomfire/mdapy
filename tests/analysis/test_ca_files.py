# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The CA files of the dislocation analysis, read by OVITO 3.17.1: the same
lines, Burgers vectors, clusters, junctions, defect mesh and lengths."""

import numpy as np
import pytest

import mdapy
from mdapy.analysis import DislocationAnalysis
from mdapy.build import build_crystal, dislocation_cylinder, dislocation_quadrupole

from dislocation_models import CU, QUADRUPOLES, crossing_loops, hexagonal, shear_loop

ovito = pytest.importorskip("ovito")
from ovito.io import import_file  # noqa: E402


def crossing_loops_and_a_void():
    """Junctions where the loops cross and a defect mesh around a void."""
    system = crossing_loops()
    far = np.linalg.norm(system["positions"] - [9.0, 9.0, 9.0], axis=1) > 6.0
    return "fcc", system.filter(far)


def quadrupole(name):
    crystal, args, options = QUADRUPOLES[name]
    return crystal, dislocation_quadrupole(*args, **options)


def cylinder():
    """Partials that end on free surfaces, shaken."""
    system = dislocation_cylinder(
        "fcc edge", 3.615, CU, 22.0, elements="Cu", length=3, dissociation=8
    )
    noise = np.random.default_rng(5).normal(0, 0.03, (len(system), 3))
    return "fcc", system.with_columns(positions=system["positions"] + noise)


def open_cylinder():
    """Partials that end on the free surfaces at either end of the line."""
    crystal, system = cylinder()
    box = system.box
    return crystal, system.with_box(mdapy.Box(box.matrix, origin=box.origin, pbc=False))


def bcc_loop():
    """A closed loop of 1/2[111] in iron."""
    a = 2.865
    fe = build_crystal("bcc", a, elements="Fe", repeat=(12, 12, 12))
    center = fe.box.matrix.diagonal() / 2 + [0.11, 0.07, 0.03]
    return "bcc", shear_loop(
        fe, a / 2 * np.array([1, 1, 1]), center, 9.0, normal=[1, -1, 0]
    )


def triclinic():
    crystal, system = quadrupole("fcc edge, dissociated")
    matrix = system.box.matrix.copy()
    matrix[1] += 0.3 * matrix[0]
    fractional = system["positions"] @ system.box.inverse
    return crystal, system.with_box(mdapy.Box(matrix, pbc=True)).with_columns(
        positions=fractional @ matrix
    )


SAMPLES = {
    "loops and a void": crossing_loops_and_a_void,
    "fcc partials": lambda: quadrupole("fcc edge, dissociated"),
    "bcc": lambda: quadrupole("bcc screw"),
    "cubic diamond": lambda: quadrupole("diamond 60, dissociated"),
    "hcp": lambda: ("hcp", hexagonal("hcp", 3.2, "Mg", "screw")),
    "hexagonal diamond": lambda: (
        "hexagonal_diamond",
        hexagonal("lonsdaleite", 2.51, "C", "screw"),
    ),
    "free surfaces": cylinder,
    "free ends": open_cylinder,
    "a loop": bcc_loop,
    "triclinic": triclinic,
    "perfect crystal": lambda: (
        "bcc",
        build_crystal("bcc", 2.865, elements="Fe", repeat=(5, 5, 5)),
    ),
}


def nodes(result):
    """The line ends at each end of each line, as (line, at its last point),
    with the ends they meet: those at the same junction, the other end of a
    closed line, or only itself at a free end."""
    at = {}
    for k, line in enumerate(result.lines):
        for last, junction in ((False, line.junctions[0]), (True, line.junctions[1])):
            if junction >= 0:
                at.setdefault(junction, set()).add((k, last))
    found = {}
    for k, line in enumerate(result.lines):
        for last, junction in ((False, line.junctions[0]), (True, line.junctions[1])):
            if line.loop:
                found[k, last] = {(k, False), (k, True)}
            else:
                found[k, last] = at[junction] if junction >= 0 else {(k, last)}
    return found


@pytest.mark.parametrize("name", list(SAMPLES))
def test_ovito_reads_them(tmp_path, name):
    crystal, system = SAMPLES[name]()
    result = DislocationAnalysis(crystal).compute(system)
    path = tmp_path / "dislocations.ca"
    result.write(path)
    data = import_file(str(path)).compute()
    box = system.box
    np.testing.assert_array_equal(np.asarray(data.cell[...])[:, :3], box.matrix.T)
    np.testing.assert_array_equal(np.asarray(data.cell[...])[:, 3], box.origin)
    assert tuple(data.cell.pbc) == tuple(box.pbc)

    network = data.dislocations
    assert len(network.segments) == len(result.lines)
    for segment, line in zip(network.segments, result.lines):
        np.testing.assert_array_equal(np.asarray(segment.points), line.points)
        np.testing.assert_allclose(segment.true_burgers_vector, line.burgers, atol=1e-7)
        # OVITO keeps the orientations of the clusters in float32
        np.testing.assert_allclose(
            segment.spatial_burgers_vector, line.spatial_burgers, atol=1e-5
        )
        assert segment.cluster_id == line.cluster + 1
        assert segment.length == pytest.approx(line.length, rel=1e-12)
        assert (segment.is_loop, segment.is_infinite_line) == (line.loop, line.infinite)
    ours = nodes(result)
    for k, segment in enumerate(network.segments):
        for connector in segment.connectors:
            theirs = {(arm.line.id, bool(arm.is_head)) for arm in connector.arms()}
            assert theirs == ours[k, bool(connector.is_head)]

    clusters = result.clusters
    # OVITO's first cluster is that of no cluster, number 0
    assert len(network.clusters) == len(clusters.sizes) + 1
    for c, cluster in enumerate(network.clusters[1:]):
        assert (cluster.id, cluster.structure, cluster.atom_count) == (
            c + 1,
            clusters.structure[c],
            clusters.sizes[c],
        )

    total = data.attributes["DislocationAnalysis.total_line_length"]
    assert total == pytest.approx(result.total_length, rel=1e-12)
    if crystal in ("fcc", "bcc", "cubic_diamond"):
        # OVITO names the families by a vector of each, such as 1/2<011>
        theirs = sorted(
            v
            for k, v in data.attributes.items()
            if k.startswith("DislocationAnalysis.length.") and v > 0
        )
        mine = sorted(v for v in result.lengths().values() if v > 0)
        np.testing.assert_allclose(theirs, mine, rtol=1e-12)

    mesh = data.surfaces["dxa-defect-mesh"]
    faces = np.asarray(mesh.get_face_vertices(), dtype=np.int64).reshape(-1, 3)
    np.testing.assert_array_equal(
        np.asarray(mesh.vertices["Position"]), box.wrap(result.defect_mesh.vertices)[0]
    )
    # OVITO's triangles run the other way round
    np.testing.assert_array_equal(faces, result.defect_mesh.faces[:, [0, 2, 1]])
    if name == "perfect crystal":
        assert not result.lines and not len(faces)
    if name == "free ends":
        assert all(
            line.junctions == (-1, -1) and not line.loop for line in result.lines
        )
    if name == "a loop":
        assert [(line.loop, line.infinite) for line in result.lines] == [(True, False)]


def test_compressed(tmp_path):
    _, system = crossing_loops_and_a_void()
    result = DislocationAnalysis("fcc").compute(system)
    result.write(tmp_path / "plain.ca")
    result.write(tmp_path / "packed.ca.gz")
    plain = import_file(str(tmp_path / "plain.ca")).compute().dislocations
    packed = import_file(str(tmp_path / "packed.ca.gz")).compute().dislocations
    assert len(packed.segments) == len(plain.segments) > 0
    for a, b in zip(plain.segments, packed.segments):
        np.testing.assert_array_equal(np.asarray(a.points), np.asarray(b.points))


def test_the_vectors_of_the_families():
    """The vector written for each family belongs to it."""
    from mdapy.analysis._ca_files import _families
    from mdapy.analysis._dislocations import _family

    for crystal in DislocationAnalysis.NAMES[1:]:
        for _, name, vector, color in _families(crystal):
            assert _family(vector, crystal) == name
            assert color == DislocationAnalysis.FAMILY_COLORS[crystal][name]
