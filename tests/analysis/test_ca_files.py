# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The CA files of the dislocation analysis: read back, read by OVITO 3.17.1,
and those OVITO writes read here: the same lines, Burgers vectors, clusters,
junctions, defect mesh and lengths."""

import numpy as np
import pytest

import mdapy
from mdapy.analysis import DislocationAnalysis, read_dislocations
from mdapy.build import build_crystal, dislocation_cylinder, dislocation_quadrupole
from mdapy.errors import FileFormatError

from dislocation_models import CU, QUADRUPOLES, crossing_loops, hexagonal, shear_loop


def import_file(path):
    pytest.importorskip("ovito")
    from ovito.io import import_file

    return import_file(path)


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


def ovito_analysis(system, crystal, path):
    """OVITO's dislocation analysis of a system, also written as a CA file."""
    pytest.importorskip("ovito")
    from ovito.data import DataCollection
    from ovito.io import export_file
    from ovito.modifiers import DislocationAnalysisModifier
    from ovito.pipeline import Pipeline, StaticSource

    lattice = DislocationAnalysisModifier.Lattice
    lattices = {
        "fcc": lattice.FCC,
        "hcp": lattice.HCP,
        "bcc": lattice.BCC,
        "cubic_diamond": lattice.CubicDiamond,
        "hexagonal_diamond": lattice.HexagonalDiamond,
    }
    data = DataCollection()
    box = system.box
    data.create_cell(np.column_stack([box.matrix.T, box.origin]), pbc=tuple(box.pbc))
    data.create_particles(count=len(system))
    data.particles_.create_property("Position", data=system["positions"])
    pipeline = Pipeline(source=StaticSource(data=data))
    pipeline.modifiers.append(
        DislocationAnalysisModifier(input_crystal_structure=lattices[crystal])
    )
    export_file(pipeline, str(path), "ca")
    return pipeline.compute()


def same_nodes(found, segments):
    """OVITO's rings of line ends are those of the result."""
    ours = nodes(found)
    for k, segment in enumerate(segments):
        for connector in segment.connectors:
            theirs = {(arm.line.id, bool(arm.is_head)) for arm in connector.arms()}
            assert theirs == ours[k, bool(connector.is_head)]


@pytest.mark.parametrize("name", list(SAMPLES))
def test_read_back(tmp_path, name):
    """What is written reads back the same, but the atoms, which the file
    does not hold."""
    crystal, system = SAMPLES[name]()
    result = DislocationAnalysis(crystal).compute(system)
    result.write(tmp_path / "dislocations.ca.gz")
    back = read_dislocations(tmp_path / "dislocations.ca.gz")
    assert back.crystal == result.crystal
    assert len(back.lines) == len(result.lines)
    for read, line in zip(back.lines, result.lines):
        for name in ("points", "burgers"):
            np.testing.assert_array_equal(getattr(read, name), getattr(line, name))
        # in space from the orientation of the cluster
        np.testing.assert_array_equal(
            read.spatial_burgers,
            result.clusters.orientation[line.cluster] @ line.burgers,
        )
        np.testing.assert_allclose(
            read.spatial_burgers, line.spatial_burgers, atol=1e-9
        )
        for name in ("cluster", "family", "length", "loop", "infinite", "junctions"):
            assert getattr(read, name) == getattr(line, name)
        assert not read.points.flags.writeable
    np.testing.assert_allclose(back.junctions, result.junctions, atol=1e-9)
    for name in ("structure", "sizes", "orientation"):
        np.testing.assert_array_equal(
            getattr(back.clusters, name), getattr(result.clusters, name)
        )
    assert back.lengths() == result.lengths()
    mesh, written = back.defect_mesh, result.defect_mesh
    box = system.box
    np.testing.assert_array_equal(mesh.faces, written.faces)
    np.testing.assert_array_equal(mesh.vertices, box.wrap(written.vertices)[0])
    corners, before = mesh.corners(), written.corners()
    np.testing.assert_allclose(
        corners - corners[:, :1], before - before[:, :1], atol=1e-9
    )
    np.testing.assert_allclose(mesh.areas, written.areas, rtol=1e-12)
    assert mesh.box == box
    assert len(back.structure) == len(back.crystal_cluster) == 0
    assert (back.defect_atoms == -1).all() and len(back.defect_atoms) == len(
        mesh.vertices
    )
    assert back.interface_mesh is None


@pytest.mark.parametrize("name", list(SAMPLES))
def test_reads_the_files_of_ovito(tmp_path, name):
    """OVITO's analysis written by OVITO reads as OVITO has it, the crystal
    that of the most atoms in clusters."""
    crystal, system = SAMPLES[name]()
    data = ovito_analysis(system, crystal, tmp_path / "ovito.ca")
    found = read_dislocations(tmp_path / "ovito.ca")
    assert found.crystal == crystal
    segments = data.dislocations.segments
    assert len(found.lines) == len(segments)
    for segment, line in zip(segments, found.lines):
        # OVITO writes 10 digits
        np.testing.assert_allclose(segment.points, line.points, rtol=1e-9, atol=1e-8)
        np.testing.assert_allclose(segment.true_burgers_vector, line.burgers, atol=1e-9)
        assert segment.cluster_id == line.cluster + 1
        assert (segment.is_loop, segment.is_infinite_line) == (line.loop, line.infinite)
        assert line.length == pytest.approx(segment.length, rel=1e-8)
    same_nodes(found, segments)
    clusters = data.dislocations.clusters[1:]
    assert [c.atom_count for c in clusters] == found.clusters.sizes.tolist()
    assert [c.structure for c in clusters] == found.clusters.structure.tolist()
    mesh = data.surfaces["dxa-defect-mesh"]
    faces = np.asarray(mesh.get_face_vertices(), dtype=np.int64).reshape(-1, 3)
    np.testing.assert_array_equal(found.defect_mesh.faces, faces[:, [0, 2, 1]])
    total = data.attributes["DislocationAnalysis.total_line_length"]
    assert found.total_length == pytest.approx(total, rel=1e-8)


def written(tmp_path):
    _, system = crossing_loops_and_a_void()
    path = tmp_path / "good.ca"
    DislocationAnalysis("fcc").compute(system).write(path)
    return path.read_text()


def test_files_that_do_not_follow_the_format(tmp_path):
    good = written(tmp_path)
    head = good.index("STRUCTURE_TYPES")
    vertices = good.index("DEFECT_MESH_VERTICES")
    junctions = good.index("DISLOCATION_JUNCTIONS")
    cases = [
        ("CA_FILE_VER 6\n", "not a CA file"),
        ("CA_FILE_VERSION 4\nCA_LIB_VERSION 0\n", "version 4 are not read"),
        ("CA_FILE_VERSION 6\nOTHER\n", "expected CA_LIB_VERSION"),
        (good.replace("PBC_FLAGS 1 1 1", "PBC_FLAGS 1 1"), "expected 3 numbers"),
        (good.replace("PBC_FLAGS 1 1 1", "PBC_FLAGS 1 x 1"), "expected 3 numbers"),
        (good.replace("STRUCTURE_TYPE 1", "STRUCTURE 1"), "expected STRUCTURE_TYPE"),
        (good.replace("SIMULATION_CELL_MATRIX", "CELL_MATRIX"), "unknown keyword"),
        (good.replace("CLUSTER 1\n", "CLUSTER 7\n"), "cluster 1, which the file"),
        (
            good.replace("METADATA CRYSTAL fcc", "METADATA CRYSTAL fct"),
            "unknown crystal",
        ),
        (good[: vertices + 30], "unexpected end of file"),
        (
            good[:junctions] + "DISLOCATION_JUNCTIONS\n" + "0 9\n" * 8,
            "junction of a line",
        ),
        (
            good[:junctions] + "DISLOCATION_JUNCTIONS\n" + "0 0\n" * 8,
            "do not close",
        ),
        (
            good[:head] + "STRUCTURE_TYPES 0\nCLUSTERS 0\nDISLOCATIONS 0\n",
            "has no crystal clusters",
        ),
    ]
    # the files of OVITO do not name their crystal
    cases[-1] = (cases[-1][0].replace("METADATA CRYSTAL fcc\n", ""), cases[-1][1])
    for text, message in cases:
        path = tmp_path / "bad.ca"
        path.write_text(text)
        with pytest.raises(FileFormatError, match=message):
            read_dislocations(path)
    points = good.index("\n", good.index("DISLOCATIONS"))
    short = good[: points + 1]
    rest = good[points + 1 :].split("\n")
    rest[3] = "1"  # a line of one point
    (tmp_path / "bad.ca").write_text(short + "\n".join(rest))
    with pytest.raises(FileFormatError, match="of 1 points"):
        read_dislocations(tmp_path / "bad.ca")
    faces = good.index("DEFECT_MESH_FACETS")
    line_end = good.index("\n", faces)
    bad = good[: line_end + 1] + "0 1 99999" + good[good.index("\n", line_end + 1) :]
    (tmp_path / "bad.ca").write_text(bad)
    with pytest.raises(FileFormatError, match="vertex that the file"):
        read_dislocations(tmp_path / "bad.ca")


def test_other_files(tmp_path):
    """Frames after the first, the older layout of the cell, paths, comments
    of no meaning, structures of other names and no junctions: the lines are
    then free at both ends."""
    good = written(tmp_path)
    junctions = good.index("DISLOCATION_JUNCTIONS")
    vertices = good.index("DEFECT_MESH_VERTICES")
    matrix = good.index("SIMULATION_CELL_MATRIX")
    pbc = good.index("PBC_FLAGS")
    rows = [r.split() for r in good[matrix:pbc].split("\n")[1:4]]
    cell = "SIMULATION_CELL " + " ".join(" ".join(r) for r in rows) + "\n"
    text = (
        good[:matrix]
        + cell
        + good[pbc:junctions]
        + good[vertices:]
        + "\nOUTPUT_PATH x.ca\nATOMS_PATH x.dump\nMETADATA SIMULATION_TIMESTEP 3\n"
        + good
    ).replace("NAME fcc\nFULL_NAME FCC", "NAME unknown\nFULL_NAME unknown")
    (tmp_path / "other.ca").write_text(text)
    other = read_dislocations(tmp_path / "other.ca")
    reference = read_dislocations(tmp_path / "good.ca")
    assert len(other.lines) == len(reference.lines) > 0
    assert all(line.junctions == (-1, -1) and not line.loop for line in other.lines)
    assert len(other.junctions) == 0
    np.testing.assert_array_equal(
        other.defect_mesh.box.matrix, reference.defect_mesh.box.matrix
    )
    # an fcc of another name: a structure the result does not know
    assert (other.clusters.structure == 0).all()
