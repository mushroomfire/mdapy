# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The crystal analysis (CA) files of OVITO, version 6: the dislocation lines,
the crystal clusters of their Burgers vectors and the defect mesh."""

import os
from typing import TYPE_CHECKING

import numpy as np

from mdapy.core.box import Box
from mdapy.core.mesh import TriangleMesh
from mdapy.io._files import Source, open_output
from mdapy.io._text import TextReader

if TYPE_CHECKING:
    from mdapy.analysis._dislocations import DislocationResult

# OVITO's short and full names of the structures, whose numbers are the
# indices of the names of the result
_OVITO_NAMES = {
    "fcc": ("fcc", "FCC"),
    "hcp": ("hcp", "HCP"),
    "bcc": ("bcc", "BCC"),
    "cubic_diamond": ("diamond", "Cubic diamond"),
    "hexagonal_diamond": ("hex_diamond", "Hexagonal diamond"),
}


def _number(x: float) -> str:
    """A float64 in the fewest digits that read back the same."""
    return repr(float(x))


def _row(values) -> str:
    return " ".join(_number(x) for x in values)


def _families(name: str) -> list[tuple[int, str, np.ndarray, tuple]]:
    """The families of Burgers vectors of a structure: OVITO's number of
    each, its name, a vector of it in the lattice frame and its color."""
    from mdapy.analysis._dislocations import (
        _A,
        _C,
        _CUBIC_FAMILIES,
        DislocationAnalysis,
    )

    hexagonal = {
        "1/3<11-20>": (_A, 0.0, 0.0),
        "<0001>": (0.0, 0.0, _C),
        "<1-100>": (0.0, np.sqrt(3) * _A, 0.0),
        "1/3<1-100>": (0.0, _A / np.sqrt(3), 0.0),
        "1/3<11-23>": (_A, 0.0, _C),
    }
    vectors = _CUBIC_FAMILIES.get(name, hexagonal)
    colors = DislocationAnalysis.FAMILY_COLORS[name]
    base = 10 * (DislocationAnalysis.NAMES.index(name) - 1)
    named = [family for family in colors if family != "other"]
    return [
        (
            base + k + 1,
            family,
            np.array(vectors[family], dtype=np.float64),
            colors[family],
        )
        for k, family in enumerate(named)
    ]


def _junction_rings(result: "DislocationResult") -> list[tuple[int, int]]:
    """The node each node of the lines joins, as OVITO's junction rings: for
    line k, its node at the last point (OVITO's forward node, 0) and at the
    first (backward, 1), each as (forward or not, line)."""
    count = len(result.lines)
    rings = [(0, 0)] * (2 * count)
    meeting: dict[int, list[int]] = {}
    for k, line in enumerate(result.lines):
        forward, backward = 2 * k, 2 * k + 1
        if line.loop:
            rings[forward], rings[backward] = (0, k), (1, k)
            continue
        first, last = line.junctions
        for node, junction in ((forward, last), (backward, first)):
            if junction < 0:
                rings[node] = (int(node == forward), k)  # a free end: itself
            else:
                meeting.setdefault(junction, []).append(node)
    for nodes in meeting.values():
        for node, after in zip(nodes, nodes[1:] + nodes[:1]):
            rings[node] = (int(after % 2 == 0), after // 2)
    return rings


def _mesh_lines(result: "DislocationResult") -> list[str]:
    """The defect mesh as OVITO keeps it: vertices in the box, triangles
    the other way round, and the triangle beyond each edge."""
    mesh = result.defect_mesh
    box = mesh.box
    vertices = box.wrap(mesh.vertices)[0]
    # OVITO's triangles run the other way: corners 0, 2, 1
    faces = mesh.faces[:, [0, 2, 1]]
    shifts = mesh.shifts[:, [0, 2, 1]].astype(np.int64)
    edges = {}
    for f in range(len(faces)):
        for c in range(3):
            a, b = c, (c + 1) % 3
            key = (faces[f, a], faces[f, b], *(shifts[f, b] - shifts[f, a]))
            edges[key] = f
    beyond = np.empty_like(faces)
    for f in range(len(faces)):
        for c in range(3):
            a, b = c, (c + 1) % 3
            back = (faces[f, b], faces[f, a], *(shifts[f, a] - shifts[f, b]))
            beyond[f, c] = edges[back]
    lines = [f"DEFECT_MESH_VERTICES {len(vertices)}"]
    lines += [_row(v) for v in vertices.tolist()]
    lines.append(f"DEFECT_MESH_FACETS {len(faces)}")
    lines += [" ".join(map(str, f)) for f in faces.tolist()]
    lines += [" ".join(map(str, f)) for f in beyond.tolist()]
    return lines


def write_ca(result: "DislocationResult", path: str | os.PathLike) -> None:
    """Writes a result of the dislocation analysis as a CA file."""
    box = result.defect_mesh.box
    # OVITO passes over METADATA lines; reading takes the crystal from it
    lines = [
        "CA_FILE_VERSION 6",
        "CA_LIB_VERSION 0.0.0",
        f"METADATA CRYSTAL {result.crystal}",
    ]
    structures = result.names[1:]
    lines.append(f"STRUCTURE_TYPES {len(structures)}")
    for number, name in enumerate(structures, start=1):
        short, full = _OVITO_NAMES[name]
        lines += [
            f"STRUCTURE_TYPE {number}",
            f"NAME {short}",
            f"FULL_NAME {full}",
            f"COLOR {_row(result.colors[number])}",
            "TYPE LATTICE",
        ]
        families = _families(name)
        lines.append(f"BURGERS_VECTOR_FAMILIES {len(families)}")
        for family_id, family, vector, color in families:
            lines += [
                f"BURGERS_VECTOR_FAMILY ID {family_id}",
                family,
                _row(vector),
                _row(color),
            ]
        lines.append("END_STRUCTURE_TYPE")
    lines.append(f"SIMULATION_CELL_ORIGIN {_row(box.origin)}")
    lines.append("SIMULATION_CELL_MATRIX")
    lines += [_row(row) for row in box.matrix.T.tolist()]
    lines.append("PBC_FLAGS " + " ".join(str(int(p)) for p in box.pbc))
    clusters = result.clusters
    lines.append(f"CLUSTERS {len(clusters.sizes)}")
    for c in range(len(clusters.sizes)):
        lines += [
            f"CLUSTER {c + 1}",
            f"CLUSTER_STRUCTURE {clusters.structure[c]}",
            "CLUSTER_ORIENTATION",
            *(_row(row) for row in clusters.orientation[c].tolist()),
            f"CLUSTER_SIZE {clusters.sizes[c]}",
            "END_CLUSTER",
        ]
    lines.append("CLUSTER_TRANSITIONS 0")
    lines.append(f"DISLOCATIONS {len(result.lines)}")
    for k, line in enumerate(result.lines):
        lines += [str(k), _row(line.burgers), str(line.cluster + 1)]
        lines.append(str(len(line.points)))
        lines += [_row(p) for p in line.points.tolist()]
    lines.append("DISLOCATION_JUNCTIONS")
    lines += [f"{forward} {k}" for forward, k in _junction_rings(result)]
    lines += _mesh_lines(result)
    with open_output(path) as f:
        f.write(("\n".join(lines) + "\n").encode())


# --------------------------------------------------------------- reading

# the structures by their short and full names in OVITO's files, in any case
_ALIASES = {
    alias.lower(): name
    for name, aliases in {
        "fcc": ("fcc", "FCC", "Face-centered cubic"),
        "hcp": ("hcp", "HCP", "Hexagonal close-packed"),
        "bcc": ("bcc", "BCC", "Body-centered cubic"),
        "cubic_diamond": ("diamond", "Cubic diamond", "cubic_diamond"),
        "hexagonal_diamond": ("hex_diamond", "Hexagonal diamond", "hexagonal_diamond"),
    }.items()
    for alias in aliases
}


def _values(reader: TextReader, text: str, count: int, kind: type = float) -> list:
    """The first count numbers of a line, or an error at it."""
    try:
        values = [kind(v) for v in text.split()[:count]]
    except ValueError:
        values = []
    if len(values) != count:
        raise reader.error(f"expected {count} numbers, not {text!r}")
    return values


def _next(reader: TextReader, count: int, kind: type = float) -> list:
    return _values(reader, reader.readline().strip(), count, kind)


def _keyword(reader: TextReader, keyword: str) -> int:
    """The integer after the keyword that starts the next line."""
    text = reader.readline().strip()
    if not text.startswith(keyword + " "):
        raise reader.error(f"expected {keyword}, not {text!r}")
    return _values(reader, text[len(keyword) :], 1, int)[0]


class _Contents:
    """What a CA file holds, as read."""

    def __init__(self) -> None:
        self.crystal: str | None = None
        self.structures: dict[int, int] = {}  # the file's number: index of NAMES
        self.matrix = np.zeros((3, 3))
        self.origin = np.zeros(3)
        self.pbc = np.ones(3, dtype=bool)
        self.clusters: dict[int, int] = {}  # the file's number: index
        self.cluster_structure: list[int] = []
        self.sizes: list[int] = []
        self.orientation: list[list[list[float]]] = []
        self.lines: list[tuple[np.ndarray, int, np.ndarray]] = []
        self.rings: list[tuple[int, int]] | None = None
        self.vertices = np.zeros((0, 3))
        self.faces = np.zeros((0, 3), dtype=np.int64)


def _structure_types(reader: TextReader, count: int, found: _Contents) -> None:
    from mdapy.analysis._dislocations import DislocationAnalysis

    for _ in range(count):
        number = _keyword(reader, "STRUCTURE_TYPE")
        name = 0
        while True:
            text = reader.readline().strip()
            key, _, rest = text.partition(" ")
            if key in ("END_STRUCTURE_TYPE", "END_PATTERN"):
                break
            if key in ("NAME", "FULL_NAME") and rest.strip().lower() in _ALIASES:
                name = DislocationAnalysis.NAMES.index(_ALIASES[rest.strip().lower()])
            elif key == "BURGERS_VECTOR_FAMILIES":
                reader.skip(4 * _values(reader, rest, 1, int)[0])
        found.structures[number] = name


def _clusters(reader: TextReader, count: int, found: _Contents) -> None:
    for index in range(count):
        number, structure, size = 0, 0, 0
        orientation = np.eye(3).tolist()
        while True:
            text = reader.readline().strip()
            key, _, rest = text.partition(" ")
            if key == "END_CLUSTER":
                break
            if key == "CLUSTER":
                number = _values(reader, rest, 1, int)[0]
            elif key == "CLUSTER_STRUCTURE":
                structure = _values(reader, rest, 1, int)[0]
            elif key == "CLUSTER_SIZE":
                size = _values(reader, rest, 1, int)[0]
            elif key == "CLUSTER_ORIENTATION":
                orientation = [_next(reader, 3) for _ in range(3)]
        found.clusters[number] = index
        found.cluster_structure.append(found.structures.get(structure, 0))
        found.sizes.append(size)
        found.orientation.append(orientation)


def _dislocations(reader: TextReader, count: int, found: _Contents) -> None:
    for _ in range(count):
        _next(reader, 1, int)  # the number of the line, its place
        burgers = np.array(_next(reader, 3))
        number = _next(reader, 1, int)[0]
        if number not in found.clusters:
            raise reader.error(f"a line in cluster {number}, which the file has not")
        points = _next(reader, 1, int)[0]
        if points < 2:
            raise reader.error(f"a line of {points} points; it needs 2 or more")
        # the points, perhaps with the size of the core after them
        p = np.array([_next(reader, 3) for _ in range(points)])
        found.lines.append((burgers, found.clusters[number], p))


def _parse(reader: TextReader) -> _Contents:
    first = reader.readline().strip()
    if not first.startswith("CA_FILE_VERSION "):
        raise reader.error("not a CA file: it starts without CA_FILE_VERSION")
    version = _values(reader, first[16:], 1, int)[0]
    if version not in (5, 6, 7):
        raise reader.error(f"CA files of version {version} are not read; 5 to 7 are")
    if not reader.readline().startswith("CA_LIB_VERSION"):
        raise reader.error("expected CA_LIB_VERSION")
    found = _Contents()
    while not reader.at_end():
        text = reader.readline().strip()
        key, _, rest = text.partition(" ")
        if not key or key in ("OUTPUT_PATH", "ATOMS_PATH"):
            continue
        if key in ("STRUCTURE_TYPES", "STRUCTURE_PATTERNS"):
            _structure_types(reader, _values(reader, rest, 1, int)[0], found)
        elif key == "SIMULATION_CELL_ORIGIN":
            found.origin = np.array(_values(reader, rest, 3))
        elif key == "SIMULATION_CELL_MATRIX":
            found.matrix = np.array([_next(reader, 3) for _ in range(3)]).T
        elif key == "SIMULATION_CELL":
            found.matrix = np.array(_values(reader, rest, 9)).reshape(3, 3).T
        elif key == "PBC_FLAGS":
            found.pbc = np.array(_values(reader, rest, 3, int)) != 0
        elif key == "CLUSTERS":
            _clusters(reader, _values(reader, rest, 1, int)[0], found)
        elif key == "CLUSTER_TRANSITIONS":
            reader.skip(2 * _values(reader, rest, 1, int)[0])
        elif key == "DISLOCATIONS":
            _dislocations(reader, _values(reader, rest, 1, int)[0], found)
        elif key == "DISLOCATION_JUNCTIONS":
            found.rings = [
                tuple(_next(reader, 2, int)) for _ in range(2 * len(found.lines))
            ]
            if any(not 0 <= k < len(found.lines) for _, k in found.rings):
                raise reader.error("a junction of a line that the file has not")
        elif key == "DEFECT_MESH_VERTICES":
            count = _values(reader, rest, 1, int)[0]
            found.vertices = reader.table(count, [("f", 3)])[0].reshape(-1, 3)
        elif key == "DEFECT_MESH_FACETS":
            count = _values(reader, rest, 1, int)[0]
            found.faces = reader.table(count, [("i", 3)])[0].reshape(-1, 3)
            if len(found.faces) and found.faces.max() >= len(found.vertices):
                raise reader.error("a triangle of a vertex that the file has not")
            reader.skip(count)  # the triangles beyond the edges
        elif key == "METADATA":
            tag, _, value = rest.partition(" ")
            if tag == "CRYSTAL":
                found.crystal = value.strip()
        elif key == "CA_FILE_VERSION":  # the next frame
            break
        else:
            raise reader.error(f"unknown keyword {key!r}")
    return found


def _ends(found: _Contents, reader: TextReader) -> tuple[list, list, list]:
    """Whether each line is closed, the junctions at its first and last
    point, and the point of each junction, from the rings of its ends: end
    2k of line k at its last point, 2k + 1 at its first."""
    count = len(found.lines)
    rings = found.rings or [(int(n % 2 == 0), n // 2) for n in range(2 * count)]
    after = [2 * k + (0 if forward else 1) for forward, k in rings]
    closed, ends, points = [False] * count, [[-1, -1] for _ in range(count)], []
    seen = [False] * (2 * count)
    for start in (n for k in range(count) for n in (2 * k + 1, 2 * k)):
        if seen[start]:
            continue
        cycle, node = [], start
        while not seen[node]:
            seen[node] = True
            cycle.append(node)
            node = after[node]
        if node != start:
            raise reader.error("the junctions of the lines do not close")
        k = start // 2
        if sorted(cycle) == [2 * k, 2 * k + 1]:
            closed[k] = True
        elif len(cycle) > 1:
            for node in cycle:
                ends[node // 2][1 if node % 2 == 0 else 0] = len(points)
            line = found.lines[start // 2][2]
            points.append(line[-1] if start % 2 == 0 else line[0])
    return closed, ends, points


def read_ca(path: str | os.PathLike) -> "DislocationResult":
    """Reads the first frame of a CA file as a result of the dislocation
    analysis."""
    from mdapy.analysis._dislocations import (
        CrystalClusters,
        DislocationAnalysis,
        DislocationLine,
        DislocationResult,
        _family,
    )

    with Source(path) as source:
        reader = source.reader()
        found = _parse(reader)
        names = DislocationAnalysis.NAMES
        crystal = found.crystal
        if crystal is None:
            # as OVITO: the structure of the most atoms in clusters
            atoms = np.zeros(len(names), dtype=np.int64)
            np.add.at(atoms, found.cluster_structure, found.sizes)
            atoms[0] = 0
            if not atoms.any():
                raise reader.error(
                    "the file tells no crystal: it has no crystal clusters"
                )
            crystal = names[int(np.argmax(atoms))]
        elif crystal not in names[1:]:
            raise reader.error(f"unknown crystal {crystal!r}")
        closed, ends, junction_points = _ends(found, reader)
    box = Box(found.matrix, origin=found.origin, pbc=found.pbc)
    orientation = np.array(found.orientation, dtype=np.float64).reshape(-1, 3, 3)
    lines = []
    for k, (burgers, cluster, points) in enumerate(found.lines):
        spatial = orientation[cluster] @ burgers
        for array in (burgers, spatial, points):
            array.flags.writeable = False
        lines.append(
            DislocationLine(
                points=points,
                burgers=burgers,
                spatial_burgers=spatial,
                cluster=cluster,
                family=_family(burgers, crystal),
                length=float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()),
                loop=closed[k],
                infinite=closed[k]
                and not np.allclose(points[0], points[-1], atol=1e-4),
                junctions=(ends[k][0], ends[k][1]),
            )
        )
    junctions = np.array(junction_points, dtype=np.float64).reshape(-1, 3)
    if len(junctions):
        junctions = box.wrap(junctions)[0]
    # MDAPY's triangles run the other way round, their corners as near the
    # first as the periodic images allow
    faces = found.faces[:, [0, 2, 1]]
    corners = found.vertices[faces]
    fraction = (corners - corners[:, :1]) @ box.inverse
    shifts = -np.where(box.pbc, np.round(fraction), 0.0).astype(np.int32)
    return DislocationResult(
        crystal=crystal,
        lines=tuple(lines),
        junctions=junctions,
        structure=np.zeros(0, dtype=np.int8),
        names=names,
        colors=DislocationAnalysis.COLORS,
        crystal_cluster=np.zeros(0, dtype=np.int64),
        clusters=CrystalClusters(
            structure=np.array(found.cluster_structure, dtype=np.int8),
            sizes=np.array(found.sizes, dtype=np.int64),
            orientation=orientation,
        ),
        defect_mesh=TriangleMesh(found.vertices, faces, shifts, box),
        defect_atoms=np.full(len(found.vertices), -1, dtype=np.int64),
    )
