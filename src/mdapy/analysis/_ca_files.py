# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The crystal analysis (CA) files of OVITO, version 6: the dislocation lines,
the crystal clusters of their Burgers vectors and the defect mesh."""

import os
from typing import TYPE_CHECKING

import numpy as np

from mdapy.io._files import open_output

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
    lines = ["CA_FILE_VERSION 6", "CA_LIB_VERSION 0.0.0"]
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
