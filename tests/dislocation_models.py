# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Dislocations for the tests of the dislocation analysis and the elastic
strain: loops from the displacement field of isotropic elasticity's
leading term, quadrupoles, and straight dislocations of hexagonal crystals."""

import numpy as np

import mdapy
from mdapy.build import build_crystal, insert_dislocation


def solid_angle(positions, vertices):
    """The signed solid angle a flat polygon subtends at each position: the
    sum over the triangles from its center (Van Oosterom and Strackee)."""
    center = vertices.mean(axis=0)
    total = np.zeros(len(positions))
    for k in range(len(vertices)):
        r = [p - positions for p in (center, vertices[k], vertices[k - 1])]
        n = [np.linalg.norm(v, axis=1) for v in r]

        def dot(u, v):
            return np.einsum("ij,ij->i", u, v)

        numerator = dot(r[0], np.cross(r[2], r[1]))
        denominator = (
            n[0] * n[1] * n[2]
            + dot(r[0], r[1]) * n[2]
            + dot(r[0], r[2]) * n[1]
            + dot(r[1], r[2]) * n[0]
        )
        total += 2 * np.arctan2(numerator, denominator)
    return total


def shear_loop(system, burgers, center, radius, normal=(0, 0, 1), sides=64):
    """The system with a dislocation loop of the given Burgers vector (Å) on
    the plane through center normal to normal, its atoms displaced by
    b Ω / 4π, Ω the solid angle of the loop; the loop runs counterclockwise
    about the normal."""
    normal = np.asarray(normal, dtype=float) / np.linalg.norm(normal)
    u = np.cross(normal, [1.0, 0, 0] if abs(normal[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    t = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    vertices = np.asarray(center, dtype=float) + radius * (
        np.outer(np.cos(t), u) + np.outer(np.sin(t), v)
    )
    omega = solid_angle(system["positions"], vertices)
    burgers = np.asarray(burgers, dtype=float)
    return system.with_columns(
        positions=system["positions"] + np.outer(omega / (4 * np.pi), burgers)
    )


CU = (168.4, 121.4, 75.4)
FE = (243.3, 145.0, 116.1)
SI = (165.7, 63.9, 79.6)
PLANE_FCC = [(1, -1, 0), (1, 1, -2), (1, 1, 1)]

QUADRUPOLES = {
    "bcc screw": ("bcc", ("bcc screw", 2.865, FE, (6, 10, 5)), {"elements": "Fe"}),
    "bcc edge": ("bcc", ("bcc edge", 2.865, FE, (16, 10, 2)), {"elements": "Fe"}),
    "fcc screw": ("fcc", ("fcc screw", 3.615, CU, (6, 10, 4)), {"elements": "Cu"}),
    "fcc edge, dissociated": (
        "fcc",
        ("fcc edge", 3.615, CU, (14, 8, 3)),
        {"elements": "Cu", "dissociation": 10},
    ),
    "diamond screw": (
        "cubic_diamond",
        ("diamond screw", 5.431, SI, (6, 10, 3)),
        {"elements": "Si"},
    ),
    "diamond 60": (
        "cubic_diamond",
        ("diamond 60", 5.431, SI, (8, 6, 3)),
        {"elements": "Si"},
    ),
    "diamond 60, dissociated": (
        "cubic_diamond",
        ("diamond 60", 5.431, SI, (8, 6, 3)),
        {"elements": "Si", "dissociation": 12},
    ),
}


def cubic_loops():
    cu = build_crystal(
        "fcc", 3.615, elements="Cu", orientation=PLANE_FCC, repeat=(12, 7, 6)
    )
    center = cu.box.matrix.diagonal() / 2 + [0.1, 0.1, 0.05]
    fe = build_crystal(
        "bcc",
        2.87,
        elements="Fe",
        orientation=[(1, 1, -2), (-1, 1, 0), (1, 1, 1)],
        repeat=(8, 11, 6),
    )
    center_fe = fe.box.matrix.diagonal() / 2 + [0.1, 0.05, 0.1]
    a = 3.615
    return {
        "perfect fcc loop": (
            "fcc",
            shear_loop(cu, [a / np.sqrt(2), 0, 0], center, 14.0),
        ),
        "shockley loop": ("fcc", shear_loop(cu, [0, -a / np.sqrt(6), 0], center, 14.0)),
        "bcc loop": (
            "bcc",
            shear_loop(
                fe, [0, 0, 2.87 * np.sqrt(3) / 2], center_fe, 11.0, normal=[0, 1, 0]
            ),
        ),
    }


def crossing_loops():
    """Two loops on crossing {111} planes of copper whose Burgers vectors
    add up to a perfect one: a glissile junction where they meet."""
    a = 3.615
    cu = build_crystal("fcc", a, elements="Cu", repeat=(12, 12, 12))
    center = cu.box.matrix.diagonal() / 2 + [0.11, 0.07, 0.03]
    first = shear_loop(cu, a / 2 * np.array([1, -1, 0]), center, 13.0, normal=[1, 1, 1])
    return shear_loop(
        first, a / 2 * np.array([0, 1, 1]), center, 13.0, normal=[1, 1, -1]
    )


def hexagonal(structure, a, element, kind):
    """A straight basal dislocation of a hexagonal crystal in a box open
    across the line: a screw along a1 or an edge of Burgers vector a1."""
    if kind == "screw":
        orientation, repeat, burgers = (
            [(0, 1, -1, 0), (0, 0, 0, 1), (2, -1, -1, 0)],
            (9, 5, 4),
            [0, 0, a],
        )
    else:
        orientation, repeat, burgers = (
            [(2, -1, -1, 0), (0, 0, 0, 1), (0, 1, -1, 0)],
            (14, 5, 4),
            [a, 0, 0],
        )
    cell = build_crystal(
        structure, a, elements=element, orientation=orientation, repeat=repeat
    )
    cell = cell.with_box(mdapy.Box(cell.box.matrix, pbc=(False, False, True)))
    return insert_dislocation(
        cell, burgers, center=cell.box.matrix.diagonal() / 2 + [0.3, 0.2, 0]
    )
