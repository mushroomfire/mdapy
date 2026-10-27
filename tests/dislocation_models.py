# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Dislocation loops for the tests of the dislocation analysis, from the
displacement field of isotropic elasticity's leading term."""

import numpy as np


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
