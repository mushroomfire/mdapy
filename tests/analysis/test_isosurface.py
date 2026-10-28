# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from collections import Counter

import numpy as np
import pytest

import mdapy
from mdapy.analysis import SpatialBinning, isosurface
from mdapy.errors import InvalidParameter
from mdapy.visualization import voxel_faces

TRICLINIC = np.array([[30.0, 0, 0], [6.0, 28.0, 0], [-4.0, 3.0, 32.0]])


def centers(box, n):
    """(na, nb, nc, 3) the centers of the cells of a grid over the box."""
    axes = [(np.arange(k) + 0.5) / k for k in n]
    fractional = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    return fractional @ box.matrix + box.origin


def distance(box, points, center):
    """The distance of points from a center, by the nearest image."""
    step = box.to_fractional(points.reshape(-1, 3)) - box.to_fractional(center[None])
    periodic = box.pbc
    step[:, periodic] -= np.round(step[:, periodic])
    return np.linalg.norm(step @ box.matrix, axis=1).reshape(points.shape[:-1])


def corners(mesh):
    """(F, 3, 3) the corners of the triangles, each triangle whole."""
    return mesh.vertices[mesh.faces] + mesh.shifts @ mesh.box.matrix


def closed(mesh):
    """Whether every edge has two triangles, once each way round."""
    edges = Counter()
    for f, face in enumerate(mesh.faces):
        for k in range(3):
            a, b = face[k], face[(k + 1) % 3]
            edges[(a, b)] += 1
    return all(n == 1 and edges[(b, a)] == 1 for (a, b), n in edges.items())


@pytest.mark.parametrize("where", ["middle", "corner"])
def test_a_sphere(where):
    """A ball of high values in a periodic triclinic box, in the middle or
    about a corner, where its surface crosses every periodic boundary: a
    closed surface of the radius, normals pointing out."""
    box = mdapy.Box(TRICLINIC, origin=[1.0, -2.0, 0.5])
    n = (48, 46, 52)
    center = box.origin + (0.5 if where == "middle" else 0.02) * box.matrix.sum(axis=0)
    radius = 8.0
    field = -distance(box, centers(box, n), center)
    mesh = isosurface(field, box, -radius)
    triangles = corners(mesh)
    r = distance(box, triangles.reshape(-1, 3), center)
    assert np.abs(r - radius).max() < 0.02
    assert mesh.area == pytest.approx(4 * np.pi * radius**2, rel=0.01)
    middle = triangles.mean(axis=1)
    out = middle - center
    out = box.to_fractional(middle) - box.to_fractional(center[None])
    out -= np.round(out)
    assert (np.einsum("ij,ij->i", mesh.normals, out @ box.matrix) > 0).all()
    assert closed(mesh)
    # the longest edge spans no more than a cell
    edges = np.linalg.norm(np.diff(triangles[:, [0, 1, 2, 0]], axis=1), axis=-1)
    assert edges.max() < np.linalg.norm(box.matrix / np.array(n)[:, None], axis=1).sum()
    if where == "corner":
        assert (mesh.shifts != 0).any()


def test_a_plane_ends_at_the_outermost_centers():
    """A field rising along c, which is not periodic: a flat surface at the
    level, exact, its normals toward c below, over the face spanned by the
    centers along a and b, which are periodic."""
    box = mdapy.Box([20.0, 18.0, 30.0], pbc=(True, True, False))
    n = (10, 9, 15)
    field = centers(box, n)[..., 2]
    mesh = isosurface(field, box, 12.3)
    np.testing.assert_allclose(mesh.vertices[:, 2], 12.3, atol=1e-12)
    np.testing.assert_allclose(
        mesh.normals, np.broadcast_to([0, 0, -1.0], mesh.normals.shape), atol=1e-12
    )
    assert mesh.area == pytest.approx(20.0 * 18.0)
    open_box = mdapy.Box([20.0, 18.0, 30.0], pbc=False)
    assert isosurface(field, open_box, 12.3).area == pytest.approx(18.0 * 16.0)


def test_a_left_handed_box():
    """The normals still point to lower values."""
    box = mdapy.Box(
        [[20.0, 0, 0], [0, 18.0, 0], [0, 0, -30.0]], pbc=(True, True, False)
    )
    field = centers(box, (10, 9, 15))[..., 2]
    mesh = isosurface(field, box, -12.3)
    np.testing.assert_allclose(
        mesh.normals, np.broadcast_to([0, 0, -1.0], mesh.normals.shape), atol=1e-12
    )


def test_no_crossing_and_small_grids():
    """No surface where the field does not cross the level, nor on a grid
    of one cell along a direction that is not periodic."""
    box = mdapy.Box(10.0)
    field = np.arange(27.0).reshape(3, 3, 3)
    assert len(isosurface(field, box, 100.0)) == 0
    assert len(isosurface(field, box, -1.0)) == 0
    flat = mdapy.Box(10.0, pbc=(True, True, False))
    assert len(isosurface(field[:, :, :1], flat, 5.0)) == 0
    assert len(isosurface(field, box, 13.5)) > 0


def test_atoms_binned():
    """The number of atoms scattered in a ball, binned along the three box
    vectors: a surface about the ball."""
    rng = np.random.default_rng(4)
    box = mdapy.Box(48.0)
    points = rng.normal(size=(40000, 3))
    points *= (
        14 * rng.random((40000, 1)) ** (1 / 3) / np.linalg.norm(points, axis=1)[:, None]
    )
    ball = mdapy.System(points + 24.0, box).with_columns(one=1.0)
    counts = SpatialBinning("one", 8, axes=(0, 1, 2), reduction="sum").compute(ball)
    mesh = isosurface(counts.values, box, counts.values.max() / 2)
    r = np.linalg.norm(mesh.vertices - 24.0, axis=1)
    assert 10 < r.min() and r.max() < 18


def test_voxel_faces():
    """The six faces of the box in cells, each with the value of the cell
    inside it, the normals pointing out."""
    box = mdapy.Box(TRICLINIC, origin=[1.0, -2.0, 0.5])
    n = (3, 4, 5)
    values = np.arange(60.0).reshape(n)
    mesh, found = voxel_faces(values, box)
    assert len(mesh) == len(found) == 2 * 2 * (3 * 4 + 4 * 5 + 3 * 5)
    m = box.matrix
    faces = [
        np.linalg.norm(np.cross(m[1], m[2])),
        np.linalg.norm(np.cross(m[2], m[0])),
        np.linalg.norm(np.cross(m[0], m[1])),
    ]
    assert mesh.area == pytest.approx(2 * sum(faces))
    middle = mesh.vertices[mesh.faces].mean(axis=1)
    center = box.origin + m.sum(axis=0) / 2
    assert (np.einsum("ij,ij->i", mesh.normals, middle - center) > 0).all()
    # the cell inside each triangle, from its middle moved a little inward
    inner = box.to_fractional(middle - 1e-6 * mesh.normals)
    cell = np.floor(inner * n).astype(int)
    np.testing.assert_array_equal(found, values[tuple(cell.T)])


def test_errors():
    box = mdapy.Box(10.0)
    with pytest.raises(InvalidParameter, match="grid"):
        isosurface(np.zeros((3, 3)), box, 0.0)
    with pytest.raises(InvalidParameter, match="grid"):
        isosurface(np.zeros((3, 0, 3)), box, 0.0)
    with pytest.raises(InvalidParameter, match="finite"):
        isosurface(np.full((3, 3, 3), np.nan), box, 0.0)
    with pytest.raises(InvalidParameter, match="level"):
        isosurface(np.zeros((3, 3, 3)), box, np.inf)
