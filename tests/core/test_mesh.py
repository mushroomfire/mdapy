# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import copy

import numpy as np
import pytest

from mdapy import Box, TriangleMesh
from mdapy.errors import InvalidParameter

# a tetrahedron, its faces counterclockwise seen from outside
CORNERS = np.array([[0.0, 0, 0], [2, 0, 0], [0, 3, 0], [0, 0, 4]])
FACES = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]


def test_areas_and_normals_of_a_tetrahedron():
    mesh = TriangleMesh(CORNERS, FACES)
    np.testing.assert_allclose(mesh.areas[:3], [3.0, 4.0, 6.0])
    # the slanted face from the three edges of the right corner
    assert mesh.areas[3] == pytest.approx(0.5 * np.sqrt(6**2 + 8**2 + 12**2))
    assert mesh.area == pytest.approx(mesh.areas.sum())
    # outward: away from the center
    center = CORNERS.mean(axis=0)
    outward = (mesh.corners().mean(axis=1) - center) * mesh.normals
    assert (outward.sum(axis=1) > 0).all()
    np.testing.assert_allclose(np.linalg.norm(mesh.normals, axis=1), 1.0)
    # closed: the area-weighted normals add up to zero
    np.testing.assert_allclose(
        (mesh.areas[:, None] * mesh.normals).sum(axis=0), 0, atol=1e-12
    )


def test_a_triangle_without_area_has_no_normal():
    mesh = TriangleMesh([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[0, 1, 2]])
    assert mesh.areas[0] == 0
    np.testing.assert_array_equal(mesh.normals, [[0, 0, 0]])


def test_shifts_place_corners_in_periodic_images():
    box = Box([[4.0, 0, 0], [1, 5, 0], [0, 0, 6]])
    shifts = [[[0, 0, 0], [1, 0, 0], [0, -1, 1]]]
    mesh = TriangleMesh(CORNERS[:3], [[0, 1, 2]], shifts, box)
    expected = CORNERS[:3] + np.array(shifts[0]) @ box.matrix
    np.testing.assert_allclose(mesh.corners()[0], expected)
    assert mesh.shifts.dtype == np.int8
    assert mesh.box is box


def random_triangles(box, count, rng):
    """Small triangles anywhere around the box, in the image 0."""
    first = rng.uniform(-0.5, 1.5, (count, 3)) @ box.matrix + box.origin
    corners = first[:, None, :] + rng.uniform(-2, 2, (count, 3, 3))
    corners[:, 0] = first
    faces = np.arange(3 * count).reshape(count, 3)
    return corners.reshape(-1, 3), faces


@pytest.mark.parametrize("pbc", [True, (True, False, True)])
def test_pieces_lie_in_the_box_and_keep_area_and_turn(pbc):
    rng = np.random.default_rng(7)
    box = Box([[10.0, 0, 0], [3, 9, 0], [-2, 1, 11]], origin=[1, -2, 0.5], pbc=pbc)
    vertices, faces = random_triangles(box, 400, rng)
    mesh = TriangleMesh(vertices, faces, np.zeros((len(faces), 3, 3)), box)
    pieces, triangle = mesh.pieces()
    assert len(pieces) > len(faces)  # some were cut
    cross = np.cross(pieces[:, 1] - pieces[:, 0], pieces[:, 2] - pieces[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    np.testing.assert_allclose(np.bincount(triangle, area, len(faces)), mesh.areas)
    assert ((cross * mesh.normals[triangle]).sum(axis=1) >= -1e-9).all()
    fractional = (pieces - box.origin) @ box.inverse
    periodic = fractional[:, :, box.pbc]
    assert periodic.min() > -1e-9 and periodic.max() < 1 + 1e-9
    if not box.pbc[1]:  # the open direction is not cut
        assert fractional[:, :, 1].min() < -0.1 and fractional[:, :, 1].max() > 1.1


def test_pieces_without_a_box_are_the_triangles():
    mesh = TriangleMesh(CORNERS, FACES)
    pieces, triangle = mesh.pieces()
    np.testing.assert_array_equal(pieces, mesh.corners())
    np.testing.assert_array_equal(triangle, np.arange(4))


def test_a_triangle_on_a_face_of_the_box_is_not_cut():
    box = Box(10.0)
    mesh = TriangleMesh(
        [[0.0, 1, 1], [0, 3, 1], [0, 1, 3]], [[0, 1, 2]], np.zeros((1, 3, 3)), box
    )
    pieces, triangle = mesh.pieces()
    assert len(pieces) == 1
    np.testing.assert_allclose(pieces[0], mesh.corners()[0])


def test_empty_mesh():
    mesh = TriangleMesh(np.zeros((0, 3)), [], np.zeros((0, 3, 3)), Box(5.0))
    assert len(mesh) == 0 and mesh.area == 0
    assert mesh.faces.shape == (0, 3)
    assert mesh.pieces()[0].shape == (0, 3, 3)


def test_immutable():
    mesh = TriangleMesh(CORNERS, FACES)
    with pytest.raises(AttributeError):
        mesh.faces = None
    with pytest.raises(AttributeError):
        del mesh.faces
    with pytest.raises(ValueError):
        mesh.vertices[0, 0] = 1.0
    assert copy.copy(mesh) is mesh and copy.deepcopy(mesh) is mesh
    assert repr(mesh) == "TriangleMesh(4 vertices, 4 triangles)"
    periodic = TriangleMesh(CORNERS, FACES, np.zeros((4, 3, 3)), Box(9.0))
    assert repr(periodic) == "TriangleMesh(4 vertices, 4 triangles, periodic)"


@pytest.mark.parametrize(
    "arguments, message",
    [
        (([[0, 0, np.nan]], [[0, 0, 0]]), "finite"),
        ((CORNERS, [[0, 1]]), "shape"),
        ((CORNERS, [[0, 1, 4]]), "does not exist"),
        ((CORNERS, [[0, 1, -1]]), "does not exist"),
        ((CORNERS, FACES, np.zeros((4, 3, 3))), "need the box"),
        ((CORNERS, FACES, np.zeros((3, 3, 3)), Box(5.0)), "shape"),
        ((CORNERS, FACES, None, np.eye(3)), "Box"),
    ],
)
def test_invalid(arguments, message):
    with pytest.raises(InvalidParameter, match=message):
        TriangleMesh(*arguments)
