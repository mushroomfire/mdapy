# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import pytest

import mdapy
from mdapy.build import add_graphene, build_crystal, build_polycrystal
from mdapy.build._graphene import _boundaries, _sheet
from mdapy.errors import InvalidParameter, MissingProperty
from mdapy.neighbor import NeighborList


@pytest.fixture(scope="module")
def polycrystal():
    unit = build_crystal("fcc", 3.615, elements="Cu")
    return build_polycrystal(unit, 50.0, 5, seed=1)


def carbon_of(system):
    return system.gather(np.flatnonzero((system.table["element"] == "C").to_numpy()))


def off_boundary(points, seeds, box):
    """Distance of each point from the plane halfway between its two
    nearest seeds, periodic images included."""
    shifts = np.stack(np.meshgrid(*[[-1, 0, 1]] * 3, indexing="ij"), -1).reshape(-1, 3)
    images = (seeds[:, None, :] + (shifts @ box.matrix)[None]).reshape(-1, 3)
    r = np.linalg.norm(points[:, None, :] - images[None], axis=2)
    order = np.argsort(r, axis=1)[:, :2]
    first, second = np.take_along_axis(r, order, 1).T
    gap = np.linalg.norm(images[order[:, 1]] - images[order[:, 0]], axis=1)
    return (second**2 - first**2) / (2 * gap)


def test_one_layer_on_every_boundary(polycrystal):
    result = add_graphene(polycrystal, seed=2)
    carbon = carbon_of(result)
    assert (carbon["grain"] == -1).all() and len(carbon) > 1000
    seeds = polycrystal.info["seeds"]
    np.testing.assert_allclose(
        off_boundary(carbon["positions"], seeds, result.box), 0, atol=1e-6
    )
    # a honeycomb of 1.42 Å bonds, three for most atoms
    nl = NeighborList.build(carbon, 1.6)
    assert nl.distances.min() >= 0.85 * 1.42 - 1e-9
    bonds = np.bincount(nl.i, minlength=len(carbon))
    assert np.mean(bonds == 3) > 0.8
    # the shorter pairs are those of two faces, where they meet
    short = nl.distances[nl.distances < 1.43]
    assert np.mean(np.abs(short - 1.42) < 1e-9) > 0.9
    # no atom of the polycrystal within 3 Å of the carbon
    both = NeighborList.build(result, 3.0)
    is_carbon = (result.table["element"] == "C").to_numpy()
    assert not (is_carbon[both.i] & ~is_carbon[both.j]).any()
    assert len(result) - len(carbon) < len(polycrystal)
    # the same for the same seed
    again = add_graphene(polycrystal, seed=2)
    np.testing.assert_array_equal(again["positions"], result["positions"])
    np.testing.assert_array_equal(result.info["seeds"], polycrystal.info["seeds"])


def test_two_layers_ab_stacked(polycrystal):
    result = add_graphene(polycrystal, layers=2, spacing=3.35, seed=3)
    carbon = carbon_of(result)
    seeds = polycrystal.info["seeds"]
    away = off_boundary(carbon["positions"], seeds, result.box)
    # the two layers 1.675 Å on either side, but near the edges of faces,
    # where other seeds come nearer
    middle = np.abs(np.abs(away) - 1.675) < 1e-6
    assert middle.mean() > 0.7
    # each sheet: its layers on either side of its face, cut to it
    rng = np.random.default_rng(0)
    for corners, normal, _, inner, outer in _boundaries(seeds, result.box):
        center = corners.mean(axis=0)
        np.testing.assert_allclose((corners - center) @ normal, 0, atol=1e-9)
        # the seeds: mirror images across the face
        np.testing.assert_allclose(
            outer - center,
            inner - center - 2 * ((inner - center) @ normal) * normal,
            atol=1e-9,
        )
        sheet, heights = _sheet(corners, normal, 2, 1.42, 3.35, rng)
        np.testing.assert_allclose((sheet - center) @ normal, heights, atol=1e-9)
        np.testing.assert_allclose(np.unique(heights), [-1.675, 1.675])
    # over each atom of one layer, an atom of the other or the middle of a hexagon
    nl = NeighborList.build(carbon, 3.6)
    across = np.abs(nl.distances - 3.35) < 1e-6
    assert across.sum() > 0.3 * len(carbon)


def test_grains_facing_their_own_images():
    # two grains in a cube: each faces its own periodic images too
    unit = build_crystal("fcc", 3.615, elements="Cu")
    seeds = np.array([[5.0, 5.0, 5.0], [15.0, 12.0, 14.0]])
    crystal = build_polycrystal(unit, 20.0, seeds, seed=0)
    faces = _boundaries(seeds, crystal.box)
    assert len(faces) > 6
    result = add_graphene(crystal, seed=1)
    carbon = carbon_of(result)
    np.testing.assert_allclose(
        off_boundary(carbon["positions"], seeds, result.box), 0, atol=1e-6
    )
    # about the area of the boundaries times 2 atoms per hexagon of graphene
    area = sum(face[2] for face in faces)
    density = 2 / (3 * np.sqrt(3) / 2 * 1.42**2)
    assert 0.8 * area * density < len(carbon) < 1.05 * area * density


def walls(points, seeds, box):
    """Distance of each point from the nearest face of its Voronoi cell, and
    from the next nearest."""
    shifts = np.stack(np.meshgrid(*[[-1, 0, 1]] * 3, indexing="ij"), -1).reshape(-1, 3)
    images = (seeds[:, None, :] + (shifts @ box.matrix)[None]).reshape(-1, 3)
    r2 = np.sum((points[:, None, :] - images[None]) ** 2, axis=2)
    first = np.argmin(r2, axis=1)
    gap = np.linalg.norm(images[None] - images[first][:, None], axis=2)
    d = (r2 - r2[np.arange(len(points)), first][:, None]) / (
        2 * np.where(gap > 0, gap, 1)
    )
    d[gap == 0] = np.inf
    d.sort(axis=1)
    return d[:, 0], d[:, 1]


@pytest.mark.parametrize("layers", [1, 2])
def test_template_matching_finds_graphene_and_fcc(polycrystal, layers):
    from mdapy.identify import PolyhedralTemplateMatching

    result = add_graphene(polycrystal, layers=layers, seed=2)
    found = PolyhedralTemplateMatching(("fcc", "hcp", "bcc", "graphene")).compute(
        result
    )
    names = np.array(found.names)[found.structure]
    carbon = (result.table["element"] == "C").to_numpy()
    face, next_face = walls(result["positions"], polycrystal.info["seeds"], result.box)
    # carbon away from the edges of its face, copper away from the carbon
    inside = carbon & (next_face - face > 4.0)
    deep = ~carbon & (face > 3.0 + 1.675 * (layers - 1) + 4.0)
    assert inside.sum() > 500 and deep.sum() > 500
    assert (names[inside] == "graphene").all()
    assert (names[deep] == "fcc").all()
    # the layers of two faces meet without passing through each other
    sheets = NeighborList.build(carbon_of(result), 1.6)
    assert sheets.distances.min() >= 0.85 * 1.42 - 1e-9
    assert np.mean(np.bincount(sheets.i) == 3) > 0.8


def test_coverage(polycrystal):
    full = len(carbon_of(add_graphene(polycrystal, seed=4)))
    half = len(carbon_of(add_graphene(polycrystal, coverage=0.5, seed=4)))
    assert 0.3 * full < half < 0.8 * full
    assert add_graphene(polycrystal, coverage=0.0) is polycrystal
    single = build_polycrystal(
        build_crystal("fcc", 3.615, elements="Cu"), 20.0, 1, seed=1
    )
    assert add_graphene(single) is single


def test_invalid_arguments(polycrystal):
    with pytest.raises(MissingProperty, match="seeds"):
        add_graphene(build_crystal("fcc", 3.615, elements="Cu", repeat=(3, 3, 3)))
    for arguments, message in [
        ({"layers": 0}, "layers must be at least 1"),
        ({"coverage": 1.5}, "coverage must be from 0 to 1"),
        ({"clearance": 0}, "clearance must be positive"),
        ({"bond": -1}, "bond must be positive"),
        ({"spacing": 0}, "spacing must be positive"),
    ]:
        with pytest.raises(InvalidParameter, match=message):
            add_graphene(polycrystal, **arguments)


def test_seeds_in_info():
    unit = build_crystal("fcc", 3.615, elements="Cu")
    seeds = np.array([[5.0, 5.0, 5.0], [15.0, 15.0, 15.0]])
    crystal = build_polycrystal(unit, 20.0, seeds, seed=0)
    np.testing.assert_array_equal(crystal.info["seeds"], seeds)
    assert isinstance(crystal, mdapy.System)
