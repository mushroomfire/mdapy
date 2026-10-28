# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Coordination polyhedra: the convex hulls of the neighbors, against the
polyhedra of perfect crystals and those of OVITO."""

import numpy as np
import pytest

import mdapy
from mdapy.analysis import CoordinationPolyhedra
from mdapy.build import build_crystal
from mdapy.errors import InvalidParameter, MissingProperty


def faces_of(result, m):
    """The triangles of polyhedron m as sets of vertex positions."""
    mesh = result.mesh
    triangles = mesh.faces[result.owners == m]
    return {
        tuple(sorted(tuple(np.round(mesh.vertices[v], 6)) for v in t))
        for t in triangles
    }


@pytest.mark.parametrize(
    "structure, a, options, triangles, volume",
    [
        # a cuboctahedron of edge a/√2
        (
            "fcc",
            3.615,
            {"cutoff": 3.0},
            20,
            5 * np.sqrt(2) / 3 * (3.615 / np.sqrt(2)) ** 3,
        ),
        # a cube of edge a
        ("bcc", 2.865, {"num_neighbors": 8}, 12, 2.865**3),
        # rhombic dodecahedron of the 14 first and second neighbors
        ("bcc", 2.865, {"num_neighbors": 14}, 24, 2 * 2.865**3),
    ],
)
def test_polyhedra_of_crystals(structure, a, options, triangles, volume):
    crystal = build_crystal(structure, a, elements="Fe", repeat=(3, 3, 3))
    result = CoordinationPolyhedra(**options).compute(crystal, [0, 5])
    assert result.atoms.tolist() == [0, 5]
    np.testing.assert_allclose(result.volumes, volume, rtol=1e-10)
    for m in range(2):
        assert (result.owners == m).sum() == triangles
    # the first atom at a corner of the box: its polyhedron whole, about it
    corner = result.mesh.vertices[result.mesh.faces[result.owners == 0]]
    assert (corner < -0.1).any()
    # the faces turn outward: the volume by the divergence theorem
    total = (
        np.einsum(
            "fd,fd->f",
            result.mesh.vertices[result.mesh.faces[:, 0]],
            np.cross(
                result.mesh.vertices[result.mesh.faces[:, 1]],
                result.mesh.vertices[result.mesh.faces[:, 2]],
            ),
        ).sum()
        / 6
    )
    assert total == pytest.approx(result.volumes.sum(), rel=1e-10)


def test_octahedra_of_rock_salt_by_bonds():
    salt = build_crystal("rocksalt", 5.64, elements=["Na", "Cl"], repeat=(2, 2, 2))
    pairs = mdapy.NeighborList.build(salt, 3.0)
    i = np.repeat(np.arange(len(salt)), np.diff(pairs.offsets))
    bonds = np.column_stack([i, pairs.j])[i < pairs.j]
    bonded = salt.with_topology(mdapy.Topology(bonds=bonds))
    sodium = np.asarray(salt["element"]) == "Na"
    result = CoordinationPolyhedra(bonds=True).compute(bonded, sodium)
    assert len(result.atoms) == sodium.sum()
    np.testing.assert_allclose(result.volumes, 4 / 3 * (5.64 / 2) ** 3, rtol=1e-10)
    assert len(result.owners) == 8 * sodium.sum()
    by_cutoff = CoordinationPolyhedra(3.0).compute(salt, sodium)
    np.testing.assert_allclose(by_cutoff.volumes, result.volumes, rtol=1e-12)


def test_as_ovito():
    """A shaken copper crystal, so that no four neighbors lie in a plane:
    the same triangles around each atom as OVITO's coordination polyhedra
    on bonds within 3 Å."""
    pytest.importorskip("ovito")
    from ovito.modifiers import (
        CoordinationPolyhedraModifier,
        CreateBondsModifier,
        ExpressionSelectionModifier,
    )
    from ovito.pipeline import Pipeline

    from ovito_reference import ovito_source

    crystal = build_crystal("fcc", 3.615, elements="Cu", repeat=(3, 3, 3))
    rng = np.random.default_rng(2)
    shaken = crystal.with_columns(
        positions=crystal["positions"] + rng.normal(0, 0.05, (len(crystal), 3))
    )
    chosen = [0, 7, 50, 107]
    result = CoordinationPolyhedra(3.0).compute(shaken, chosen)
    pipeline = Pipeline(source=ovito_source(shaken))
    pipeline.modifiers.append(CreateBondsModifier(cutoff=3.0))
    pipeline.modifiers.append(
        ExpressionSelectionModifier(
            expression=" || ".join(f"ParticleIndex == {k}" for k in chosen)
        )
    )
    pipeline.modifiers.append(CoordinationPolyhedraModifier())
    mesh = pipeline.compute().surfaces["coord-polyhedra"]
    theirs = mesh.get_face_vertices()
    region = np.asarray(mesh.faces["Region"])
    particle = np.asarray(mesh.regions["Particle"])
    vertices = np.asarray(mesh.vertices["Position"])
    for m, atom in enumerate(chosen):
        r = int(np.flatnonzero(particle == atom)[0])
        triangles = theirs[region == r]
        expected = {
            tuple(sorted(tuple(np.round(vertices[v], 6)) for v in t)) for t in triangles
        }
        assert faces_of(result, m) == expected


def test_errors():
    crystal = build_crystal("fcc", 3.615, elements="Cu", repeat=(2, 2, 2))
    for options in ({}, {"cutoff": 3.0, "bonds": True}):
        with pytest.raises(InvalidParameter, match="exactly one"):
            CoordinationPolyhedra(**options)
    with pytest.raises(MissingProperty, match="no bonds"):
        CoordinationPolyhedra(bonds=True).compute(crystal)
    with pytest.raises(InvalidParameter, match="mask or indices"):
        CoordinationPolyhedra(3.0).compute(crystal, [99])
    # fewer than four neighbors: no faces and no volume
    line = mdapy.System(
        [[0.0, 0, 0], [2.0, 0, 0], [4.0, 0, 0]],
        mdapy.Box(20.0, pbc=False),
        elements="Ar",
    )
    flat = CoordinationPolyhedra(2.5).compute(line)
    assert len(flat.owners) == 0 and (flat.volumes == 0).all()
    assert repr(CoordinationPolyhedra(3.0)) == "CoordinationPolyhedra(3.0)"
    assert (
        repr(CoordinationPolyhedra(num_neighbors=8))
        == "CoordinationPolyhedra(num_neighbors=8)"
    )
    assert (
        repr(CoordinationPolyhedra(bonds=True)) == "CoordinationPolyhedra(bonds=True)"
    )
