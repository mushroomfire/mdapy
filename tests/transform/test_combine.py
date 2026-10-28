# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import polars as pl
import pytest

import mdapy
from mdapy.errors import InvalidParameter
from mdapy.transform import combine

BOX = mdapy.Box([[10.0, 0, 0], [1.0, 9.0, 0], [0, 0.5, 11.0]])


def water(count, start, seed=0):
    """count water molecules numbered from start, with ids, charges and
    bonds, in a triclinic box."""
    rng = np.random.default_rng(seed)
    centers = BOX.to_cartesian(rng.random((count, 1, 3)))
    positions = (centers + [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]]).reshape(-1, 3)
    system = mdapy.System(
        positions,
        BOX,
        elements=["O", "H", "H"] * count,
        info={"energy": -4.0, "virial": np.eye(3), "timestep": 7},
        topology=mdapy.Topology(
            bonds=pl.DataFrame(
                {
                    "atoms": [[3 * k, 3 * k + j] for k in range(count) for j in (1, 2)],
                    "type": [1] * (2 * count),
                },
                schema={"atoms": pl.Array(pl.Int64, 2), "type": pl.Int32},
            ),
            angles=[[3 * k + 1, 3 * k, 3 * k + 2] for k in range(count)],
        ),
    )
    return system.with_columns(
        id=np.arange(1, 3 * count + 1) * 2,
        molecule=np.repeat(np.arange(start, start + count), 3),
        charge=np.tile([-0.8, 0.4, 0.4], count),
    )


def test_atoms_columns_and_ids():
    """The atoms of both in order, the box and info of the first without
    its energy and virial; ids numbered on, molecules offset, zeros for a
    column only one has."""
    first = water(3, 1).with_columns(name=list("abcdefghi"), flag=True)
    second = (
        water(2, 1, seed=1)
        .drop("charge")
        .with_columns(velocities=np.ones((6, 3)), tag=np.arange(6))
    )
    second = second.with_columns(molecule=[1, 1, 1, 0, 0, 0])
    both = combine(first, second)
    assert len(both) == 15 and both.box == first.box
    assert both.info == {"timestep": 7}
    np.testing.assert_array_equal(
        both["positions"], np.vstack([first["positions"], second["positions"]])
    )
    assert both["element"].tolist() == ["O", "H", "H"] * 5
    np.testing.assert_array_equal(both["id"], [*first["id"], *range(19, 25)])
    np.testing.assert_array_equal(
        both["molecule"], [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 0, 0, 0]
    )
    np.testing.assert_array_equal(both["charge"][9:], 0.0)
    np.testing.assert_array_equal(both["velocities"][:9], 0.0)
    np.testing.assert_array_equal(both["tag"], [0] * 9 + list(range(6)))
    assert both["name"].tolist() == list("abcdefghi") + [""] * 6
    # columns of other types, such as lists, are left missing
    listed = first.with_table(
        first.table.with_columns(pl.Series("groups", [[1, 2]] * 9))
    )
    assert combine(listed, second).table["groups"].null_count() == 6
    np.testing.assert_array_equal(both["flag"], [True] * 9 + [False] * 6)
    assert both.table.columns[: len(first.table.columns)] == first.table.columns


def test_topology():
    """Bonds and angles of both, those of the second moved to its atoms;
    a topology only one has."""
    first, second = water(3, 1), water(2, 1, seed=1)
    both = combine(first, second)
    bonds = both.topology.bonds["atoms"].to_numpy()
    np.testing.assert_array_equal(bonds[:6], first.topology.bonds["atoms"].to_numpy())
    np.testing.assert_array_equal(
        bonds[6:], second.topology.bonds["atoms"].to_numpy() + 9
    )
    angles = both.topology.angles["atoms"].to_numpy()
    np.testing.assert_array_equal(
        angles[3:], second.topology.angles["atoms"].to_numpy() + 9
    )
    alone = combine(first.with_topology(None), second)
    np.testing.assert_array_equal(
        alone.topology.bonds["atoms"].to_numpy(),
        second.topology.bonds["atoms"].to_numpy() + 9,
    )
    labelled = second.with_topology(
        mdapy.Topology(
            bonds=second.topology.bonds.with_columns(type_label=pl.lit("O-H"))
        )
    )
    mixed = combine(first, labelled).topology.bonds
    assert mixed["type_label"].to_list() == [""] * 6 + ["O-H"] * 4
    assert (
        combine(first.with_topology(None), second.with_topology(None)).topology is None
    )


def test_without_ids_and_empty():
    """Without ids in the first, its atoms get id 0 and the second keeps
    its own; an empty first numbers the added atoms from 1."""
    first = water(2, 1).drop("id")
    second = water(1, 5, seed=2)
    both = combine(first, second)
    np.testing.assert_array_equal(both["id"], [0] * 6 + [2, 4, 6])
    empty = water(1, 1).filter(np.zeros(3, dtype=bool))
    np.testing.assert_array_equal(combine(empty, second)["id"], [1, 2, 3])
    np.testing.assert_array_equal(combine(empty, second)["molecule"], [5, 5, 5])
    bare = mdapy.System(np.zeros((2, 3)), BOX)
    assert combine(bare, second)["element"].tolist() == ["X", "X", "O", "H", "H"]
    assert combine(second, bare)["element"].tolist() == ["O", "H", "H", "X", "X"]


def test_mixed_types():
    """Integers joined with floats; text with numbers cannot be."""
    first = water(1, 1).with_columns(score=[1, 2, 3])
    second = water(1, 1).with_columns(score=[0.5, 1.5, 2.5])
    np.testing.assert_allclose(
        combine(first, second)["score"], [1, 2, 3, 0.5, 1.5, 2.5]
    )
    with pytest.raises(InvalidParameter, match="column 'score'"):
        combine(first, second.with_columns(score=["a", "b", "c"]))


def test_as_ovito(tmp_path):
    """OVITO's combine datasets of two LAMMPS data files: the same atoms,
    ids, molecules, charges, velocities, elements, bonds and angles; OVITO
    also offsets molecule 0, which mdapy keeps."""
    pytest.importorskip("ovito")
    from ovito.io import import_file
    from ovito.modifiers import CombineDatasetsModifier

    first = water(3, 1)
    second = water(2, 1, seed=1).with_columns(
        molecule=[2, 2, 2, 0, 0, 0], velocities=np.arange(18.0).reshape(6, 3)
    )
    mdapy.write(tmp_path / "first.data", first, atom_style="full")
    mdapy.write(tmp_path / "second.data", second, atom_style="full")
    ours = combine(
        mdapy.read(tmp_path / "first.data"), mdapy.read(tmp_path / "second.data")
    )
    pipeline = import_file(str(tmp_path / "first.data"), atom_style="full")
    modifier = CombineDatasetsModifier()
    modifier.source.load(str(tmp_path / "second.data"), atom_style="full")
    pipeline.modifiers.append(modifier)
    data = pipeline.compute()
    particles = data.particles
    np.testing.assert_allclose(ours["positions"], particles.positions, atol=1e-9)
    np.testing.assert_array_equal(ours["id"], particles["Particle Identifier"])
    molecules = np.asarray(particles["Molecule Identifier"])
    np.testing.assert_array_equal(ours["molecule"][:12], molecules[:12])
    assert (ours["molecule"][12:] == 0).all() and (molecules[12:] == 3).all()
    np.testing.assert_allclose(ours["charge"], particles["Charge"])
    np.testing.assert_allclose(ours["velocities"], particles["Velocity"])
    types = particles["Particle Type"]
    names = [types.type_by_id(t).name for t in np.asarray(types)]
    assert ours["element"].tolist() == names
    np.testing.assert_array_equal(
        ours.topology.bonds["atoms"].to_numpy(), particles.bonds.topology
    )
    np.testing.assert_array_equal(
        ours.topology.angles["atoms"].to_numpy(), particles.angles["Topology"]
    )
    np.testing.assert_array_equal(np.asarray(data.cell)[:, :3], first.box.matrix.T)
