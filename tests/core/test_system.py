# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

import numpy as np
import polars as pl
import pytest

from mdapy import Box, System, Topology
from mdapy.elements import ELEMENT_DTYPE
from mdapy.errors import InvalidParameter


def water_dimer():
    """Two water molecules, O-H 0.96 Å, with bonds and angles."""
    positions = [
        [0.000, 0.000, 0.000],
        [0.960, 0.000, 0.000],
        [-0.240, 0.930, 0.000],
        [3.000, 0.000, 0.000],
        [3.960, 0.000, 0.000],
        [2.760, 0.930, 0.000],
    ]
    topology = Topology(
        bonds=[[0, 1], [0, 2], [3, 4], [3, 5]],
        angles=[[1, 0, 2], [4, 3, 5]],
    )
    return System(
        positions,
        Box([10.0, 10.0, 10.0]),
        elements=["O", "H", "H"] * 2,
        types=[1, 2, 2] * 2,
        info={"energy": -27.5, "virial": np.eye(3)},
        topology=topology,
    )


# --- construction ----------------------------------------------------------------


def test_construction_from_positions():
    s = water_dimer()
    assert len(s) == 6
    assert s.table.schema["positions"] == pl.Array(pl.Float64, 3)
    assert s.table.schema["element"] == ELEMENT_DTYPE
    assert s.table.schema["type"] == pl.Int32
    assert s.table["element"].to_list() == ["O", "H", "H"] * 2
    assert s.info["energy"] == -27.5


def test_construction_from_table_normalizes_standard_columns():
    table = pl.DataFrame(
        {
            "positions": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],  # list, not array
            "velocities": np.ones((2, 3)),
            "element": ["Cu", "Ni"],  # string, not enum
            "type": [1, 2],  # int64
            "id": np.array([1, 2], dtype=np.int32),
            "image": [[0, 0, 0], [1, -1, 0]],
            "charge": [0, 1],
            "my_property": [3.5, 4.5],
        }
    )
    s = System(table=table, box=4.0)
    schema = s.table.schema
    assert schema["positions"] == pl.Array(pl.Float64, 3)
    assert schema["velocities"] == pl.Array(pl.Float64, 3)
    assert schema["element"] == ELEMENT_DTYPE
    assert schema["type"] == pl.Int32
    assert schema["id"] == pl.Int64
    assert schema["image"] == pl.Array(pl.Int32, 3)
    assert schema["charge"] == pl.Float64
    assert schema["my_property"] == pl.Float64
    assert s.box == Box(4.0)


def test_whole_number_floats_are_accepted_for_integer_columns():
    table = pl.DataFrame(
        {"positions": [[0.0, 0, 0]], "type": [2.0], "image": [[1.0, -1.0, 0.0]]}
    )
    s = System(table=table)
    assert s.table["type"].to_list() == [2]
    assert s.table["image"].to_list() == [[1, -1, 0]]


def test_scalar_elements_and_types_apply_to_all_atoms():
    s = System(np.zeros((4, 3)), elements="Al", types=3)
    assert s.table["element"].to_list() == ["Al"] * 4
    assert s.table["type"].to_list() == [3] * 4


def test_empty_system():
    s = System(np.empty((0, 3)), box=5.0, elements=[])
    assert len(s) == 0
    assert s.positions.shape == (0, 3)
    assert len(s.select([])) == 0


def test_default_box_encloses_the_atoms():
    s = System([[0, 0, 0], [1, 2, 3]])
    assert not s.box.pbc.any()
    frac = s.box.to_fractional(s.positions)
    assert np.all((frac > 0) & (frac < 1))


def test_box_from_array():
    s = System(np.zeros((1, 3)), [[2, 0, 0], [0, 3, 0], [0, 0, 4]])
    assert s.box == Box([2, 3, 4])


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({}, "either positions or table"),
        ({"positions": np.zeros((2, 3)), "table": pl.DataFrame()}, "either"),
        ({"positions": np.zeros((2, 2))}, r"shape \(N, 3\)"),
        ({"positions": [[0, 0, np.nan]]}, "finite"),
        ({"table": {"positions": [[0, 0, 0]]}}, "polars DataFrame"),
        ({"table": pl.DataFrame({"x": [0.0]})}, "'positions' column"),
        ({"positions": np.zeros((2, 3)), "elements": ["H"]}, "1 values for 2 atoms"),
        ({"positions": np.zeros((1, 3)), "elements": ["Hx"]}, "standard column"),
        ({"positions": np.zeros((1, 3)), "types": [0]}, "at least 1"),
        ({"positions": np.zeros((1, 3)), "types": [1.5]}, "must hold integers"),
        (
            {
                "table": pl.DataFrame(
                    {"positions": [[0.0, 0, 0]], "image": [[0.5, 0, 0]]}
                )
            },
            "must hold integers",
        ),
        (
            {
                "table": pl.DataFrame({"positions": [[0.0, 0, 0]], "element": ["H"]}),
                "elements": "O",
            },
            "'element' column",
        ),
        (
            {
                "table": pl.DataFrame({"positions": [[0.0, 0, 0]], "type": [1]}),
                "types": 2,
            },
            "'type' column",
        ),
        (
            {"table": pl.DataFrame({"positions": [[0.0, 0, 0]], "charge": [None]})},
            "missing values",
        ),
        (
            {"positions": np.zeros((2, 3)), "topology": Topology(bonds=[[0, 2]])},
            "atom 2",
        ),
        ({"positions": np.zeros((2, 3)), "topology": "bonds"}, "Topology"),
    ],
)
def test_invalid_construction(kwargs, message):
    with pytest.raises(InvalidParameter, match=message):
        System(**kwargs)


# --- data access -----------------------------------------------------------------


def test_positions_are_a_read_only_view():
    s = water_dimer()
    r = s.positions
    assert r.shape == (6, 3) and r.dtype == np.float64
    assert not r.flags.writeable
    assert np.shares_memory(r, s.positions)  # same buffer each time, no copy
    with pytest.raises(ValueError):
        r[0, 0] = 1.0


def test_table_is_single_chunk_even_from_concatenated_input():
    parts = [pl.DataFrame({"positions": [[float(i), 0.0, 0.0]]}) for i in range(3)]
    table = pl.concat(parts, rechunk=False)
    s = System(table=table)
    assert s.table.n_chunks() == 1
    np.testing.assert_array_equal(s.positions[:, 0], [0, 1, 2])


def test_info_is_read_only_and_copied():
    virial = np.eye(3)
    s = System(
        np.zeros((1, 3)), info={"virial": virial, "tags": [1, 2], "meta": {"a": 1}}
    )
    virial[0, 0] = 5.0
    assert s.info["virial"][0, 0] == 1.0
    assert not s.info["virial"].flags.writeable
    assert s.info["tags"] == (1, 2)
    with pytest.raises(TypeError):
        s.info["energy"] = 1.0
    with pytest.raises(TypeError):
        s.info["meta"]["a"] = 2


# --- new systems -----------------------------------------------------------------


def test_with_methods_do_not_change_the_original():
    s = water_dimer()
    table, box, info, topology = s.table, s.box, dict(s.info), s.topology
    s.with_positions(np.ones((6, 3)))
    s.with_columns(charge=0.5)
    s.with_box(Box(20.0), scale_positions=True)
    s.with_info(energy=1.0)
    s.with_topology(None)
    s.select([0, 1, 2])
    assert s.table.equals(table) and s.box == box and s.topology == topology
    assert s.info["energy"] == info["energy"]


def test_with_columns():
    s = water_dimer()
    s2 = s.with_columns(
        charge=np.array([-0.8, 0.4, 0.4] * 2),
        forces=np.zeros((6, 3)),
        flag=True,
    )
    assert s2.table.schema["charge"] == pl.Float64
    assert s2.table.schema["forces"] == pl.Array(pl.Float64, 3)
    assert s2.table["flag"].to_list() == [True] * 6
    s3 = s2.with_columns(pl.col("charge") * 2, half=pl.col("charge") / 2)
    np.testing.assert_allclose(s3.table["charge"], [-1.6, 0.8, 0.8] * 2)
    np.testing.assert_allclose(s3.table["half"], [-0.4, 0.2, 0.2] * 2)
    assert s3.topology == s.topology and s3.box == s.box
    with pytest.raises(InvalidParameter, match="3 values for 6 atoms"):
        s.with_columns(charge=[1.0, 2.0, 3.0])
    with pytest.raises(InvalidParameter, match="cannot add columns"):
        s.with_columns(pl.col("missing") * 2)
    with pytest.raises(InvalidParameter, match="standard column"):
        s.with_columns(element="Xx")


def test_with_table():
    s = water_dimer()
    s2 = s.with_table(s.table.drop("type"))
    assert "type" not in s2.table.columns and s2.topology == s.topology
    with pytest.raises(InvalidParameter, match="6 rows"):
        s.with_table(s.table.head(3))
    with pytest.raises(InvalidParameter, match="6 rows"):
        s.with_table({"positions": []})


def test_with_positions():
    s = water_dimer()
    new = np.arange(18, dtype=float).reshape(6, 3)
    np.testing.assert_array_equal(s.with_positions(new).positions, new)
    with pytest.raises(InvalidParameter, match=r"\(6, 3\)"):
        s.with_positions(np.zeros((5, 3)))


def test_with_box_keeps_or_scales_positions():
    s = water_dimer()
    bigger = Box([20.0, 10.0, 10.0])
    kept = s.with_box(bigger)
    np.testing.assert_array_equal(kept.positions, s.positions)
    scaled = s.with_box(bigger, scale_positions=True)
    np.testing.assert_allclose(scaled.positions[:, 0], 2 * s.positions[:, 0])
    np.testing.assert_allclose(
        scaled.box.to_fractional(scaled.positions), s.box.to_fractional(s.positions)
    )
    assert s.with_box([[5, 0, 0], [0, 5, 0], [0, 0, 5]]).box == Box(5.0)


def test_with_info_adds_replaces_and_removes():
    s = water_dimer()
    s2 = s.with_info(energy=-30.0, timestep=100, virial=None)
    assert s2.info["energy"] == -30.0 and s2.info["timestep"] == 100
    assert "virial" not in s2.info
    assert s.with_info(missing=None).info.keys() == s.info.keys()


def test_with_topology():
    s = water_dimer()
    assert s.with_topology(None).topology is None
    with pytest.raises(InvalidParameter, match="atom 9"):
        s.with_topology(Topology(bonds=[[0, 9]]))


# --- selection ------------------------------------------------------------------


def test_select_by_mask_keeps_whole_molecules():
    s = water_dimer()
    second = s.select(np.array([False] * 3 + [True] * 3))
    assert len(second) == 3
    np.testing.assert_array_equal(second.positions, s.positions[3:])
    assert second.topology.bonds["atoms"].to_list() == [[0, 1], [0, 2]]
    assert second.topology.angles["atoms"].to_list() == [[1, 0, 2]]


def test_select_drops_terms_with_unselected_atoms():
    s = water_dimer()
    oxygens = s.select(s.table["element"] == "O")
    assert len(oxygens) == 2
    assert len(oxygens.topology) == 0


def test_select_reorders_and_permutation_round_trip():
    s = water_dimer()
    rng = np.random.default_rng(1)
    order = rng.permutation(len(s))
    shuffled = s.select(order)
    np.testing.assert_array_equal(shuffled.positions, s.positions[order])
    back = shuffled.select(np.argsort(order))
    assert back.table.equals(s.table)
    assert back.topology == s.topology


def test_select_negative_indices():
    s = water_dimer()
    np.testing.assert_array_equal(s.select([-1]).positions, s.positions[[5]])


@pytest.mark.parametrize(
    "atoms, message",
    [
        (np.array([True, False]), "6 entries"),
        ([0, 6], "out of range"),
        ([-7], "out of range"),
        ([0, 0], "must not repeat"),
        ([0.5], "boolean mask or integer"),
        ([[0, 1]], "boolean mask or integer"),
    ],
)
def test_select_invalid(atoms, message):
    with pytest.raises(InvalidParameter, match=message):
        water_dimer().select(atoms)


def test_select_without_topology():
    s = System(np.arange(12, dtype=float).reshape(4, 3), elements="Fe")
    assert s.select([2, 0]).topology is None


# --- display -------------------------------------------------------------------


def test_repr():
    text = repr(water_dimer())
    assert text.startswith("System(6 atoms; H4 O2; box lengths (10, 10, 10) pbc TTT")
    assert "Topology(bonds=4, angles=2" in text
    assert "columns: positions, element, type" in text
    assert repr(System(np.zeros((0, 3)))).startswith("System(0 atoms; box")
