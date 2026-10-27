# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

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
    assert s["positions"].shape == (0, 3)
    assert len(s.gather([])) == 0 and len(s.filter("element == 'H'")) == 0


def test_default_box_encloses_the_atoms():
    s = System([[0, 0, 0], [1, 2, 3]])
    assert not s.box.pbc.any()
    frac = s.box.to_fractional(s["positions"])
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
    r = s["positions"]
    assert r.shape == (6, 3) and r.dtype == np.float64
    assert not r.flags.writeable
    assert np.shares_memory(r, s["positions"])  # same buffer each time, no copy
    with pytest.raises(ValueError):
        r[0, 0] = 1.0


def test_table_is_single_chunk_even_from_concatenated_input():
    parts = [pl.DataFrame({"positions": [[float(i), 0.0, 0.0]]}) for i in range(3)]
    table = pl.concat(parts, rechunk=False)
    s = System(table=table)
    assert s.table.n_chunks() == 1
    np.testing.assert_array_equal(s["positions"][:, 0], [0, 1, 2])


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


def test_arrays_given_are_copied():
    """A system never changes, even when the arrays it was given change."""
    positions, types = np.zeros((2, 3)), np.array([1, 2])
    charges, moved = np.array([1.0, -1.0]), np.ones((2, 3))
    s = System(positions, types=types)
    charged = s.with_columns(charge=charges)
    shifted = s.with_columns(positions=moved)
    for array in (positions, types, charges, moved):
        array[0] = 7
    assert s["positions"][0, 0] == 0.0 and s.table["type"][0] == 1
    assert charged.table["charge"][0] == 1.0
    assert shifted["positions"][0, 0] == 1.0


# --- new systems -----------------------------------------------------------------


def test_with_methods_do_not_change_the_original():
    s = water_dimer()
    table, box, info, topology = s.table, s.box, dict(s.info), s.topology
    s.with_columns(positions=np.ones((6, 3)))
    s.with_columns(charge=0.5)
    s.with_box(Box(20.0), scale_positions=True)
    s.with_info(energy=1.0)
    s.with_topology(None)
    s.filter(element="O")
    s.gather([0, 1, 2])
    s.sort("element")
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


def test_new_positions():
    s = water_dimer()
    new = np.arange(18, dtype=float).reshape(6, 3)
    np.testing.assert_array_equal(s.with_columns(positions=new)["positions"], new)
    with pytest.raises(InvalidParameter, match="5 values for 6 atoms"):
        s.with_columns(positions=np.zeros((5, 3)))
    for wrong in (np.zeros((6, 2)), np.zeros(6)):
        with pytest.raises(InvalidParameter, match="must hold 3 values per atom"):
            s.with_columns(positions=wrong)
    with pytest.raises(InvalidParameter, match="must hold 3 values per atom"):
        s.with_columns(velocities=np.zeros((6, 4)))
    assert not hasattr(s, "positions") and not hasattr(s, "with_positions")


def test_with_box_keeps_or_scales_positions():
    s = water_dimer()
    bigger = Box([20.0, 10.0, 10.0])
    kept = s.with_box(bigger)
    np.testing.assert_array_equal(kept["positions"], s["positions"])
    scaled = s.with_box(bigger, scale_positions=True)
    np.testing.assert_allclose(scaled["positions"][:, 0], 2 * s["positions"][:, 0])
    np.testing.assert_allclose(
        scaled.box.to_fractional(scaled["positions"]),
        s.box.to_fractional(s["positions"]),
    )
    assert s.with_box([[5, 0, 0], [0, 5, 0], [0, 0, 5]]).box == Box(5.0)


def test_with_info_adds_replaces_and_removes():
    s = water_dimer()
    s2 = s.with_info(energy=-30.0, timestep=100, virial=None)
    assert s2.info["energy"] == -30.0 and s2.info["timestep"] == 100
    assert "virial" not in s2.info
    assert s.with_info(missing=None).info.keys() == s.info.keys()


def test_with_info_replaces_all_with_a_mapping():
    s = water_dimer()
    assert dict(s.with_info({}).info) == {}
    replaced = s.with_info({"label": "dimer", "energy": 1.0}, energy=None, step=3)
    assert dict(replaced.info) == {"label": "dimer", "step": 3}
    assert s.with_info().info.keys() == s.info.keys()
    part = s.filter(element="O").with_info({})
    assert len(part) == 2 and len(part.info) == 0
    with pytest.raises(InvalidParameter, match="must be a mapping"):
        s.with_info([("energy", 1.0)])


# --- columns chosen and renamed -------------------------------------------------------


def test_select_keeps_columns_in_order():
    s = water_dimer().with_columns(charge=[-0.8, 0.4, 0.4] * 2, csp=0.0)
    kept = s.select("element", "positions")
    assert kept.table.columns == ["element", "positions"]
    assert np.shares_memory(kept["positions"], s["positions"])
    assert kept.topology == s.topology and kept.info == s.info and kept.box == s.box
    computed = s.select(
        "positions", (pl.col("charge") * 2).alias("double"), pl.col("type")
    )
    assert computed.table.columns == ["positions", "double", "type"]
    np.testing.assert_allclose(computed["double"], [-1.6, 0.8, 0.8] * 2)
    total = s.select("positions", pl.col("charge").sum().alias("total"))
    np.testing.assert_allclose(total["total"], np.zeros(6), atol=1e-12)
    every = s.select(pl.all())
    assert every.table.equals(s.table)


@pytest.mark.parametrize(
    "columns, message",
    [
        (("element",), "select 'positions' too"),
        (("positions", "missing"), "cannot select columns"),
        (("positions", pl.col("type").head(2)), "cannot select columns"),
        (("positions", pl.lit(0.5).alias("type")), "must hold integers"),
    ],
)
def test_select_invalid(columns, message):
    with pytest.raises(InvalidParameter, match=message):
        water_dimer().select(*columns)


def test_drop_removes_columns():
    s = water_dimer().with_columns(csp=0.0, structure=1)
    dropped = s.drop("csp", "structure")
    assert dropped.table.columns == ["positions", "element", "type"]
    assert np.shares_memory(dropped["positions"], s["positions"])
    assert s.drop().table.columns == s.table.columns
    with pytest.raises(InvalidParameter, match="cannot be dropped"):
        s.drop("positions")
    with pytest.raises(InvalidParameter, match="no column 'missing'"):
        s.drop("csp", "missing")


def test_rename_columns_and_check_standard_names():
    s = water_dimer().with_columns(q=[-0.8, 0.4, 0.4] * 2, kind=[1.0, 2.0, 2.0] * 2)
    renamed = s.rename({"q": "charge", "type": "species"})
    assert renamed.table.columns == [
        "positions",
        "element",
        "species",
        "charge",
        "kind",
    ]
    assert renamed.table.schema["charge"] == pl.Float64
    assert s.drop("type").rename({"kind": "type"}).table.schema["type"] == pl.Int32
    assert s.rename({"positions": "positions"}).table.equals(s.table)
    with pytest.raises(InvalidParameter, match="cannot be renamed"):
        s.rename({"positions": "r"})
    with pytest.raises(InvalidParameter, match="no column 'missing'"):
        s.rename({"missing": "x"})
    with pytest.raises(InvalidParameter, match="duplicate"):
        s.rename({"q": "kind"})
    with pytest.raises(InvalidParameter, match="must hold integers"):
        s.drop("type").rename({"q": "type"})


def test_with_topology():
    s = water_dimer()
    assert s.with_topology(None).topology is None
    with pytest.raises(InvalidParameter, match="atom 9"):
        s.with_topology(Topology(bonds=[[0, 9]]))


# --- values by name -----------------------------------------------------------------


def test_values_of_columns_components_and_expressions():
    s = water_dimer().with_columns(
        charge=[-0.8, 0.4, 0.4] * 2,
        flag=[True, False, False] * 2,
        score=pl.Series([1.0, None, 3.0, 4.0, None, 6.0]),
        group=pl.Series([[0, 1]] * 6, dtype=pl.Array(pl.Int64, 2)),
        **{"Shear Strain": np.arange(6.0)},
    )
    r = s["positions"]
    # columns and components are read where they are, without a copy
    for name, expected in (
        ("positions", r),
        ("positions.z", r[:, 2]),
        ("positions.0", r[:, 0]),
    ):
        values = s[name]
        np.testing.assert_array_equal(values, expected)
        assert np.shares_memory(values, r) and not values.flags.writeable
    np.testing.assert_array_equal(s["charge"], [-0.8, 0.4, 0.4] * 2)
    assert s["element"].tolist() == ["O", "H", "H"] * 2
    assert s["flag"].dtype == bool and s["group.1"].tolist() == [1] * 6
    np.testing.assert_array_equal(s["Shear Strain"], np.arange(6.0))
    assert np.isnan(s["score"][[1, 4]]).all()
    for name in ("element", "flag", "score", "charge", "group"):
        assert not s[name].flags.writeable
    # expressions are computed
    np.testing.assert_allclose(
        s["sqrt(positions.x**2 + positions.y**2)"], np.hypot(r[:, 0], r[:, 1])
    )
    np.testing.assert_array_equal(
        s["`Shear Strain` * 2 + type"], np.arange(6.0) * 2 + [1, 2, 2] * 2
    )
    np.testing.assert_array_equal(s["charge > 0"], [False, True, True] * 2)
    assert s["2.5"].tolist() == [2.5] * 6
    assert len(s["positions.z"]) == len(s)


@pytest.mark.parametrize(
    "name, error, message",
    [
        (0, TypeError, "indexed by column names"),
        (slice(0, 2), TypeError, "filter and gather"),
        ("missing", InvalidParameter, "no column 'missing'"),
        ("positions.q", InvalidParameter, "no component 'q'"),
        ("positions + 1", InvalidParameter, "3 components"),
        ("type >", InvalidParameter, "cannot read"),
        ("element + 1", InvalidParameter, "cannot evaluate"),
    ],
)
def test_values_invalid(name, error, message):
    with pytest.raises(error, match=message):
        water_dimer()[name]


def test_a_system_is_not_a_sequence_of_atoms():
    s = water_dimer()
    with pytest.raises(TypeError):
        iter(s)
    with pytest.raises(TypeError):
        "positions" in s  # noqa: B015


# --- which columns are checked -----------------------------------------------------


def test_new_systems_check_only_the_columns_that_change(monkeypatch):
    import mdapy.core.system as module

    checked = []
    original = module._normalize_table

    def spy(table, columns=None):
        checked.append(None if columns is None else sorted(columns))
        return original(table, columns)

    monkeypatch.setattr(module, "_normalize_table", spy)
    s = water_dimer()
    checked.clear()
    s.with_box(Box(20.0))
    s.with_info(energy=1.0)
    s.with_topology(None)
    assert checked == []  # the table is the same
    s.filter(element="O")
    s.gather([1, 0])
    s.sort("type")
    assert checked == [[], [], []]  # rows of a table already checked
    s.with_columns(charge=0.5, forces=np.zeros((6, 3)))
    s.with_columns(positions=np.ones((6, 3)))
    s.with_columns(pl.col("type") + 1)
    assert checked[3:] == [["charge", "forces"], ["positions"], ["type"]]
    s.with_columns(pl.all())  # cannot tell which: every column
    s.with_table(s.table)
    assert checked[6:] == [None, None]
    with pytest.raises(InvalidParameter, match="finite"):
        s.with_columns(positions=np.full((6, 3), np.nan))
    with pytest.raises(InvalidParameter, match="at least 1"):
        s.with_columns(type=0)


# --- immutability and copies ---------------------------------------------------------


def test_nothing_can_be_set_or_deleted():
    s = water_dimer()
    for target in (s, s.box, s.topology):
        with pytest.raises(AttributeError, match="cannot be changed"):
            target.anything = 1
        with pytest.raises(AttributeError, match="cannot be changed"):
            del target.anything
    with pytest.raises(AttributeError, match="cannot be changed"):
        s._System__table = s.table.head(1)
    assert len(s) == 6
    assert not hasattr(s, "_cache") and not hasattr(s, "_table")


def test_copies_are_the_same_object():
    import copy

    s = water_dimer()
    for target in (s, s.box, s.topology):
        assert copy.copy(target) is target
        assert copy.deepcopy(target) is target
    pair = copy.deepcopy([s, s])
    assert pair[0] is s and pair[1] is s


def test_systems_pickle_for_other_processes():
    import pickle
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    s = water_dimer().with_info(
        source={"code": "x", "steps": [1, 2]}, weights=np.arange(3.0)
    )
    s = s.with_box(Box(s.box.matrix, (1.0, 2.0, 3.0), (True, False, True)))
    t = pickle.loads(pickle.dumps(s))
    assert t is not s and t.table.equals(s.table)
    assert t.box == s.box and t.topology == s.topology
    assert t.info["source"]["steps"] == (1, 2)
    np.testing.assert_array_equal(t.info["weights"], np.arange(3.0))
    # still as immutable as the original
    with pytest.raises(TypeError):
        t.info["source"]["code"] = "y"
    with pytest.raises(ValueError):
        t.info["weights"][0] = 1.0
    with pytest.raises(AttributeError, match="cannot be changed"):
        t.anything = 1
    with ProcessPoolExecutor(1, mp_context=get_context("spawn")) as pool:
        assert list(pool.map(len, [s, s])) == [6, 6]


def test_new_systems_share_the_columns_they_keep():
    s = water_dimer().with_columns(mass=np.arange(6.0))
    charged = s.with_columns(charge=0.1)
    moved = s.with_columns(positions=s["positions"] + 1.0)
    assert np.shares_memory(charged["positions"], s["positions"])
    assert not np.shares_memory(moved["positions"], s["positions"])
    mass = s.table["mass"].to_numpy()
    assert np.shares_memory(moved.table["mass"].to_numpy(), mass)
    assert np.shares_memory(s.with_box(Box(30.0))["positions"], s["positions"])
    np.testing.assert_array_equal(s["positions"], water_dimer()["positions"])


# --- filter ---------------------------------------------------------------------


def test_filter_by_mask_keeps_whole_molecules():
    s = water_dimer()
    second = s.filter(np.array([False] * 3 + [True] * 3))
    assert len(second) == 3
    np.testing.assert_array_equal(second["positions"], s["positions"][3:])
    assert second.topology.bonds["atoms"].to_list() == [[0, 1], [0, 2]]
    assert second.topology.angles["atoms"].to_list() == [[1, 0, 2]]
    assert second.info == s.info and second.box == s.box


def test_filter_drops_terms_with_atoms_left_out():
    s = water_dimer()
    oxygens = s.filter(element="O")
    assert len(oxygens) == 2
    assert len(oxygens.topology) == 0


def test_filter_by_expressions_conditions_and_keywords():
    s = water_dimer().with_columns(charge=[-0.8, 0.4, 0.4] * 2)
    hydrogens = [1, 2, 4, 5]
    for ways in (
        (s.table["element"] == "H",),
        (pl.col("element") == "H",),
        ("element == 'H'",),
        ("charge > 0",),
        ("charge > 0", pl.col("type") == 2),
        (),
    ):
        expected = hydrogens if ways else list(range(6))
        kept = s.filter(*ways, element="H") if ways else s.filter()
        if not ways:
            assert kept is not s and len(kept) == 6
        np.testing.assert_array_equal(kept["positions"], s["positions"][expected])
    right = s.filter("positions.x > 2 and element != 'O'")
    np.testing.assert_array_equal(right["positions"], s["positions"][[4, 5]])
    near = s.filter("positions.x**2 + positions.y**2 < 1", type=2)
    np.testing.assert_array_equal(near["positions"], s["positions"][[1, 2]])
    assert len(s.filter("element in ['Cu', 'Ni']")) == 0


def test_filter_treats_missing_values_as_false():
    s = water_dimer().with_columns(score=pl.Series([1.0, None, 3.0, None, 5.0, 6.0]))
    np.testing.assert_array_equal(
        s.filter("score > 2")["positions"], s["positions"][[2, 4, 5]]
    )


@pytest.mark.parametrize(
    "conditions, keywords, message",
    [
        ((np.array([True, False]),), {}, "6 booleans"),
        ((np.arange(6),), {}, "6 booleans"),
        (("charge > 0",), {}, "no column 'charge'"),
        (("positions > 0",), {}, "3 components"),
        ((), {"positions": 1.0}, "3 components"),
        (("positions.q > 0",), {}, "no component 'q'"),
        ((), {"missing": 1}, "no column 'missing'"),
        (("type + 1",), {}, "does not give true or false"),
        ((pl.col("type") * 2,), {}, "does not give true or false"),
        (("element > 3",), {}, "cannot evaluate"),
        (("type >",), {}, "cannot read"),
    ],
)
def test_filter_invalid(conditions, keywords, message):
    with pytest.raises(InvalidParameter, match=message):
        water_dimer().filter(*conditions, **keywords)


# --- gather ---------------------------------------------------------------------


def test_gather_reorders_and_a_permutation_round_trips():
    s = water_dimer()
    rng = np.random.default_rng(1)
    order = rng.permutation(len(s))
    shuffled = s.gather(order)
    np.testing.assert_array_equal(shuffled["positions"], s["positions"][order])
    back = shuffled.gather(np.argsort(order))
    assert back.table.equals(s.table)
    assert back.topology == s.topology


def test_gather_negative_indices_and_no_topology():
    s = water_dimer()
    np.testing.assert_array_equal(s.gather([-1])["positions"], s["positions"][[5]])
    bare = System(np.arange(12, dtype=float).reshape(4, 3), elements="Fe")
    assert bare.gather([2, 0]).topology is None
    assert len(bare.gather([])) == 0


@pytest.mark.parametrize(
    "atoms, message",
    [
        ([0, 6], "out of range"),
        ([-7], "out of range"),
        ([0, 0], "must not repeat"),
        ([0.5], "integer indices"),
        ([[0, 1]], "integer indices"),
        (np.ones(6, dtype=bool), "filter takes a boolean mask"),
    ],
)
def test_gather_invalid(atoms, message):
    with pytest.raises(InvalidParameter, match=message):
        water_dimer().gather(atoms)


# --- sort -----------------------------------------------------------------------


def test_sort_by_columns_components_and_expressions():
    s = water_dimer().with_columns(id=[6, 5, 4, 3, 2, 1])
    by_id = s.sort("id")
    np.testing.assert_array_equal(by_id["positions"], s["positions"][::-1])
    # the renumbered topology still joins the same atoms
    assert by_id.gather(np.argsort([6, 5, 4, 3, 2, 1])[::-1]).topology is not None
    assert by_id.topology == s.gather([5, 4, 3, 2, 1, 0]).topology
    # elements by atomic number, ties kept in order
    np.testing.assert_array_equal(
        s.sort("element").table["element"].to_list(), ["H"] * 4 + ["O"] * 2
    )
    np.testing.assert_array_equal(
        s.sort("element").table["id"].to_list(), [5, 4, 2, 1, 6, 3]
    )
    np.testing.assert_array_equal(
        s.sort("positions.x")["positions"][:, 0], np.sort(s["positions"][:, 0])
    )
    np.testing.assert_array_equal(
        s.sort("type", "positions.y", descending=[False, True]).table["id"].to_list(),
        [6, 3, 4, 1, 5, 2],
    )
    np.testing.assert_array_equal(
        s.sort(pl.col("positions").arr.get(0), descending=True)["positions"][:, 0],
        np.sort(s["positions"][:, 0])[::-1],
    )


@pytest.mark.parametrize(
    "by, message",
    [
        ((), "give a column"),
        (("missing",), "no column 'missing'"),
        (("positions",), "3 components"),
        (("positions.q",), "no component 'q'"),
        ((pl.col("missing"),), "cannot sort"),
    ],
)
def test_sort_invalid(by, message):
    with pytest.raises(InvalidParameter, match=message):
        water_dimer().sort(*by)


def test_a_neighbor_list_does_not_fit_reordered_atoms():
    from mdapy.identify import CentroSymmetryParameter
    from mdapy.neighbor import NeighborList

    s = System(np.random.default_rng(2).random((30, 3)) * 6, Box(6.0))
    nl = NeighborList.build(s, num_neighbors=4)
    flipped = s.gather(np.arange(30)[::-1])
    assert nl.fits(s) and not nl.fits(flipped) and not nl.fits(s.sort("positions.x"))
    with pytest.raises(InvalidParameter, match="other positions or another box"):
        CentroSymmetryParameter(4).compute(flipped, neighbors=nl)
    # attached before reordering, a result follows its atoms
    csp = CentroSymmetryParameter(4).compute(s, neighbors=nl)
    moved = s.with_columns(csp).gather(np.arange(30)[::-1])
    np.testing.assert_array_equal(moved.table["csp"], csp.csp[::-1])


# --- display -------------------------------------------------------------------


def test_repr():
    assert repr(water_dimer()) == "\n".join(
        [
            "System: 6 atoms (H4 O2)",
            "  box:      10 × 10 × 10 Å, orthogonal, periodic x y z",
            "  columns:  positions (3), element, type",
            "  info:     energy = -27.5, virial (3, 3)",
            "  topology: 4 bonds, 2 angles",
        ]
    )
    water = System(
        [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]],
        Box(10.0, origin=(-5, -5, -5), pbc=False),
        elements=["O", "H", "H"],
        topology=Topology(bonds=[[0, 1]], angles=[[1, 0, 2]]),
        info={"name": "water", "step": 7, "fixed": True, "tags": [1, 2]},
    )
    assert repr(water).splitlines() == [
        "System: 3 atoms (H2 O)",
        "  box:      10 × 10 × 10 Å at (-5, -5, -5), orthogonal, not periodic",
        "  columns:  positions (3), element",
        "  info:     name = 'water', step = 7, fixed = True, tags (2 values)",
        "  topology: 1 bond, 1 angle",
    ]
    tilted = System(
        np.zeros((1, 3)),
        Box([[10, 0, 0], [3, 10, 0], [0, 0, 10]], pbc=(True, True, False)),
        info={"meta": {"a": 1}, "note": "x" * 30, "nothing": None, "scale": 2.5},
        topology=Topology(),
    )
    assert repr(tilted).splitlines() == [
        "System: 1 atom",
        "  box:      a 10, b 10.44, c 10 Å; α 90°, β 90°, γ 73.3°; periodic x y",
        "  columns:  positions (3)",
        "  info:     meta (1 entries), note = 'xxxxxxxxxxxxxxxxxxxxx'..., nothing = None,",
        "            scale = 2.5",
        "  topology: none",
    ]
    wide = System(np.zeros((2, 3))).with_columns(
        **{f"column_{k}": float(k) for k in range(12)}
    )
    lines = repr(wide).splitlines()
    assert lines[0] == "System: 2 atoms" and all(len(line) <= 88 for line in lines)
    assert lines[3].startswith("            column_") and "column_11" in lines[-1]
    assert (
        repr(System(np.zeros((0, 3)), Box(10.0))).splitlines()[0] == "System: 0 atoms"
    )
    assert "info:" not in repr(System(np.zeros((1, 3)), Box(10.0)))


@pytest.mark.parametrize(
    "change",
    [
        lambda t: t.insert_column(0, pl.Series("z", [0, 0, 0, 0, 0, 0])),
        lambda t: t.replace_column(0, pl.Series("positions", np.ones((6, 3)))),
        lambda t: t.drop_in_place("element"),
        lambda t: setattr(t, "columns", [c + "_x" for c in t.columns]),
        lambda t: t.extend(t.clone()),
        lambda t: t.vstack(t.clone(), in_place=True),
        lambda t: t["element"].scatter(0, "H"),
        lambda t: t["mass"].append(pl.Series("mass", [1.0])),
    ],
)
def test_the_table_cannot_be_changed_in_place(change):
    s = water_dimer().with_columns(mass=np.arange(6.0))
    before = s.table
    change(s.table)
    assert s.table.equals(before) and len(s) == 6
    with pytest.raises(ValueError):
        s.table["mass"].to_numpy()[0] = 5.0
    # nor through the table it was made from
    table = s.table
    made = System(table=table)
    table.insert_column(0, pl.Series("z", np.zeros(6)))
    assert "z" not in made.table.columns
