# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

import numpy as np
import polars as pl
import pytest

from mdapy import Topology
from mdapy.errors import InvalidParameter


def ethane_topology():
    """C0-C1 with H2, H3, H4 on C0 and H5, H6, H7 on C1."""
    h0, h1 = [2, 3, 4], [5, 6, 7]
    bonds = [[0, 1]] + [[0, h] for h in h0] + [[1, h] for h in h1]
    angles = (
        [[a, 0, b] for i, a in enumerate(h0) for b in h0[i + 1 :]]
        + [[h, 0, 1] for h in h0]
        + [[a, 1, b] for i, a in enumerate(h1) for b in h1[i + 1 :]]
        + [[0, 1, h] for h in h1]
    )
    dihedrals = [[a, 0, 1, b] for a in h0 for b in h1]
    return Topology(bonds=bonds, angles=angles, dihedrals=dihedrals)


def test_counts_of_ethane():
    t = ethane_topology()
    assert (len(t.bonds), len(t.angles), len(t.dihedrals), len(t.impropers)) == (
        7,
        12,
        9,
        0,
    )
    assert len(t) == 28
    assert t.max_atom_index() == 7
    assert repr(t) == "Topology(bonds=7, angles=12, dihedrals=9, impropers=0)"


def test_schema():
    t = ethane_topology()
    assert t.bonds.schema == pl.Schema(
        {"atoms": pl.Array(pl.Int64, 2), "type": pl.Int32}
    )
    assert t.dihedrals.schema["atoms"] == pl.Array(pl.Int64, 4)
    assert t.impropers.schema["atoms"] == pl.Array(pl.Int64, 4)
    np.testing.assert_array_equal(t.bonds["type"], np.ones(7))


def test_empty_topology():
    t = Topology()
    assert len(t) == 0 and t.max_atom_index() == -1
    assert t.angles.schema["atoms"] == pl.Array(pl.Int64, 3)
    assert Topology(bonds=np.empty((0, 2), dtype=int)) == t
    assert Topology(bonds=[]) == t


def test_dataframe_input_keeps_types_labels_and_extra_columns():
    bonds = pl.DataFrame(
        {
            "atoms": [[0, 1], [1, 2]],
            "type": [2, 1],
            "type_label": ["C-O", "O-H"],
            "order": [2.0, 1.0],
        }
    )
    t = Topology(bonds=bonds)
    assert t.bonds["type"].dtype == pl.Int32
    assert t.bonds["type"].to_list() == [2, 1]
    assert t.bonds["type_label"].to_list() == ["C-O", "O-H"]
    assert t.bonds["order"].to_list() == [2.0, 1.0]
    # a DataFrame without types gets type 1
    t2 = Topology(angles=pl.DataFrame({"atoms": [[0, 1, 2]]}))
    assert t2.angles["type"].to_list() == [1]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"bonds": [[0, 1, 2]]}, r"\(M, 2\)"),
        ({"angles": [0, 1, 2]}, r"\(M, 3\)"),
        ({"bonds": [[0, -1]]}, "negative"),
        ({"angles": [[1, 0, 1]]}, "repeats an atom"),
        ({"dihedrals": [[0, 1, 2, 2]]}, "repeats an atom"),
        ({"bonds": pl.DataFrame({"atoms": [[0, 1]], "type": [0]})}, "at least 1"),
        ({"bonds": pl.DataFrame({"pairs": [[0, 1]]})}, "'atoms' column"),
        ({"bonds": pl.DataFrame({"atoms": [[0, 1, 2]], "type": [1]})}, "2 integers"),
        ({"bonds": pl.DataFrame({"atoms": [[0, 1]], "type": ["x"]})}, "integers"),
        ({"bonds": pl.DataFrame({"atoms": [[0, 1]], "type": [None]})}, "missing"),
    ],
)
def test_invalid_terms(kwargs, message):
    with pytest.raises(InvalidParameter, match=message):
        Topology(**kwargs)


def test_remap_drops_terms_with_removed_atoms():
    t = ethane_topology()
    mapping = np.array([0, 1, -1, -1, -1, -1, -1, -1])  # keep only the carbons
    r = t.remap(mapping)
    assert r.bonds["atoms"].to_list() == [[0, 1]]
    assert len(r.angles) == len(r.dihedrals) == 0


def test_remap_renumbers_atoms():
    t = Topology(bonds=[[0, 1], [1, 2]], angles=[[0, 1, 2]])
    r = t.remap(np.array([2, 0, 1]))
    assert r.bonds["atoms"].to_list() == [[2, 0], [0, 1]]
    assert r.angles["atoms"].to_list() == [[2, 0, 1]]


def test_remap_rejects_short_mapping():
    with pytest.raises(InvalidParameter, match="beyond the mapping"):
        Topology(bonds=[[0, 5]]).remap(np.arange(3))


def test_equality():
    assert ethane_topology() == ethane_topology()
    assert ethane_topology() != Topology(bonds=[[0, 1]])
    assert (Topology() == 1) is False
