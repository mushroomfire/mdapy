# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

import numpy as np
import polars as pl
from mdapy import System
from mdapy.box import Box


def test_build_bond_with_scalar_cutoff():
    data = pl.DataFrame(
        {
            "x": [0.0, 1.0, 2.3, 5.0],
            "y": [0.0, 0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0, 0.0],
            "type": [1, 2, 2, 1],
        }
    )
    system = System(data=data, box=Box([10.0, 10.0, 10.0], boundary=[0, 0, 0]))

    bond = system.create_bonds(1.5)

    expected = np.array([[0, 1], [1, 2]], np.int32)
    assert np.array_equal(bond, expected)
    assert np.array_equal(system.bond, expected)


def test_build_bond_with_pairwise_cutoff():
    data = pl.DataFrame(
        {
            "x": [0.0, 1.0, 2.3, 5.0],
            "y": [0.0, 0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0, 0.0],
            "type": [1, 2, 2, 1],
        }
    )
    system = System(data=data, box=Box([10.0, 10.0, 10.0], boundary=[0, 0, 0]))

    bond = system.create_bonds({(1, 1): 0.5, (1, 2): 1.1, (2, 2): 1.2})

    expected = np.array([[0, 1]], np.int32)
    assert np.array_equal(bond, expected)


def test_build_bond_with_element_cutoff():
    data = pl.DataFrame(
        {
            "x": [0.0, 1.0, 2.3, 5.0],
            "y": [0.0, 0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0, 0.0],
            "element": ["Cu", "Zr", "Zr", "Cu"],
        }
    )
    system = System(data=data, box=Box([10.0, 10.0, 10.0], boundary=[0, 0, 0]))

    bond = system.create_bonds({("Cu", "Cu"): 0.5, ("Cu", "Zr"): 1.1, ("Zr", "Zr"): 1.2})

    expected = np.array([[0, 1]], np.int32)
    assert np.array_equal(bond, expected)


def test_build_bond_in_small_periodic_box():
    system = System(
        pos=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], float),
        box=Box([2.0, 2.0, 2.0]),
    )

    bond = system.create_bonds(1.1)

    expected = np.array([[0, 1]], np.int32)
    assert np.array_equal(bond, expected)


# ---------------------------------------------------------------------------
# delete_overlap
# ---------------------------------------------------------------------------

def _line_system(xs, box=20.0, boundary=(1, 1, 1)):
    """Atoms strung along x at the given coordinates."""
    xs = np.asarray(xs, float)
    pos = np.column_stack([xs, np.zeros_like(xs), np.zeros_like(xs)])
    return System(pos=pos, box=Box([box, box, box], boundary=list(boundary)))


def test_delete_overlap_basic_pair():
    """The higher-index atom of an overlapping pair is removed."""
    s = _line_system([1.0, 1.05, 5.0])
    n = s.delete_overlap(0.1)
    assert n == 1 and s.N == 2
    # atom 1 (x=1.05) removed; atom 0 (x=1.0) kept.
    np.testing.assert_allclose(sorted(s.data["x"].to_list()), [1.0, 5.0])


def test_delete_overlap_cluster_keeps_lowest_index():
    """A cluster of mutually overlapping atoms collapses to its
    lowest-index member."""
    s = _line_system([1.0, 1.03, 1.06])
    n = s.delete_overlap(0.1)
    assert n == 2 and s.N == 1
    np.testing.assert_allclose(s.data["x"].to_list(), [1.0])


def test_delete_overlap_deleted_atom_does_not_cascade():
    """A removed atom cannot itself trigger further removals: with
    0-1 and 1-2 overlapping but 0-2 apart, only atom 1 is removed."""
    # spacings: 0-1 = 0.08, 1-2 = 0.08, 0-2 = 0.16 (> rc)
    s = _line_system([1.0, 1.08, 1.16])
    n = s.delete_overlap(0.1)
    assert n == 1 and s.N == 2
    np.testing.assert_allclose(sorted(s.data["x"].to_list()), [1.0, 1.16])


def test_delete_overlap_no_overlap_is_noop():
    s = _line_system([1.0, 5.0, 9.0])
    n = s.delete_overlap(0.5)
    assert n == 0 and s.N == 3


def test_delete_overlap_respects_pbc():
    """Atoms close across a periodic boundary count as overlapping."""
    s = _line_system([0.02, 19.98], box=20.0)  # PBC separation = 0.04
    n = s.delete_overlap(0.1)
    assert n == 1 and s.N == 1


def test_delete_overlap_small_box_enlarge_path():
    """Works when the box is smaller than 2*rc and the neighbor list is
    built on an internally replicated system."""
    s = _line_system([0.1, 0.5], box=1.0)
    n = s.delete_overlap(2.0)  # 2*rc = 4 > box, forces replication
    assert n == 1 and s.N == 1


def test_delete_overlap_resets_neighbor_state():
    """Cached neighbor / bond data is cleared since indices change."""
    s = _line_system([1.0, 1.05, 5.0])
    s.create_bonds(0.5)
    assert hasattr(s, "bond")
    s.delete_overlap(0.1)
    assert not hasattr(s, "bond")
    assert not hasattr(s, "verlet_list")
