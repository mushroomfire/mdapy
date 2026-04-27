# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""
Brute-force verification of mdapy's cutoff neighbor list.

We do not depend on Ovito here. For each test atom we compute the reference
neighbor set by an O(N) minimum-image scan against every atom in the box
(or its mdapy-internal replica when the box is too small for the cutoff),
then compare indices, distances and counts to what `build_neighbor` returned.
"""

import numpy as np
import pytest

import mdapy as mp
from mdapy.box import Box


# ---------------------------------------------------------------------------
# Brute-force reference
# ---------------------------------------------------------------------------

def _bf_neighbors(idx, pos, box, rc):
    """Return (indices, distances) of atoms within rc of `idx`, self excluded."""
    rij = pos - pos[idx]
    frac = rij @ box.inverse_box
    pbc = np.asarray(box.boundary, dtype=float)
    # Apply the minimum-image shift only on periodic axes.
    frac -= np.round(frac) * pbc
    rij = frac @ box.box
    dist = np.linalg.norm(rij, axis=1)
    mask = np.ones_like(dist, dtype=bool)
    mask[idx] = False
    mask &= dist <= rc + 1e-9        # mdapy's convention is `<= rc`
    inds = np.nonzero(mask)[0]
    return inds, dist[inds]


def _ref_pos_and_box(system):
    """Use the (possibly replicated) data that mdapy actually built against."""
    if hasattr(system, "_enlarge_data"):
        pos = system._enlarge_data.select(["x", "y", "z"]).to_numpy()
        box = system._enlarge_box
    else:
        pos = system.data.select(["x", "y", "z"]).to_numpy()
        box = system.box
    return pos, box


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------

def _check(system, rc, sample_idx=None):
    pos, box = _ref_pos_and_box(system)
    N_total = pos.shape[0]
    if sample_idx is None:
        # Exercise boundary, middle and edge atoms; original atoms only
        N_orig = system.N
        sample_idx = sorted(set(
            [0, N_orig // 4, N_orig // 2, (3 * N_orig) // 4, N_orig - 1]
        ))

    for i in sample_idx:
        ref_idx, ref_dist = _bf_neighbors(i, pos, box, rc)
        nn = int(system.neighbor_number[i])

        assert nn == len(ref_idx), (
            f"atom {i}: count mismatch — mdapy={nn} brute={len(ref_idx)} (rc={rc})"
        )

        mdapy_idx = system.verlet_list[i, :nn]
        mdapy_dist = system.distance_list[i, :nn]
        assert np.all(mdapy_idx >= 0), f"atom {i}: negative index in valid slot"
        assert mdapy_idx.max(initial=-1) < N_total, (
            f"atom {i}: index out of range vs reference data"
        )

        # Sort both sides by index for a direct comparison.
        order_mdapy = np.argsort(mdapy_idx)
        order_ref = np.argsort(ref_idx)
        assert np.array_equal(mdapy_idx[order_mdapy], ref_idx[order_ref]), (
            f"atom {i}: neighbor index sets differ\n"
            f"  mdapy: {sorted(mdapy_idx.tolist())}\n"
            f"  brute: {sorted(ref_idx.tolist())}"
        )
        assert np.allclose(
            mdapy_dist[order_mdapy], ref_dist[order_ref], atol=1e-6
        ), f"atom {i}: distances differ"


# ---------------------------------------------------------------------------
# File-driven tests (cover ortho/triclinic, big/small, with/without max_neigh)
# ---------------------------------------------------------------------------

FILES = [
    "tests/input_files/rec_box_big.xyz",
    "tests/input_files/rec_box_small.xyz",
    "tests/input_files/tri_box_big.xyz",
    "tests/input_files/tri_box_small.xyz",
    "tests/input_files/AlCrNi.xyz",
    "tests/input_files/HexDiamond.xyz",
]


@pytest.mark.parametrize("filename", FILES)
@pytest.mark.parametrize("rc", [2.5, 3.5, 5.0])
# NOTE: max_neigh must be >= the real maximum coordination for all combos in
# this list, otherwise the C++ kernel overruns the array and corrupts memory
# (no bounds check). 150 is safe for every (file, rc) pair we sweep here.
@pytest.mark.parametrize("max_neigh", [None, 150])
def test_files(filename, rc, max_neigh):
    s = mp.System(filename)
    s.build_neighbor(rc, max_neigh)
    _check(s, rc)


# ---------------------------------------------------------------------------
# Synthetic tests — boundaries, geometries, edge cases
# ---------------------------------------------------------------------------

def _rand_pos(rng, n, box_mat):
    """Random fractional positions mapped through `box_mat`."""
    return rng.random((n, 3)) @ box_mat


@pytest.mark.parametrize("boundary", [
    [1, 1, 1],
    [0, 0, 0],
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1],
    [1, 0, 0],
])
def test_synthetic_orthogonal_mixed_pbc(boundary):
    rng = np.random.default_rng(42)
    box_mat = np.diag([12.0, 13.0, 14.0])
    pos = _rand_pos(rng, 200, box_mat)
    s = mp.System(pos=pos, box=Box(box_mat, boundary=boundary))
    s.build_neighbor(3.0)
    _check(s, 3.0, sample_idx=list(range(0, 200, 17)))


def test_synthetic_triclinic():
    rng = np.random.default_rng(7)
    box_mat = np.array([[10.0, 0.0, 0.0],
                        [2.5, 9.0, 0.0],
                        [1.0, 1.5, 11.0]])
    pos = _rand_pos(rng, 400, box_mat)
    s = mp.System(pos=pos, box=Box(box_mat, boundary=[1, 1, 1]))
    s.build_neighbor(3.5)
    _check(s, 3.5, sample_idx=list(range(0, 400, 23)))


def test_synthetic_triclinic_open_z():
    """Triclinic with one open boundary — replication should not happen on z."""
    rng = np.random.default_rng(11)
    box_mat = np.array([[8.0, 0.0, 0.0],
                        [1.5, 9.0, 0.0],
                        [0.5, 0.7, 10.0]])
    pos = _rand_pos(rng, 250, box_mat)
    s = mp.System(pos=pos, box=Box(box_mat, boundary=[1, 1, 0]))
    s.build_neighbor(3.0)
    _check(s, 3.0, sample_idx=list(range(0, 250, 13)))


def test_very_small_cutoff_returns_no_neighbors():
    rng = np.random.default_rng(3)
    box_mat = np.diag([10.0, 10.0, 10.0])
    pos = _rand_pos(rng, 50, box_mat)
    s = mp.System(pos=pos, box=Box(box_mat, boundary=[1, 1, 1]))
    s.build_neighbor(0.05)
    assert s.neighbor_number.sum() == 0
    # And the brute-force says the same.
    pos_ref, box_ref = _ref_pos_and_box(s)
    for i in range(s.N):
        assert len(_bf_neighbors(i, pos_ref, box_ref, 0.05)[0]) == 0


def test_cutoff_larger_than_box_replicates():
    """Cutoff larger than half the smallest box vector triggers replication."""
    rng = np.random.default_rng(5)
    box_mat = np.diag([4.0, 4.0, 4.0])
    pos = _rand_pos(rng, 8, box_mat)
    s = mp.System(pos=pos, box=Box(box_mat, boundary=[1, 1, 1]))
    s.build_neighbor(5.0)
    assert hasattr(s, "_enlarge_data"), "expected internal replication"
    _check(s, 5.0, sample_idx=list(range(s.N)))


def test_two_atoms_exactly_at_cutoff_are_included():
    """mdapy's convention is `dist <= rc`; brute-force must agree."""
    pos = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    s = mp.System(pos=pos, box=Box(np.eye(3) * 100.0, boundary=[1, 1, 1]))
    s.build_neighbor(3.0)
    assert s.neighbor_number.tolist() == [1, 1]
    _check(s, 3.0, sample_idx=[0, 1])


def test_dense_random_orthogonal_full_check():
    """Small enough to verify EVERY atom, not just samples."""
    rng = np.random.default_rng(99)
    box_mat = np.diag([6.0, 6.0, 6.0])
    pos = _rand_pos(rng, 60, box_mat)
    s = mp.System(pos=pos, box=Box(box_mat, boundary=[1, 1, 1]))
    s.build_neighbor(2.5)
    _check(s, 2.5, sample_idx=list(range(s.N)))
