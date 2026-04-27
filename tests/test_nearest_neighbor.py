# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""
Brute-force verification of `build_nearest_neighbor`.

mdapy's KNN treats periodic images of the same atom as distinct neighbors
(same convention as Ovito's NearestNeighborFinder), so the brute-force
reference enumerates all periodic images within enough shells to be
guaranteed to dominate the k-th nearest distance.
"""

import numpy as np
import pytest

import mdapy as mp
from mdapy.box import Box

# mdapy enforces k < 25 in src/mdapy/knn.py
K_VALUES = [1, 4, 8, 12, 24]


# ---------------------------------------------------------------------------
# Brute-force reference (with explicit periodic images)
# ---------------------------------------------------------------------------

def _bf_full_image_set(idx, pos, box, n_shells=3):
    """Return (distances, atom_indices) for every periodic image of every
    atom (self excluded) within the enumerated shell range."""
    pbc = np.asarray(box.boundary, dtype=int)
    rngs = [range(-n_shells, n_shells + 1) if pbc[a] else range(1) for a in range(3)]
    base = pos - pos[idx]
    all_d, all_i = [], []
    for nx in rngs[0]:
        for ny in rngs[1]:
            for nz in rngs[2]:
                shift = nx * box.box[0] + ny * box.box[1] + nz * box.box[2]
                d = np.linalg.norm(base + shift, axis=1)
                if nx == 0 and ny == 0 and nz == 0:
                    d = d.copy()
                    d[idx] = np.inf
                all_d.append(d)
                all_i.append(np.arange(pos.shape[0], dtype=np.int64))
    return np.concatenate(all_d), np.concatenate(all_i)


def _bf_knn(idx, pos, box, k, n_shells=3):
    """Return the k nearest (image_atom_index, distance) pairs for atom `idx`.

    Periodic images are enumerated up to ±n_shells along each periodic axis.
    A safety assertion checks that the k-th distance is strictly inside the
    enumerated region, so we know we did not miss a closer image.
    """
    pbc = np.asarray(box.boundary, dtype=int)
    rngs = [range(-n_shells, n_shells + 1) if pbc[a] else range(1) for a in range(3)]

    base = pos - pos[idx]
    all_d, all_i = [], []
    for nx in rngs[0]:
        for ny in rngs[1]:
            for nz in rngs[2]:
                shift = nx * box.box[0] + ny * box.box[1] + nz * box.box[2]
                rij = base + shift
                d = np.linalg.norm(rij, axis=1)
                if nx == 0 and ny == 0 and nz == 0:
                    d = d.copy()
                    d[idx] = np.inf  # exclude self
                all_d.append(d)
                all_i.append(np.arange(pos.shape[0], dtype=np.int64))

    all_d = np.concatenate(all_d)
    all_i = np.concatenate(all_i)
    order = np.argsort(all_d, kind="stable")[:k]
    sel_d = all_d[order]
    sel_i = all_i[order]

    # Safety: the n-th-shell boundary distance along any periodic axis is
    # n_shells * thickness; the kth distance must be safely under it.
    thick = box.get_thickness()
    safe_radius = n_shells * thick.min()
    assert sel_d[-1] < safe_radius, (
        f"brute-force KNN may be missing further images for atom {idx}: "
        f"k-th distance {sel_d[-1]} >= safe radius {safe_radius}"
    )
    return sel_i, sel_d


def _check_knn(system, k, sample_idx=None, atol=1e-6, n_shells=3):
    if hasattr(system, "_enlarge_data"):
        pos = system._enlarge_data.select(["x", "y", "z"]).to_numpy()
        box = system._enlarge_box
    else:
        pos = system.data.select(["x", "y", "z"]).to_numpy()
        box = system.box

    if sample_idx is None:
        N_orig = system.N
        sample_idx = sorted(set(
            [0, N_orig // 4, N_orig // 2, (3 * N_orig) // 4, N_orig - 1]
        ))

    for i in sample_idx:
        ref_idx, ref_dist = _bf_knn(i, pos, box, k, n_shells)
        my_dist = system.distance_list[i, :k]
        my_idx = system.verlet_list[i, :k]

        # mdapy returns sorted distances.
        assert np.all(np.diff(my_dist) >= -atol), (
            f"atom {i}: returned distances are not sorted"
        )
        # Distances must match (multiset, sorted).
        assert np.allclose(np.sort(my_dist), np.sort(ref_dist), atol=atol), (
            f"atom {i}: knn distance set differs\n"
            f"  mdapy: {np.sort(my_dist)}\n"
            f"  brute: {np.sort(ref_dist)}"
        )

        # Per-atom check: every returned atom index must have a periodic
        # image whose distance matches the reported one. This tolerates
        # ties in the k-th shell (mdapy and brute-force may pick different
        # equidistant images).
        all_d_full, all_i_full = _bf_full_image_set(i, pos, box, n_shells)
        for j_ret, d_ret in zip(my_idx, my_dist):
            d_for_j = all_d_full[all_i_full == j_ret]
            assert np.any(np.abs(d_for_j - d_ret) <= atol), (
                f"atom {i}: returned neighbor {j_ret}@{d_ret} has no matching"
                f" periodic image (closest images at {np.sort(d_for_j)[:5]})"
            )


# ---------------------------------------------------------------------------
# File-driven tests
# ---------------------------------------------------------------------------

FILES = [
    "tests/input_files/HexDiamond.xyz",
    "tests/input_files/tri_box_small.xyz",
    "tests/input_files/rec_box_small.xyz",
    "tests/input_files/rec_box_big.xyz",
    "tests/input_files/AlCrNi.xyz",
    "tests/input_files/tri_box_big.xyz",
]


@pytest.mark.parametrize("filename", FILES)
@pytest.mark.parametrize("k", K_VALUES)
def test_files(filename, k):
    s = mp.System(filename)
    # Degenerate: k <= N AND N == 1 means there are no real neighbors and
    # mdapy never replicates (replication only triggers when k > N), so it
    # returns a -1 sentinel. Skip that single combination.
    if s.N == 1 and k <= s.N:
        pytest.skip("single-atom system with k<=N returns no neighbor")
    s.build_nearest_neighbor(k)
    _check_knn(s, k)


# ---------------------------------------------------------------------------
# Synthetic tests
# ---------------------------------------------------------------------------

def _rand_pos(rng, n, box_mat):
    return rng.random((n, 3)) @ box_mat


@pytest.mark.parametrize("boundary", [
    [1, 1, 1],
    [0, 0, 0],
    [1, 1, 0],
    [1, 0, 0],
])
@pytest.mark.parametrize("k", [1, 6, 12, 24])
def test_synthetic_orthogonal(boundary, k):
    rng = np.random.default_rng(2026)
    box_mat = np.diag([12.0, 13.5, 11.0])
    pos = _rand_pos(rng, 300, box_mat)
    s = mp.System(pos=pos, box=Box(box_mat, boundary=boundary))
    s.build_nearest_neighbor(k)
    _check_knn(s, k, sample_idx=list(range(0, 300, 19)))


@pytest.mark.parametrize("k", [4, 12, 24])
def test_synthetic_triclinic(k):
    rng = np.random.default_rng(31)
    box_mat = np.array([[10.0, 0.0, 0.0],
                        [3.0, 9.0, 0.0],
                        [1.5, 2.0, 11.0]])
    pos = _rand_pos(rng, 400, box_mat)
    s = mp.System(pos=pos, box=Box(box_mat, boundary=[1, 1, 1]))
    s.build_nearest_neighbor(k)
    _check_knn(s, k, sample_idx=list(range(0, 400, 23)))


def test_small_box_replication():
    """k larger than N forces internal replication."""
    rng = np.random.default_rng(0)
    box_mat = np.diag([3.5, 3.5, 3.5])
    pos = _rand_pos(rng, 4, box_mat)
    s = mp.System(pos=pos, box=Box(box_mat, boundary=[1, 1, 1]))
    s.build_nearest_neighbor(20)
    _check_knn(s, 20, sample_idx=list(range(s.N)))


def test_perfect_fcc_first_shell_is_12():
    """In a perfect FCC lattice the 12 nearest neighbors are equidistant."""
    a = 4.05
    n = 5
    base = np.array([[0, 0, 0], [0.5, 0.5, 0],
                     [0.5, 0, 0.5], [0, 0.5, 0.5]]) * a
    cells = np.stack(np.meshgrid(np.arange(n), np.arange(n), np.arange(n),
                                 indexing="ij"), axis=-1).reshape(-1, 3) * a
    pos = (cells[:, None, :] + base[None, :, :]).reshape(-1, 3)
    s = mp.System(pos=pos, box=Box(np.eye(3) * (n * a), boundary=[1, 1, 1]))
    s.build_nearest_neighbor(12)
    expected = a / np.sqrt(2.0)
    assert np.allclose(s.distance_list, expected, atol=1e-6), (
        "FCC nearest-neighbor distances should all be a/sqrt(2)"
    )
    _check_knn(s, 12, sample_idx=list(range(0, s.N, 7)))
