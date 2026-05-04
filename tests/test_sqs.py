# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""Tests for SQS (Special Quasirandom Structure) generation.

Layers of validation:

* Correlation calculation matches ATAT (pair *and* triplet).
* End-to-end MC convergence drives correlations below the random baseline.
* Cell / position / composition are preserved (only labels are reshuffled).
* Triclinic boxes are supported (cell shape matters only via the neighbor list).
* Small boxes (where mdapy.Neighbor internally replicates atoms) work for
  pair / triplet / quad enumeration.
* :meth:`SQS.is_sqs` returns True on a converged cubic SQS and False on a
  random alloy.
"""
import os
from collections import Counter

import numpy as np
import polars as pl
import pytest

import mdapy as mp


HERE = os.path.dirname(__file__)
ATAT_DIR = os.path.join(HERE, "input_files", "atat_sqs_n20")


def _atat_bestsqs_to_system():
    """Parse ATAT's bestsqs.out into an mdapy.System with positions wrapped."""
    lines = open(os.path.join(ATAT_DIR, "bestsqs.out")).read().splitlines()
    coord = np.array([list(map(float, lines[i].split())) for i in range(3)])
    lat = np.array([list(map(float, lines[i].split())) for i in range(3, 6)])
    pos, sp = [], []
    for line in lines[6:]:
        toks = line.split()
        if len(toks) < 4:
            continue
        pos.append(list(map(float, toks[:3])))
        sp.append(toks[3])
    pos = np.array(pos)
    pos_cart = pos @ coord.T
    lat_cart = coord @ lat
    inv = np.linalg.inv(lat_cart)
    frac = pos_cart @ inv
    frac -= np.floor(frac)
    pos_w = frac @ lat_cart
    species_list = sorted(set(sp))
    type_arr = np.array(
        [species_list.index(s) + 1 for s in sp], dtype=np.int32
    )
    df = pl.DataFrame({
        "id":      np.arange(len(pos)) + 1,
        "type":    type_arr,
        "element": sp,
        "x":       pos_w[:, 0],
        "y":       pos_w[:, 1],
        "z":       pos_w[:, 2],
    })
    return mp.System(data=df, box=lat_cart)


def _atat_corr_summary(body: int):
    """Mean/max of |corr| from ATAT bestcorr.out at the given cluster body order."""
    vals = []
    for line in open(os.path.join(ATAT_DIR, "bestcorr.out")):
        toks = line.split()
        if len(toks) < 4 or toks[0] != str(body):
            continue
        vals.append(float(toks[2]))
    a = np.abs(np.array(vals))
    return float(a.mean()), float(a.max())


def _build_triclinic_alloy(L=3.0, n=6, seed=0):
    """Build a triclinic 3-element simple-cubic alloy of size n^3."""
    box = np.array([
        [L * n, 0,           0],
        [L * 0.3 * n, L * n, 0],
        [L * 0.2 * n, L * 0.1 * n, L * n],
    ])
    N = n ** 3
    frac = np.array(
        [(i, j, k) for i in range(n) for j in range(n) for k in range(n)]
    ) / np.array([n, n, n])
    pos = frac @ box
    rng = np.random.default_rng(seed)
    elem = rng.choice(["A", "B", "C"], size=N)
    df = pl.DataFrame({
        "id":      np.arange(N) + 1,
        "type":    np.array([{"A": 1, "B": 2, "C": 3}[e] for e in elem],
                            dtype=np.int32),
        "element": elem,
        "x":       pos[:, 0],
        "y":       pos[:, 1],
        "z":       pos[:, 2],
    })
    return mp.System(data=df, box=box)


# ---------------------------------------------------------------------- tests

def test_pair_correlations_match_atat():
    """Our pair-correlation magnitudes should agree with ATAT's bestcorr.out."""
    sys_atat = _atat_bestsqs_to_system()
    sqs = mp.SQS(
        sys_atat, cutoffs={2: 1.05}, n_replicas=1, max_steps=0, seed=0
    ).compute()
    our_mean = float(np.abs(sqs.correlations).mean())
    our_max  = float(np.abs(sqs.correlations).max())
    atat_mean, atat_max = _atat_corr_summary(body=2)
    assert abs(our_mean - atat_mean) < 0.005, (
        f"pair |corr| mean mismatch: ours={our_mean:.4f} ATAT={atat_mean:.4f}"
    )
    assert our_max < atat_max + 0.02, (
        f"pair |corr| max suspiciously large: ours={our_max:.4f} ATAT={atat_max:.4f}"
    )


def test_triplet_correlations_match_atat():
    """Adding triplet clusters: small box gets replicated, enumeration works."""
    sys_atat = _atat_bestsqs_to_system()
    sqs = mp.SQS(
        sys_atat, cutoffs={2: 1.05, 3: 1.05},
        n_replicas=1, max_steps=0, seed=0,
    ).compute()
    body_count = Counter(ci["n_pts"] for ci in sqs.channel_info)
    assert 2 in body_count and 3 in body_count, "expected pair + triplet channels"

    trip_corr = np.array([
        ci["corr"] for ci in sqs.channel_info if ci["n_pts"] == 3
    ])
    atat_trip_mean, atat_trip_max = _atat_corr_summary(body=3)
    our_mean = float(np.abs(trip_corr).mean())
    assert abs(our_mean - atat_trip_mean) < 0.005, (
        f"triplet |corr| mean mismatch: ours={our_mean:.4f} ATAT={atat_trip_mean:.4f}"
    )


def test_quad_clusters_enumerated():
    """Quadruplet enumeration should produce the expected number of channels.

    For a 5-element fcc HEA, n_func = 4 → 4^4 = 256 function tuples per
    quad shell. We just check that the engine builds at least one quad
    shell and runs MC without crashing.
    """
    sys_init = mp.build_hea(
        ("Fe", "Ni", "Co", "Mn", "Cr"), (0.2,) * 5,
        "fcc", 3.55, nx=2, ny=2, nz=2, random_seed=0,
    )
    # Tight 4-body cutoff (just above NN1 ≈ 2.51 Å) to keep the
    # quadruplet enumeration cheap — we only need to verify that
    # 4-body channels are produced, not that they fully converge.
    sqs = mp.SQS(
        sys_init,
        cutoffs={2: 4.0, 3: 2.7, 4: 2.7},
        n_replicas=2, max_steps=2000, T=0.05, seed=0,
    ).compute()
    body_count = Counter(ci["n_pts"] for ci in sqs.channel_info)
    assert 4 in body_count, f"no 4-body channels: {dict(body_count)}"
    assert body_count[4] > 0


def test_small_box_triplet_enumeration():
    """Replicated (small) boxes must support triplet/quad enumeration."""
    # 1×1×5 cell of fcc; mdapy.Neighbor will replicate it into a larger cell.
    sys_init = mp.build_hea(
        ("A", "B", "C"), (1 / 3,) * 3, "fcc", 1.0,
        nx=1, ny=1, nz=5, random_seed=0,
    )
    # Just running compute() without raising NotImplementedError is the test.
    sqs = mp.SQS(
        sys_init, cutoffs={2: 1.05, 3: 1.05},
        n_replicas=2, max_steps=5000, T=0.1, seed=0,
    ).compute()
    body_count = Counter(ci["n_pts"] for ci in sqs.channel_info)
    assert 2 in body_count and 3 in body_count


def test_triclinic_box_runs():
    """SQS must work on a true triclinic cell (non-orthogonal box vectors)."""
    sys_tri = _build_triclinic_alloy(L=3.0, n=6, seed=0)
    sqs = mp.SQS(
        sys_tri, cutoffs={2: 4.0},
        n_replicas=4, max_steps=50000, T=0.02, seed=1,
    ).compute()
    # SQS should preserve cell + atom positions
    assert np.allclose(sqs.system.box.box, sys_tri.box.box)
    # Composition preserved
    assert (
        sys_tri.data["element"].value_counts().sort("element").equals(
            sqs.system.data["element"].value_counts().sort("element")
        )
    )
    # Some non-trivial correlation reduction should happen (cell is large
    # enough for the optimizer to find improvements)
    sqs0 = mp.SQS(sys_tri, cutoffs={2: 4.0}, n_replicas=1, max_steps=0).compute()
    assert (
        np.abs(sqs.correlations).mean() <= np.abs(sqs0.correlations).mean()
    )


def test_sqs_drives_correlations_down():
    """build_hea -> SQS -> correlations significantly smaller than random init."""
    sys_init = mp.build_hea(
        ("Fe", "Ni", "Co", "Mn", "Cr"), (0.2,) * 5,
        "fcc", 3.55, nx=3, ny=3, nz=3, random_seed=1,
    )
    ref = mp.SQS(sys_init, cutoffs={2: 2.7}, n_replicas=1, max_steps=0).compute()
    init_mean = float(np.abs(ref.correlations).mean())

    sqs = mp.SQS(
        sys_init, cutoffs={2: 2.7},
        n_replicas=4, max_steps=100000, T=0.02, seed=2,
    ).compute()
    after_mean = float(np.abs(sqs.correlations).mean())
    assert after_mean < 0.75 * init_mean, (
        f"SQS did not improve: init={init_mean:.4f}, after={after_mean:.4f}"
    )

    init_counts = sys_init.data["element"].value_counts().sort("element")
    out_counts  = sqs.system.data["element"].value_counts().sort("element")
    assert init_counts.equals(out_counts)


def test_sqs_preserves_cell_and_positions():
    """Only species labels change; cell and positions are untouched."""
    sys_init = mp.build_hea(
        ("A", "B", "C"), (1 / 3,) * 3, "bcc", 2.87,
        nx=3, ny=3, nz=3, random_seed=42,
    )
    sqs = mp.SQS(
        sys_init, cutoffs={2: 3.5},
        n_replicas=2, max_steps=20000, T=0.05, seed=0,
    ).compute()
    assert sqs.system.N == sys_init.N
    assert np.allclose(sqs.system.box.box, sys_init.box.box)
    for col in ("x", "y", "z"):
        np.testing.assert_array_equal(
            sqs.system.data[col].to_numpy(),
            sys_init.data[col].to_numpy(),
        )


def test_atat_objective_rewards_dmin():
    """ATAT objective should be lower (more negative) than 'abs' on a good SQS,
    because the d1 perfect-match reward subtracts a positive quantity."""
    sys_init = mp.build_hea(
        ("Fe", "Ni", "Co", "Mn", "Cr"), (0.2,) * 5,
        "fcc", 3.55, nx=2, ny=2, nz=2, random_seed=1,
    )
    common = dict(
        cutoffs={2: 4.0, 3: 3.0},
        n_replicas=2, max_steps=10000, T=0.02, seed=3,
    )
    sqs_atat = mp.SQS(sys_init, objective="atat", **common).compute()
    sqs_abs  = mp.SQS(sys_init, objective="abs",  **common).compute()
    # ATAT objective subtracts the d1 reward → strictly negative on success
    assert sqs_atat.objective < 0.0, (
        f"ATAT objective should be negative on a converged SQS, got {sqs_atat.objective}"
    )
    # 'abs' objective is sum of non-negative values → non-negative
    assert sqs_abs.objective >= 0.0


def test_is_sqs_true_on_converged_cubic():
    """A 256-atom cubic 3-element SQS should pass the verdict (absolute
    correlation residual + Warren-Cowley)."""
    sys_init = mp.build_hea(
        ("A", "B", "C"), (1 / 3,) * 3, "fcc", 3.6,
        nx=4, ny=4, nz=4, random_seed=0,
    )
    sqs = mp.SQS(
        sys_init, cutoffs={2: 4.0, 3: 3.0, 4: 3.0},
        n_replicas=8, max_steps=200000, T=0.02, seed=1,
    ).compute()
    verdict, report = sqs.is_sqs(
        tol=0.05, n_random=30, return_report=True, seed=42,
    )
    assert verdict, f"expected SQS verdict True; report={report}"
    assert report["absolute"]["pass"]
    assert report["warren_cowley"]["pass"]


def test_is_sqs_false_on_random_alloy():
    """An unoptimized random alloy at small N should NOT pass is_sqs(tol=0.02)."""
    sys_init = mp.build_hea(
        ("Fe", "Ni", "Co", "Mn", "Cr"), (0.2,) * 5,
        "fcc", 3.55, nx=2, ny=2, nz=2, random_seed=1,
    )
    # Diagnostic mode: no MC, just evaluate the random init.
    sqs = mp.SQS(sys_init, cutoffs={2: 4.0}, max_steps=0, n_replicas=1).compute()
    verdict = sqs.is_sqs(tol=0.02, n_random=20, seed=0)
    assert not verdict, "32-atom random alloy should not pass tol=0.02"
