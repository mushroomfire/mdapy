# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""Tests for mdapy.BondStiffness — bond stiffness vs bond length, modeled
after ATAT *fitsvsl*.

We use the bundled UNEP-v1 NEP4 alloy potential as the force calculator and
verify:

* Per-element-pair longitudinal / transverse stiffnesses match ATAT
  *fitsvsl* output to high precision when ``poly_order=0`` (single
  lattice constant). The reference numbers below are pre-computed from a
  prior ``fitsvsl -f -op=0`` run on the same NEP-derived forces.
* The output ``slspring.out`` file format is parseable by the same
  reader rules ATAT uses (one line per element pair, then 2 polynomial
  blocks).
* :meth:`generate_perturbed_structures` produces ATAT-compatible
  ``str_ideal.out`` / ``str_unpert.out`` / ``str.out`` files.
"""
import os
import shutil
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import mdapy as mp
from _fixture_helper import input_path


def _nep():
    return mp.NEP(input_path("UNEP-v1.txt"))


def test_bond_stiffness_pure_al_matches_atat():
    """fcc Al, poly_order=0, n_lattice=1: matches ATAT fitsvsl to 4 decimals."""
    sys_ = mp.build_crystal("Al", "fcc", a=4.05, nx=2, ny=2, nz=2)
    bsl = mp.BondStiffness(
        sys_, calculator=_nep(),
        rc_bond=3.0, delta=0.01, poly_order=0, n_lattice=1,
        central_diff=True,
    ).compute()
    # Reference values from ATAT fitsvsl -f -op=0 on the same NEP forces.
    ref_kl = 1.24587
    ref_kt = -0.06445
    # single shell → key is (Al, Al, 0)
    np.testing.assert_allclose(
        bsl.k_long[("Al", "Al", 0)][0], ref_kl, atol=1e-4,
        err_msg="k_long does not match ATAT fitsvsl",
    )
    np.testing.assert_allclose(
        bsl.k_trans[("Al", "Al", 0)][0], ref_kt, atol=1e-4,
        err_msg="k_trans does not match ATAT fitsvsl",
    )


def test_bond_stiffness_binary_alcu_matches_atat():
    """Binary AlCu (50:50) fcc: per-element-pair stiffnesses match ATAT."""
    sys_ = mp.build_hea(
        ("Al", "Cu"), (0.5, 0.5), "fcc", a=3.85,
        nx=2, ny=2, nz=2, random_seed=1,
    )
    bsl = mp.BondStiffness(
        sys_, calculator=_nep(),
        rc_bond=2.95, delta=0.01, poly_order=0, n_lattice=1,
        central_diff=True,
    ).compute()
    # Reference from ATAT fitsvsl -f -op=0 on the same NEP forces.
    refs = {
        ("Al", "Al"): (2.35828, -0.15451),
        ("Al", "Cu"): (0.94644, -0.01820),
        ("Cu", "Cu"): (0.92183, -0.01896),
    }
    for pair, (ref_kl, ref_kt) in refs.items():
        key = (pair[0], pair[1], 0)
        np.testing.assert_allclose(
            bsl.k_long[key][0], ref_kl, atol=1e-4,
            err_msg=f"{pair} k_long mismatch",
        )
        np.testing.assert_allclose(
            bsl.k_trans[key][0], ref_kt, atol=1e-4,
            err_msg=f"{pair} k_trans mismatch",
        )


def test_slspring_output_format(tmp_path):
    """slspring.out is parseable: per pair, one (element_a element_b) line,
    then two polynomial blocks (k_long, k_trans)."""
    sys_ = mp.build_crystal("Al", "fcc", a=4.05, nx=2, ny=2, nz=2)
    bsl = mp.BondStiffness(
        sys_, calculator=_nep(),
        rc_bond=3.0, delta=0.01, poly_order=0, n_lattice=1,
    ).compute()
    out = tmp_path / "slspring.out"
    bsl.write_slspring(str(out))
    text = out.read_text().splitlines()
    assert text[0] == "Al Al"
    assert text[1] == "1"
    # k_long c0
    float(text[2])
    assert text[3] == "1"
    # k_trans c0
    float(text[4])


def test_generate_perturbed_structures_atat_format(tmp_path):
    """Each subdirectory contains the three ATAT structure files needed
    by fitsvsl -f."""
    sys_ = mp.build_crystal("Al", "fcc", a=4.05, nx=2, ny=2, nz=2)
    bsl = mp.BondStiffness(
        sys_, calculator=_nep(),
        rc_bond=3.0, delta=0.01, poly_order=0, n_lattice=1, central_diff=False,
    )
    out_dir = tmp_path / "train"
    perts = bsl.generate_perturbed_structures(output_dir=str(out_dir))
    # forward differences only → 3N perturbations (32 atoms × 3 axes)
    assert len(perts) == sys_.N * 3
    # each pNNNNN dir has the three ATAT-format files
    for sub in sorted(p for p in out_dir.glob("p*") if p.is_dir()):
        assert (sub / "str.out").exists()
        assert (sub / "str_ideal.out").exists()
        assert (sub / "str_unpert.out").exists()


def test_bcc_two_shells():
    """BCC W (NN1 ≈ a√3/2, NN2 = a) gets split into two distinct shells
    when rc_bond covers both. The fit produces one (k_l, k_t) polynomial
    per (element pair, shell)."""
    a = 3.165
    sys_ = mp.build_crystal("W", "bcc", a=a, nx=2, ny=2, nz=2)
    bsl = mp.BondStiffness(
        sys_, calculator=_nep(),
        rc_bond=3.6,         # NN1 = 2.74 Å, NN2 = 3.165 Å — both included
        shell_tol=0.1,
        delta=0.01, poly_order=0, n_lattice=1,
        central_diff=True,
    ).compute()
    assert len(bsl.shells) == 2, f"expected 2 shells, got {bsl.shells}"
    # NN1 < NN2
    assert bsl.shells[0] < bsl.shells[1]
    # both shells have a fitted (k_l, k_t) for the W-W pair
    assert ("W", "W", 0) in bsl.k_long
    assert ("W", "W", 1) in bsl.k_long


def test_symmetric_strain_range():
    """n_lattice=3, max_strain=0.02 → strains [-0.02, 0, +0.02]."""
    sys_ = mp.build_crystal("Al", "fcc", a=4.05, nx=2, ny=2, nz=2)
    bsl = mp.BondStiffness(
        sys_, calculator=_nep(),
        rc_bond=3.0, delta=0.01, poly_order=1,
        n_lattice=3, max_strain=0.02,
    ).compute()
    strains = sorted(set(bsl.bond_table["strain"].to_list()))
    assert len(strains) == 3
    np.testing.assert_allclose(strains, [-0.02, 0.0, 0.02], atol=1e-12)


def test_bond_table_columns():
    """bond_table is a polars DataFrame with the documented schema."""
    sys_ = mp.build_crystal("Al", "fcc", a=4.05, nx=2, ny=2, nz=2)
    bsl = mp.BondStiffness(
        sys_, calculator=_nep(),
        rc_bond=3.0, delta=0.01, poly_order=0, n_lattice=1,
    ).compute()
    cols = set(bsl.bond_table.columns)
    assert {"element_a", "element_b", "shell", "r", "k_long", "k_trans", "strain"} <= cols
    # all r values are positive
    assert (bsl.bond_table["r"].to_numpy() > 0).all()
    # one shell on a pure-Al fcc test (NN1 only within rc=3.0)
    assert len(bsl.shells) == 1
