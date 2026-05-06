# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""Static structure factor (direct + Debye, partial + total) — fixture-driven."""

import numpy as np
import polars as pl
import pytest

import mdapy as mp
from _fixture_helper import load_misc


@pytest.mark.parametrize("mode", ["direct", "debye"])
def test_sfc(mode):
    data = load_misc("structure_factor")
    N = int(data["N"])
    nbins = int(data["nbins"])
    k_min = float(data["k_min"])
    k_max = float(data["k_max"])
    box = data["box"]
    points = data["points"]

    system = mp.System(box=box, pos=points)
    system.update_data(
        system.data.with_columns(
            pl.lit(np.array([1] * (N // 2) + [2] * (N // 2))).alias("type")
        )
    )

    sf1 = system.cal_structure_factor(k_min, k_max, nbins, cal_partial=True, mode=mode)
    atol = 1e-4

    # Sk_partial keys are element/type tuples (e.g. (1, 1)).
    assert np.allclose(sf1.Sk_partial[(1, 1)], data[f"{mode}_11"], atol=atol, equal_nan=True), (
        f"(1, 1) differs in {mode} mode"
    )
    assert np.allclose(sf1.Sk_partial[(1, 2)], data[f"{mode}_12"], atol=atol, equal_nan=True), (
        f"(1, 2) differs in {mode} mode"
    )
    assert np.allclose(sf1.Sk_partial[(2, 2)], data[f"{mode}_22"], atol=atol, equal_nan=True), (
        f"(2, 2) differs in {mode} mode"
    )
    assert np.allclose(sf1.Sk, data[f"{mode}_all"], atol=atol, equal_nan=True), (
        f"all differs in {mode} mode (partial=True)"
    )

    sf2 = system.cal_structure_factor(k_min, k_max, nbins, cal_partial=False, mode=mode)
    assert np.allclose(sf2.Sk, data[f"{mode}_all"], atol=atol, equal_nan=True), (
        f"all differs in {mode} mode (partial=False)"
    )
