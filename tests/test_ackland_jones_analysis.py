# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""Ackland-Jones analysis (AJA) — fixture-driven, no OVITO at runtime."""

import numpy as np
import pytest

import mdapy as mp

from _fixture_helper import fixtures_with, fixture_ids, system_from_fixture

PATHS = fixtures_with("aja")


@pytest.mark.parametrize("path", PATHS, ids=fixture_ids(PATHS))
def test_aja_against_fixture(path):
    data = np.load(path)
    system = system_from_fixture(data)
    system.cal_ackland_jones_analysis()
    got = system.data["aja"].to_numpy(allow_copy=False)
    expected = data["aja"]
    n_diff = int(np.sum(got != expected))
    assert n_diff == 0, (
        f"{path.name}: {n_diff}/{len(got)} atoms classified differently"
    )


def test_aja_perfect_crystals():
    """Self-contained sanity check on a perfect FCC supercell."""
    system = mp.build_crystal("Al", "fcc", 4.05, nx=4, ny=4, nz=4)
    system.cal_ackland_jones_analysis()
    # AJA codes: 1 = FCC. Surface-free perfect FCC should be all 1.
    assert np.all(system.data["aja"].to_numpy() == 1), "perfect FCC should be type 1"
