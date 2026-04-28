# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""Common-neighbor-parameter (CNP) — fixture-driven, no external deps.

OVITO has no exact CNP equivalent, so the fixtures are anchored to the
*current* mdapy output. They serve as a regression guard: any accidental
change in the algorithm will trip the test, and the perfect-crystal
invariant tests below provide an absolute correctness baseline."""

import numpy as np
import pytest

import mdapy as mp

from _fixture_helper import fixtures_with, fixture_ids, system_from_fixture

PATHS = fixtures_with("cnp")


@pytest.mark.parametrize("path", PATHS, ids=fixture_ids(PATHS))
def test_cnp_against_fixture(path):
    data = np.load(path)
    system = system_from_fixture(data)
    system.cal_common_neighbor_parameter(float(data["cnp_cutoff"]))
    got = system.data["cnp"].to_numpy(allow_copy=False)
    expected = data["cnp"]
    assert np.allclose(got, expected, atol=1e-6, rtol=1e-6), (
        f"{path.name}: max |Δ| = {np.abs(got - expected).max():.3g}"
    )


def test_cnp_perfect_fcc_is_zero():
    a = 3.615
    s = mp.build_crystal("Cu", "fcc", a, nx=4, ny=4, nz=4)
    s.cal_common_neighbor_parameter(0.86 * a)
    assert np.allclose(s.data["cnp"].max(), 0.0)


def test_cnp_perfect_bcc_is_zero():
    a = 3.615
    s = mp.build_crystal("Cu", "bcc", a, nx=4, ny=4, nz=4)
    s.cal_common_neighbor_parameter(1.21 * a)
    assert np.allclose(s.data["cnp"].max(), 0.0)


def test_cnp_perfect_hcp_known_value():
    """Known reference value: HCP at c/a=1.633 has CNP = 8.71215."""
    a = 3.615
    s = mp.build_crystal("Cu", "hcp", a, c_over_a=1.633)
    s.cal_common_neighbor_parameter(1.21 * a)
    assert np.allclose(s.data["cnp"].max(), 8.71215)
