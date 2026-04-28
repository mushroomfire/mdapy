# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""
Common-neighbor-analysis (CNA) tests — fixture-driven, no OVITO at runtime.
Reference labels (Structure Type) come from OVITO's
`CommonNeighborAnalysisModifier` in fixed-cutoff mode.
"""

import numpy as np
import pytest

import mdapy as mp

from _fixture_helper import fixtures_with, fixture_ids, system_from_fixture

PATHS = fixtures_with("cna")


@pytest.mark.parametrize("path", PATHS, ids=fixture_ids(PATHS))
def test_cna_against_fixture(path):
    data = np.load(path)
    system = system_from_fixture(data)
    system.cal_common_neighbor_analysis(rc=float(data["cna_cutoff"]))
    got = system.data["cna"].to_numpy(allow_copy=False)
    expected = data["cna"]
    n_diff = int(np.sum(got != expected))
    assert n_diff == 0, (
        f"{path.name}: {n_diff}/{len(got)} atoms classified differently"
    )


def test_cna_perfect_crystals():
    """Self-contained: perfect crystals should be uniformly classified."""
    a = 4.05
    fcc = mp.build_crystal("Al", "fcc", a, nx=4, ny=4, nz=4)
    fcc.cal_common_neighbor_analysis(rc=0.854 * a)
    assert np.all(fcc.data["cna"].to_numpy() == 1), "perfect FCC -> type 1"

    bcc = mp.build_crystal("Fe", "bcc", 2.86, nx=4, ny=4, nz=4)
    bcc.cal_common_neighbor_analysis(rc=1.21 * 2.86)
    assert np.all(bcc.data["cna"].to_numpy() == 3), "perfect BCC -> type 3"

    hcp = mp.build_crystal("Mg", "hcp", 3.21, nx=4, ny=4, nz=3)
    hcp.cal_common_neighbor_analysis(rc=1.207 * 3.21)
    assert np.all(hcp.data["cna"].to_numpy() == 2), "perfect HCP -> type 2"
