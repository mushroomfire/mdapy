# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""Polyhedral template matching (PTM) — fixture-driven, no OVITO at runtime.

The fixtures store the per-atom Structure Type from OVITO's PTM modifier
(default settings). Extra outputs (RMSD, orientation quaternion,
interatomic distance) are not currently exercised here; they are
covered indirectly by the invariant tests on perfect crystals.
"""

import numpy as np
import pytest

import mdapy as mp

from _fixture_helper import fixtures_with, fixture_ids, system_from_fixture

PATHS = fixtures_with("ptm")


@pytest.mark.parametrize("path", PATHS, ids=fixture_ids(PATHS))
def test_ptm_against_fixture(path):
    data = np.load(path)
    system = system_from_fixture(data)
    system.cal_polyhedral_template_matching()
    got = system.data["ptm"].to_numpy(allow_copy=False)
    expected = data["ptm"]
    n_diff = int(np.sum(got != expected))
    assert n_diff == 0, (
        f"{path.name}: {n_diff}/{len(got)} atoms classified differently"
    )


def test_ptm_perfect_crystals():
    """Self-contained invariant: PTM on perfect crystals returns the
    expected single label everywhere.

    PTM codes: 0=Other, 1=FCC, 2=HCP, 3=BCC, 4=ICO, 5=SC, 6=DCUB,
               7=DHEX, 8=Graphene.
    """
    fcc = mp.build_crystal("Al", "fcc", 4.05, nx=4, ny=4, nz=4)
    fcc.cal_polyhedral_template_matching()
    assert np.all(fcc.data["ptm"].to_numpy() == 1), "perfect FCC -> 1"

    bcc = mp.build_crystal("Fe", "bcc", 2.86, nx=4, ny=4, nz=4)
    bcc.cal_polyhedral_template_matching()
    assert np.all(bcc.data["ptm"].to_numpy() == 3), "perfect BCC -> 3"

    hcp = mp.build_crystal("Mg", "hcp", 3.21, nx=4, ny=4, nz=3)
    hcp.cal_polyhedral_template_matching()
    assert np.all(hcp.data["ptm"].to_numpy() == 2), "perfect HCP -> 2"

    diamond = mp.build_crystal("C", "diamond", 3.5, nx=3, ny=3, nz=3)
    diamond.cal_polyhedral_template_matching(structure="all")
    assert np.all(diamond.data["ptm"].to_numpy() == 6), "perfect diamond -> 6"
