# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""FCC planar fault identification (per-atom labels) — fixture-driven."""

import numpy as np

import mdapy as mp
from _fixture_helper import load_misc, input_path


def test_fcc_pft():
    data = load_misc("fcc_planar_faults")
    system = mp.System(input_path("ISF.dump"))
    system.cal_polyhedral_template_matching(
        "all", identify_fcc_planar_faults=True, identify_esf=False)
    got = system.data["pft"].to_numpy(allow_copy=False)
    expected = data["pft"]
    assert np.array_equal(got, expected), (
        f"PFT differs ({np.sum(got != expected)}/{len(got)} mismatches)"
    )
