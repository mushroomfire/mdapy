# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""Wigner-Seitz defect analysis — fixture-driven, no OVITO at runtime."""

import numpy as np

from mdapy import System, WignerSeitzAnalysis
from _fixture_helper import load_misc, input_path


def test_wsd():
    data = load_misc("wigner_seitz")
    ref = System(input_path("hea.0.xyz"))
    cur = System(input_path("hea.1.xyz"))
    ws = WignerSeitzAnalysis(ref, True)
    res = ws.compute(cur)

    assert int(data["vacancy_count"]) == res["vacancy_count"]
    assert int(data["interstitial_count"]) == res["interstitial_count"]
    assert np.array_equal(data["site_occupancy"], res["site_occupancy"])
    assert np.array_equal(data["atom_occupancy"], res["atom_occupancy"])
    assert np.array_equal(data["atom_site_index"], res["atom_site_index"])
    for ovi_t, my_t in zip(data["atom_site_type"], res["atom_site_type"]):
        assert ovi_t == my_t
