# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the GPL-3.0-or-later.
"""Wigner-Seitz defect analysis — fixture-driven, no OVITO at runtime."""

import numpy as np

from mdapy import Box, System, WignerSeitzAnalysis
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


def test_affine_mapping_in_a_box_that_moved():
    """A box grown by 10 % about its center, as at constant pressure, so that
    its origin moves: with the affine mapping, no defect."""
    a = 3.6
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    cells = np.stack(np.meshgrid(*[np.arange(6)] * 3, indexing="ij"), -1).reshape(-1, 1, 3)
    pos = ((cells + basis) * a).reshape(-1, 3) + 0.1
    ref = System(pos=pos, box=Box(np.eye(3) * 6 * a, origin=[0.0, 0.0, 0.0]))
    center = np.full(3, 3 * a)
    grown = center + 1.1 * (pos - center)
    origin = center - 1.1 * center
    cur = System(pos=grown, box=Box(np.eye(3) * 6.6 * a, origin=origin))
    res = WignerSeitzAnalysis(ref, True).compute(cur)
    assert res["vacancy_count"] == 0 and res["interstitial_count"] == 0
    np.testing.assert_array_equal(res["atom_site_index"], np.arange(len(pos)))
    assert WignerSeitzAnalysis(ref, False).compute(cur)["vacancy_count"] > 0
