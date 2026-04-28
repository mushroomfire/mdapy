# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""Bond analysis (length + angle histograms) — fixture-driven."""

import numpy as np

from mdapy import System
from _fixture_helper import load_misc, input_path


def test_bond():
    data = load_misc("bond_analysis")
    system = System(input_path("water.xyz"))
    bo = system.cal_bond_analysis(float(data["cutoff"]),
                                  int(data["bins"]),
                                  max_neigh=int(data["max_neigh"]))
    assert np.allclose(bo.r_length, data["r_length"], atol=1e-6)
    assert np.allclose(bo.bond_length_distribution,
                       data["bond_length_distribution"], atol=1e-6)
    assert np.allclose(bo.r_angle, data["r_angle"], atol=1e-6)
    assert np.allclose(bo.bond_angle_distribution,
                       data["bond_angle_distribution"], atol=1e-6)
