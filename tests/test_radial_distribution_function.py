# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""Radial distribution function (partial RDFs) — fixture-driven."""

import numpy as np

import mdapy as mp
from _fixture_helper import load_misc, input_path


def test_rdf():
    data = load_misc("rdf")
    system = mp.System(input_path("AlCrNi.xyz"))
    rdf = system.cal_radial_distribution_function(
        float(data["cutoff"]), int(data["nbins"]))
    elements = list(data["elements"])
    K = len(elements)
    g_ref = data["g"]
    # g_partial is keyed by element-pair tuples (e.g. ("Al", "Cr"))
    for i in range(K):
        for j in range(i, K):
            assert np.allclose(
                rdf.g_partial[(elements[i], elements[j])], g_ref[i, j], atol=1e-6
            ), f"{elements[i]}-{elements[j]} RDF differs"
