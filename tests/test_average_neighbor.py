# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""Average-by-neighbor (per-atom property smoothing) — fixture-driven."""

import numpy as np
import pytest

import mdapy as mp
from _fixture_helper import load_misc, input_path

CONFIGS = ["rec_box_big", "tri_box_big"]


@pytest.mark.parametrize("name", CONFIGS)
def test_average_neighbor(name):
    data = load_misc("average_neighbor")
    rc = float(data[f"{name}__cutoff"])
    expected = data[f"{name}__x_ave"]

    system = mp.System(input_path(f"{name}.xyz"))
    system.average_by_neighbor(rc, "x", include_self=True)
    got = system.data["x_ave"].to_numpy(allow_copy=False)

    assert np.allclose(got, expected, atol=1e-6), (
        f"{name}: max |Δ| = {np.abs(got - expected).max():.3g}"
    )
