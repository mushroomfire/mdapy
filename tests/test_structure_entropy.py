# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""Structure entropy — fixture-driven, no OVITO at runtime."""

import numpy as np
import pytest

import mdapy as mp
from _fixture_helper import load_misc, input_path

CONFIGS = ["rec_box_big", "rec_box_small", "tri_box_big", "tri_box_small"]
MODES = ["default", "use_local_density", "compute_average"]


@pytest.mark.parametrize("name", CONFIGS)
@pytest.mark.parametrize("mode", MODES)
def test_structure_entropy(name, mode):
    data = load_misc("structure_entropy")
    expected = data[f"{name}__{mode}"]

    system = mp.System(input_path(f"{name}.xyz"))
    if mode == "compute_average":
        system.cal_structure_entropy(5.0, 0.2, False, average_rc=4.0)
        got = system.data["entropy_ave"].to_numpy()
    elif mode == "use_local_density":
        system.cal_structure_entropy(5.0, 0.2, True)
        got = system.data["entropy"].to_numpy()
    else:
        system.cal_structure_entropy(5.0, 0.2, False)
        got = system.data["entropy"].to_numpy()

    assert np.allclose(got, expected, atol=1e-6), (
        f"{name}/{mode}: max |Δ| = {np.abs(got - expected).max():.3g}"
    )
