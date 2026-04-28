# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""Per-element-triplet angular distribution function — fixture-driven."""

import numpy as np

from mdapy import System
from _fixture_helper import load_misc, input_path

# mdapy's index → OVITO histogram component name. mdapy returns the
# histograms in a fixed dict-key order corresponding to the dict that the
# user passed in; the original test maps each index to a name.
_MDAPY_INDEX_TO_NAME = {
    0: "H-O-H",
    1: "O-O-H",
    2: "H-H-H",
    3: "O-H-O",
    4: "O-O-O",
    5: "O-H-H",
}


def test_adf():
    data = load_misc("adf")
    system = System(input_path("water.xyz"))
    adf = system.cal_angular_distribution_function(
        {
            "O-H-H": [0, 2.0, 0, 2.0],
            "O-O-H": [0, 2.0, 0, 2.0],
            "H-H-H": [0, 2.0, 0, 2.0],
            "H-O-O": [0, 2.0, 0, 2.0],
            "O-O-O": [0, 2.0, 0, 2.0],
            "H-O-H": [0, 2.0, 0, 2.0],
        },
        int(data["bins"]),
    )

    for mdapy_idx, name in _MDAPY_INDEX_TO_NAME.items():
        key = f"adf_{name.replace('-', '_')}"
        if key in data.files:
            assert np.allclose(
                adf.bond_angle_distribution[mdapy_idx],
                data[key],
                atol=1e-6,
            ), f"{name} ADF differs"
