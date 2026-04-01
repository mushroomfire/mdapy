# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
import mdapy as mp
from ovito.modifiers import CentroSymmetryModifier
import numpy as np


def test_csp():
    for filename in [
        "AlCrNi.xyz",
        "rec_box_small.xyz",
        "tri_box_small.xyz",
        "rec_box_big.xyz",
    ]:
        system = mp.System("input_files/" + filename)
        ovi_atom = system.to_ovito()
        ovi_atom.apply(CentroSymmetryModifier(num_neighbors=12))
        system.cal_centro_symmetry_parameter(12)
        assert np.allclose(
            system.data["csp"].to_numpy(allow_copy=False),
            ovi_atom.particles["Centrosymmetry"][...],
        ), f"{filename} is wrong."
