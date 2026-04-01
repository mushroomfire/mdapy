# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
import mdapy as mp
from ovito.modifiers import (
    IdentifyFCCPlanarFaultsModifier,
    PolyhedralTemplateMatchingModifier,
)
import numpy as np


def test_fcc_pft():
    system = mp.System("input_files/ISF.dump")
    ovi_atom = system.to_ovito()

    system.cal_polyhedral_template_matching(
        "all", identify_fcc_planar_faults=True, identify_esf=False
    )
    ovi_atom.apply(
        PolyhedralTemplateMatchingModifier(
            output_orientation=True, output_interatomic_distance=True
        )
    )
    ovi_atom.apply(IdentifyFCCPlanarFaultsModifier())

    assert np.allclose(
        system.data["pft"].to_numpy(allow_copy=False),
        ovi_atom.particles["Planar Fault Type"][...],
    ), "pft is wrong."
