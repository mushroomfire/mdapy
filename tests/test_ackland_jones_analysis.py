import mdapy as mp
from ovito.modifiers import AcklandJonesModifier
import numpy as np


def test_aja():
    for filename in [
        "hea.0.xyz",
        "Ti.poscar",
        "Mo.xyz",
        "rec_box_small.xyz",
        "AlCrNi.xyz",
        "tri_box_small.xyz",
        "rec_box_big.xyz",
    ]:
        system = mp.System("input_files/" + filename)
        ovi_atom = system.to_ovito()
        system.cal_ackland_jones_analysis()
        ovi_atom.apply(AcklandJonesModifier())
        assert np.allclose(
            system.data["aja"].to_numpy(allow_copy=False),
            ovi_atom.particles["Structure Type"][...],
        ), f"{filename} is wrong."
