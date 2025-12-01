import mdapy as mp
import numpy as np


def test_wcp():
    fcc = mp.System("input_files/CoCuFeNiPd-4M.dump")
    wcp = fcc.cal_warren_cowley_parameter(rc=3.0)

    assert np.allclose(
        wcp.WCP.round(2),
        np.array(
            [
                [-1.39, 0.64, 0.39, -0.3, 0.66],
                [0.64, -1.94, 0.58, 0.51, 0.2],
                [0.39, 0.58, -0.56, 0.63, -1.04],
                [-0.3, 0.51, 0.63, -1.69, 0.85],
                [0.66, 0.2, -1.04, 0.85, -0.67],
            ]
        ),
    ), "wcp is wrong."
