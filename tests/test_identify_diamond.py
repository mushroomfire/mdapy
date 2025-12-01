from mdapy.build_lattice import build_crystal
import numpy as np
from mdapy import System


def test_ids():
    diamond = build_crystal("C", "diamond", 3.5, nx=1, ny=1, nz=1)
    diamond.cal_identify_diamond_structure()

    assert np.all(diamond.data["ids"].to_numpy(allow_copy=False) == 1), (
        "fail at 1x1x1 diamond"
    )

    diamond = build_crystal("C", "diamond", 3.5, nx=3, ny=3, nz=3)
    diamond.cal_identify_diamond_structure()

    assert np.all(diamond.data["ids"].to_numpy(allow_copy=False) == 1), (
        "fail at 3x3x3 diamond"
    )

    diamond = build_crystal("C", "diamond", 3.5, nx=10, ny=10, nz=10)
    diamond.build_neighbor(4.0)
    diamond.cal_identify_diamond_structure()

    assert np.all(diamond.data["ids"].to_numpy(allow_copy=False) == 1), (
        "fail at 10x10x10 diamond"
    )

    diamond = System("input_files/HexDiamond.xyz")
    diamond.cal_identify_diamond_structure()

    assert np.all(diamond.data["ids"].to_numpy(allow_copy=False) == 4), (
        "fail at Hex diamond"
    )
