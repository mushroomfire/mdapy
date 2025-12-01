from mdapy.elastic import ElasticConstant
from mdapy import build_crystal
from mdapy.nep import NEP
import numpy as np


def test_elastic():
    system = build_crystal("Al", "fcc", 4.05)
    system.calc = NEP("input_files/UNEP-v1.txt")

    elas = ElasticConstant(system)
    elas.compute()

    assert np.allclose(
        elas.Cij[[0, 3, 6]] * 160.2176621,
        np.array([121.30068148, 54.16021104, 38.98089167]),
    ), "Elastic constant is wrong."
