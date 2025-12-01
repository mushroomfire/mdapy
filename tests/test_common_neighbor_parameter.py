import mdapy as mp
import numpy as np


def test_cnp():
    a = 3.615
    system = mp.build_crystal("Cu", "fcc", a)

    system.cal_common_neighbor_parameter(0.86 * a)

    assert np.allclose(system.data["cnp"].max(), 0.0)

    system = mp.build_crystal("Cu", "bcc", a)

    system.cal_common_neighbor_parameter(1.21 * a)

    assert np.allclose(system.data["cnp"].max(), 0.0)

    system = mp.build_crystal("Cu", "hcp", a, c_over_a=1.633)

    system.cal_common_neighbor_parameter(1.21 * a)

    assert np.allclose(system.data["cnp"].max(), 8.71215)

    system = mp.build_crystal(
        "Cu", "fcc", a, miller1=[1, -1, 0], miller2=[1, 1, -2], miller3=[1, 1, 1]
    )

    system.box.boundary[2] = 0
    system.cal_common_neighbor_parameter(0.86 * a)
    assert np.allclose(system.data["cnp"].max(), 13.068225)

    system = mp.build_crystal("Cu", "fcc", a, nx=5, ny=5, nz=5)
    system.box.boundary[2] = 0
    system.cal_common_neighbor_parameter(0.86 * a)
    assert np.allclose(system.data["cnp"].max(), 26.13645)
