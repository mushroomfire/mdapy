import numpy as np
from mdapy.mean_squared_displacement import MeanSquaredDisplacement
import freud


def test_MSD():
    Nframe, Nparticles = 200, 1000
    np.random.seed(1)
    pos_list = np.cumsum(np.random.randn(Nframe, Nparticles, 3), axis=0)
    f = freud.msd.MSD(mode="window")
    f.compute(pos_list)

    m = MeanSquaredDisplacement(pos_list, "window")
    m.compute()

    assert np.allclose(f.msd, m.msd), "msd is wrong in windows mode."

    f = freud.msd.MSD(mode="direct")
    f.compute(pos_list)

    m = MeanSquaredDisplacement(pos_list, "direct")
    m.compute()

    assert np.allclose(f.msd, m.msd), "msd is wrong in direct mode."
