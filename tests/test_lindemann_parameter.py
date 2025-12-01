import numpy as np
import mdapy as mp


def test_lindemann():
    Nframe, Nparticles = 200, 500
    pos_list = np.cumsum(
        np.random.choice([-1.0, 0.0, 1.0], size=(Nframe, Nparticles, 3)), axis=0
    )

    LDMG = mp.LindemannParameter(pos_list, only_global=True)
    LDMG.compute()

    LDML = mp.LindemannParameter(pos_list)
    LDML.compute()

    assert np.isclose(LDMG.lindemann_trj, LDML.lindemann_trj), (
        "lindemann parameter is wrong."
    )
