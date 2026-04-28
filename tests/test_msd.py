# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""Mean-squared displacement (window + direct modes) — fixture-driven."""

import numpy as np

from mdapy.mean_squared_displacement import MeanSquaredDisplacement
from _fixture_helper import load_misc


def _trajectory_from_seed(seed, Nframe, Nparticles):
    """Reproduce the same random walk used to build the fixture."""
    np.random.seed(int(seed))
    return np.cumsum(np.random.randn(Nframe, Nparticles, 3), axis=0)


def test_MSD_window():
    data = load_misc("msd")
    pos_list = _trajectory_from_seed(int(data["seed"]),
                                     int(data["Nframe"]),
                                     int(data["Nparticles"]))
    m = MeanSquaredDisplacement(pos_list, "window")
    m.compute()
    assert np.allclose(m.msd, data["msd_window"], atol=1e-6), "MSD window mode differs"


def test_MSD_direct():
    data = load_misc("msd")
    pos_list = _trajectory_from_seed(int(data["seed"]),
                                     int(data["Nframe"]),
                                     int(data["Nparticles"]))
    m = MeanSquaredDisplacement(pos_list, "direct")
    m.compute()
    assert np.allclose(m.msd, data["msd_direct"], atol=1e-6), "MSD direct mode differs"
