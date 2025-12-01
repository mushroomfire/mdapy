from mdapy import build_crystal
from mdapy.tool_function import generate_velocity
from mdapy.data import atomic_masses, atomic_numbers
import numpy as np


def test_atomic_temp():
    fcc = build_crystal("Cu", "fcc", 3.615, nx=30, ny=30, nz=30)

    vel = generate_velocity(fcc.N, atomic_masses[atomic_numbers["Cu"]], 300, seed=1)
    fcc.update_data(fcc.data.with_columns(vx=vel[:, 0], vy=vel[:, 1], vz=vel[:, 2]))

    fcc.cal_atomic_temperature(10.0)
    assert np.allclose(fcc.data["atomic_temp"].mean(), 298.74278036044495)
