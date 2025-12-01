from mdapy import System
from mdapy.eam import EAM
import numpy as np


def test_evf():
    eam = EAM(r"input_files/ZrCu.lammps.eam.alloy")
    system = System(r"input_files/ZrCu.xyz")
    system.calc = eam

    e = system.get_energies()
    f = system.get_force()
    v = system.get_virials()
    assert np.allclose(e, system.data["energy_atom"].to_numpy(allow_copy=False)), (
        "energy is wrong."
    )
    assert np.allclose(f, system.data.select("fx", "fy", "fz").to_numpy(), atol=1e-3), (
        "force is wrong."
    )
    assert np.allclose(
        v, system.data.select(f"virial_{i}" for i in range(9)).to_numpy(), atol=1e-3
    ), "virial is wrong."
