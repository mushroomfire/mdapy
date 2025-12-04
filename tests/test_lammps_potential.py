from mdapy import System
import numpy as np
from mdapy.lammps_potential import LammpsPotential


def test_lammpspotential():
    system = System("input_files/ZrCu.xyz")

    eam = LammpsPotential(
        pair_parameter="""
    pair_style eam/alloy
    pair_coeff * * input_files/ZrCu.lammps.eam.alloy Zr Cu
    """,
        element_list=["Zr", "Cu"],
        centroid_stress=False,
    )
    system.calc = eam
    system.calc.results = {}
    e = system.get_energies()
    f = system.get_force()
    v = system.get_virials()

    assert np.allclose(e, system.data["energy_atom"].to_numpy())
    assert np.allclose(f, system.data.select("fx", "fy", "fz").to_numpy(), atol=1e-3)
    assert np.allclose(
        v, system.data.select(f"virial_{i}" for i in range(9)).to_numpy(), atol=1e-3
    ), "virial is wrong."
