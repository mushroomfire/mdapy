from mdapy import System
from mdapy.eam import EAM
import numpy as np
from mdapy.lammps_potential import LammpsPotential
from mdapy import build_hea


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


def test_evf_lammps():
    element_list = ["Co", "Ni", "Fe", "Al", "Cu"]
    element_ratio = [0.25, 0.25, 0.25, 0.075, 0.175]
    eam = EAM("input_files/CoNiFeAlCu.eam.alloy")
    eam_lmp = LammpsPotential(
        pair_parameter="""
    pair_style eam/alloy
    pair_coeff * * input_files/CoNiFeAlCu.eam.alloy Co Ni Fe Al Cu
    """,
        element_list=element_list,
    )
    model = build_hea(
        element_list, element_ratio, "fcc", 3.6, nx=3, ny=3, nz=3, random_seed=1
    )
    model.calc = eam
    e = model.get_energies()
    f = model.get_force()
    v = model.get_virials()
    s = model.get_stress()

    model.calc = eam_lmp
    model.calc.results = {}
    e1 = model.get_energies()
    f1 = model.get_force()
    v1 = model.get_virials()
    s1 = model.get_stress()

    assert np.allclose(e, e1), "energy is wrong with lammps."
    assert np.allclose(f, f1, atol=1e-3), "force is wrong with lammps."
    assert np.allclose(v, v1, atol=1e-3), "virial is wrong with lammps."
    assert np.allclose(s, s1), "stress is wrong with lammps."
