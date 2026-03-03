from mdapy.system import System
from mdapy.lammps_potential import LammpsPotential
from mdapy import build_hea
from mdapy.nep import NEP
from mdapy.nep4ase import NEP4ASE
from ase.io import read
from ase.optimize import FIRE
from pynep.calculate import NEP as pyNEP
import numpy as np


def test_nep_lmp():

    hea1 = build_hea(
        ["Al", "Cr", "Ni"],
        [1 / 3, 1 / 3, 1 / 3],
        "fcc",
        a=3.6,
        nx=3,
        ny=3,
        nz=3,
        random_seed=1,
    )
    hea2 = System("input_files/AlCrNi.xyz")
    nep_lmp = LammpsPotential(
        """pair_style nep
        pair_coeff * * input_files/UNEP-v1.txt Al Cr Ni
        """,
        ["Al", "Cr", "Ni"],
        centroid_stress=True,
    )
    nep_mda = NEP("input_files/UNEP-v1.txt")
    for i, hea in enumerate([hea1, hea2]):
        hea.calc = nep_lmp
        hea.calc.results = {}
        e_lmp = hea.get_energies()
        f_lmp = hea.get_force()
        s_lmp = hea.get_stress()
        v_lmp = hea.get_virials()

        hea.calc = nep_mda
        hea.calc.results = {}
        e_mda = hea.get_energies()
        f_mda = hea.get_force()
        s_mda = hea.get_stress()
        v_mda = hea.get_virials()

        assert np.allclose(e_lmp, e_mda), f"round {i}: energy is wrong for NEP."
        assert np.allclose(f_lmp, f_mda), f"round {i}: force is wrong for NEP."
        assert np.allclose(s_lmp, s_mda), f"round {i}: stress is wrong for NEP."
        assert np.allclose(v_lmp, v_mda), f"round {i}: virial is wrong for NEP."


def test_nep():
    atom = read("input_files/AlCrNi.xyz", format="extxyz")
    system = System(ase_atom=atom)
    pynep = pyNEP("input_files/UNEP-v1.txt")
    mynepase = NEP4ASE("input_files/UNEP-v1.txt")
    mynep = NEP("input_files/UNEP-v1.txt")

    atom.calc = pynep
    ref_e = atom.get_potential_energies()
    ref_f = atom.get_forces()
    ref_s = atom.get_stress()
    ref_des = pynep.get_property("descriptor", atom)
    ref_lat = pynep.get_property("latent", atom)

    atom.calc = mynepase
    e = atom.get_potential_energies()
    f = atom.get_forces()
    s = atom.get_stress()
    des = mynepase.get_descriptor(atom)
    lat = mynepase.get_latentspace(atom)

    assert np.allclose(ref_e, e), "energy is wrong for NEP4ASE."
    assert np.allclose(ref_f, f), "force is wrong for NEP4ASE."
    assert np.allclose(ref_s, s), "stress is wrong for NEP4ASE."
    assert np.allclose(ref_des, des), "des is wrong for NEP4ASE."
    assert np.allclose(ref_lat, lat), "lat is wrong for NEP4ASE."

    system.calc = mynep
    e = system.get_energies()
    f = system.get_force()
    s = system.get_stress()
    des = mynep.get_descriptor(system.data, system.box)
    lat = mynep.get_latentspace(system.data, system.box)
    assert np.allclose(ref_e, e), "energy is wrong for NEP4ASE."
    assert np.allclose(ref_f, f), "force is wrong for NEP4ASE."
    assert np.allclose(ref_s, s), "stress is wrong for NEP4ASE."
    assert np.allclose(ref_des, des), "des is wrong for NEP4ASE."
    assert np.allclose(ref_lat, lat), "lat is wrong for NEP4ASE."


def test_nep_minimize():
    atom = read("input_files/AlCrNi.xyz", format="extxyz")
    pynep = pyNEP("input_files/UNEP-v1.txt")
    atom.calc = pynep
    refe = []

    def log_energy_1():
        e = atom.get_potential_energy()
        refe.append(e)

    dyn = FIRE(atom, logfile=None)
    dyn.attach(log_energy_1, interval=1)
    dyn.run(fmax=1e-4, steps=10)

    atom = read("input_files/AlCrNi.xyz", format="extxyz")
    mynepase = NEP4ASE("input_files/UNEP-v1.txt")
    atom.calc = mynepase
    ce = []

    def log_energy_2():
        e = atom.get_potential_energy()
        ce.append(e)

    dyn = FIRE(atom, logfile=None)
    dyn.attach(log_energy_2, interval=1)
    dyn.run(fmax=1e-4, steps=10)

    assert np.allclose(refe, ce), "minimization energy evolution is wrong!"
