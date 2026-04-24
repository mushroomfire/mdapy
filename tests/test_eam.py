# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
from mdapy import System
from mdapy.eam import EAM
import numpy as np
import polars as pl
from mdapy.lammps_potential import LammpsPotential
from mdapy import build_hea


def test_evf_lammps_1():
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

    e1 = model.get_energies()
    f1 = model.get_force()
    v1 = model.get_virials()
    s1 = model.get_stress()

    assert np.allclose(e, e1), "energy is wrong with lammps."
    assert np.allclose(f, f1), "force is wrong with lammps."
    assert np.allclose(v, v1), "virial is wrong with lammps."
    assert np.allclose(s, s1), "stress is wrong with lammps."


def test_evf_lammps_2():
    element_list = ["Co", "Ni", "Cr"]
    element_ratio = [0.2, 0.3, 0.5]
    for filename in [
        "input_files/NiCoCr.lammps.eam",
        "input_files/FeNiCrCoTi-heamix.setfl",
    ]:
        eam = EAM(filename)
        eam_lmp = LammpsPotential(
            pair_parameter=f"""
        pair_style eam/alloy
        pair_coeff * * {filename} Co Ni Cr
        """,
            element_list=element_list,
        )
        model = build_hea(
            element_list, element_ratio, "fcc", 3.6, nx=4, ny=4, nz=4, random_seed=1
        )
        np.random.seed(1)
        noise = (np.random.random((model.N, 3)) - 0.5) * 1.4
        model.update_data(
            model.data.with_columns(
                pl.col("x") + noise[:, 0],
                pl.col("y") + noise[:, 1],
                pl.col("z") + noise[:, 2],
            )
        )
        # model.build_neighbor(5.0)
        # print(model.distance_list.min())
        model.calc = eam
        e = model.get_energies()
        f = model.get_force()
        v = model.get_virials()
        s = model.get_stress()

        model.calc = eam_lmp

        e1 = model.get_energies()
        f1 = model.get_force()
        v1 = model.get_virials()
        s1 = model.get_stress()
        print(abs(e - e1).max())
        assert np.allclose(e, e1), "energy is wrong with lammps."
        assert np.allclose(f, f1), "force is wrong with lammps."
        assert np.allclose(v, v1), "virial is wrong with lammps."
        assert np.allclose(s, s1), "stress is wrong with lammps."


if __name__ == "__main__":
    test_evf_lammps_1()
    test_evf_lammps_2()
