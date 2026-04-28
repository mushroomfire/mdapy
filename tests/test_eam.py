# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""EAM potential — fixture-driven, no LAMMPS at runtime.

Reference outputs were generated once with the LAMMPS `eam/alloy` pair
style (see tests/_generate_fixtures/generate_advanced.py::gen_eam).
"""

import numpy as np
import polars as pl

from mdapy import build_hea
from mdapy.eam import EAM
from _fixture_helper import load_advanced, input_path


def test_evf_lammps_1():
    """5-element CoNiFeAlCu, 3x3x3 FCC supercell, no random displacement."""
    data = load_advanced("eam")
    elements = ["Co", "Ni", "Fe", "Al", "Cu"]
    eam = EAM(input_path("CoNiFeAlCu.eam.alloy"))
    model = build_hea(
        elements, [0.25, 0.25, 0.25, 0.075, 0.175], "fcc",
        3.6, nx=3, ny=3, nz=3, random_seed=1,
    )
    model.calc = eam

    assert np.allclose(model.get_energies(), data["case1__energies"]), "case1 energy"
    assert np.allclose(model.get_force(),    data["case1__forces"]),   "case1 force"
    assert np.allclose(model.get_virials(),  data["case1__virials"]),  "case1 virial"
    assert np.allclose(model.get_stress(),   data["case1__stress"]),   "case1 stress"


def test_evf_lammps_2():
    """3-element CoNiCr, 4x4x4 FCC + random displacement, two potentials."""
    data = load_advanced("eam")
    elements = ["Co", "Ni", "Cr"]
    for case_idx, fname in enumerate(
        ["NiCoCr.lammps.eam", "FeNiCrCoTi-heamix.setfl"], start=2
    ):
        eam = EAM(input_path(fname))
        model = build_hea(
            elements, [0.2, 0.3, 0.5], "fcc",
            3.6, nx=4, ny=4, nz=4, random_seed=1,
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
        model.calc = eam
        prefix = f"case{case_idx}"
        assert np.allclose(model.get_energies(), data[f"{prefix}__energies"]), f"{fname}: energy"
        assert np.allclose(model.get_force(),    data[f"{prefix}__forces"]),   f"{fname}: force"
        assert np.allclose(model.get_virials(),  data[f"{prefix}__virials"]),  f"{fname}: virial"
        assert np.allclose(model.get_stress(),   data[f"{prefix}__stress"]),   f"{fname}: stress"
