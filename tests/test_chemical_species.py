# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
import mdapy as mp


def test_chemical_species():
    system = mp.System("input_files/water.xyz")
    res = system.cal_chemical_species(
        search_species=["H2O"], scale=0.4, add_mol_id=True
    )
    assert res["H2O"] * 3 == system.N, "Wrong number of water molecues."
    assert -1 not in system.data["mol_id"], "All are water molecues."
    assert system.data["mol_id"].sum() == 0, "All mol_id is 0."
