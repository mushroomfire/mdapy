import mdapy as mp


def test_chemical_species():
    system = mp.System("input_files/water.xyz")
    res = system.cal_chemical_species(search_species=["H2O"], scale=0.4, add_mol_id=True)
    assert res["H2O"] * 3 == system.N, "Wrong number of water molecues."
    assert -1 not in system.data['mol_id'], "All are water molecues."
    assert system.data['mol_id'].sum()==0, "All mol_id is 0."
