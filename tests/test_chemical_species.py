import mdapy as mp

def test_chemical_species():
    system = mp.System('input_files/water.xyz')
    res = system.cal_chemical_species(search_species=['H2O'], scale=0.4)
    assert res['H2O'] * 3 == system.N, 'Wrong number of water molecues.'