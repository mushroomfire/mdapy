import mdapy as mp
import numpy as np
from mdapy.potential import LammpsPotential

mp.init()

print("Test Lammps Potential...")

pair_parameter = """
pair_style nep
pair_coeff * * C_2024_NEP4.txt C
"""
elements_list = ["C"]
potential = LammpsPotential(pair_parameter)
nep = mp.NEP("C_2024_NEP4.txt")
gra = mp.LatticeMaker(1.42, "GRA", 6, 6, 1)
gra.compute()
np.random.seed(1)
pert_atom = 0.05
pert_atom_trans = pert_atom * 2 * np.random.random_sample((gra.N, 3)) - pert_atom
system = mp.System(box=gra.box, pos=gra.pos + pert_atom_trans)
relax_system = system.cell_opt(pair_parameter, elements_list)
e, f, v = relax_system.cal_energy_force_virial(potential, elements_list, True)
e1, f1, v1 = relax_system.cal_energy_force_virial(nep, elements_list, True)

print("Check results...")
assert np.allclose(e, e1), "energy is wrong."
assert np.allclose(f, f1), "force is wrong."
assert np.allclose(v * 6.241509125883258e-07, v1), "virial is wrong."
print("All pass!")
