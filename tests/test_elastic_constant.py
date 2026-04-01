# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
from mdapy.elastic import get_elastic_constant, _get_stress
from mdapy import build_crystal, FIRE, System
from mdapy.nep import NEP
import numpy as np
from pymatgen.core.elasticity.strain import DeformedStructureSet
from pymatgen.core.elasticity.elastic import Strain
from pymatgen.core.elasticity.elastic import ElasticTensor
from pymatgen.core.structure import Structure


def get_elastic_consant_pymatgen(
    system,
    calc,
    norm_strains=(-0.01, -0.005, 0.005, 0.01),
    shear_strains=(-0.06, -0.03, 0.03, 0.06),
    fmax: float = 1e-4,
):
    system.calc = calc

    fy = FIRE(system, optimize_cell=True)
    assert fy.run(fmax=fmax, steps=10000, show_process=False)

    equi_stress = _get_stress(system)
    atoms = system.to_ase()
    relax = Structure.from_ase_atoms(atoms)

    dfm_ss = DeformedStructureSet(
        relax, symmetry=False, norm_strains=norm_strains, shear_strains=shear_strains
    )
    strain_list = []
    stress_list = []
    for i, j in enumerate(dfm_ss):
        atoms = j.to_ase_atoms()
        dfm_system = System(ase_atom=atoms)
        dfm_system.calc = calc
        fy = FIRE(dfm_system)
        assert fy.run(fmax=fmax, steps=10000, show_process=False)
        stress_list.append(_get_stress(dfm_system))
        strain_list.append(Strain.from_deformation(dfm_ss.deformations[i]))

    et = ElasticTensor.from_independent_strains(
        strain_list, stress_list, eq_stress=equi_stress
    )
    return et


def test_elastic():
    system = build_crystal("Al", "fcc", 4.05)
    calc = NEP("input_files/UNEP-v1.txt")

    et_mda = get_elastic_constant(system, calc)
    et_pymatgen = get_elastic_consant_pymatgen(system, calc)
    # print(et_mda.voigt)
    # print(et_pymatgen.voigt)
    assert np.allclose(et_mda.voigt, et_pymatgen.voigt), "elastic tensor is different."
