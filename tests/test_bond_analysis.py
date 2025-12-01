from mdapy import System
from ovito.modifiers import BondAnalysisModifier, CreateBondsModifier
import numpy as np


def test_bond():
    system = System("input_files/water.xyz")
    ovi_atom = system.to_ovito()

    bo = system.cal_bond_analysis(2.0, 40, max_neigh=10)
    ovi_atom.apply(CreateBondsModifier(cutoff=2.0))
    ovi_atom.apply(BondAnalysisModifier(bins=40, length_cutoff=2.0))

    length = ovi_atom.tables["bond-length-distr"].xy()
    angle = ovi_atom.tables["bond-angle-distr"].xy()
    assert np.allclose(length[:, 0], bo.r_length)
    assert np.allclose(length[:, 1], bo.bond_length_distribution)
    assert np.allclose(angle[:, 0], bo.r_angle)
    assert np.allclose(angle[:, 1], bo.bond_angle_distribution)
