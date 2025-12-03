from mdapy import System
from ovito.modifiers import BondAnalysisModifier, CreateBondsModifier
import numpy as np


def test_adf():
    system = System("input_files/water.xyz")
    ovi_atom = system.to_ovito()
    adf = system.cal_angular_distribution_function(
        {
            "O-H-H": [0, 2.0, 0, 2.0],
            "O-O-H": [0, 2.0, 0, 2.0],
            "H-H-H": [0, 2.0, 0, 2.0],
            "H-O-O": [0, 2.0, 0, 2.0],
            "O-O-O": [0, 2.0, 0, 2.0],
            "H-O-H": [0, 2.0, 0, 2.0],
        },
        40,
    )

    ovi_atom.apply(CreateBondsModifier(cutoff=2.0))
    ovi_atom.apply(
        BondAnalysisModifier(
            bins=40,
            length_cutoff=2.0,
            partition=BondAnalysisModifier.Partition.ByParticleType,
        )
    )

    histogram = ovi_atom.tables["bond-angle-distr"].y

    for column, name in enumerate(histogram.component_names):
        if name == "H-O-H":
            assert np.allclose(histogram[:, column], adf.bond_angle_distribution[0])
        elif name == "O-O-H":
            assert np.allclose(histogram[:, column], adf.bond_angle_distribution[1])
        elif name == "H-H-H":
            assert np.allclose(histogram[:, column], adf.bond_angle_distribution[2])
        elif name == "O-H-O":
            assert np.allclose(histogram[:, column], adf.bond_angle_distribution[3])
        elif name == "O-O-O":
            assert np.allclose(histogram[:, column], adf.bond_angle_distribution[4])
        elif name == "O-H-H":
            assert np.allclose(histogram[:, column], adf.bond_angle_distribution[5])
