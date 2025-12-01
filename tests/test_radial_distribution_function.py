import mdapy as mp
from ovito.modifiers import CoordinationAnalysisModifier
import numpy as np


def test_rdf():
    system = mp.System("input_files/AlCrNi.xyz")

    rdf = system.cal_radial_distribution_function(5.0, 50)
    mytype = system.data["element"].unique().sort()
    ovi_atom = system.to_ovito()
    ovi_atom.apply(
        CoordinationAnalysisModifier(cutoff=5.0, number_of_bins=50, partial=True)
    )

    rdf_table = ovi_atom.tables["coordination-rdf"]
    rdf_names = rdf_table.y.component_names

    for i in range(len(mytype)):
        for j in range(i, len(mytype)):
            name = f"{mytype[i]}-{mytype[j]}"
            if name not in rdf_names:
                name = f"{mytype[j]}-{mytype[i]}"
                assert name in rdf_names
            assert np.allclose(rdf.g[i, j], rdf_table.y[:, rdf_names.index(name)]), (
                f"{name} rdf is wrong."
            )
