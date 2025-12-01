from mdapy import System, WignerSeitzAnalysis
from ovito.modifiers import WignerSeitzAnalysisModifier
from ovito.pipeline import ReferenceConfigurationModifier
from ovito.io import import_file

import numpy as np


def test_wsd():
    # from mdapy import build_hea
    # import polars as pl
    # ref = build_hea(
    #     ["Cr", "Co", "Ni"], [1 / 3, 1 / 3, 1 / 3], "fcc", 3.62, nx=5, ny=5, nz=5, random_seed=1
    # )

    # cur = System(
    #     data=ref.data.with_row_index("id")
    #     .filter(~pl.col("id").is_in([1, 10, 100, 300, 400]))
    #     .select("element", pl.col("x") * 1.2, pl.col("y") * 1.2, pl.col("z") * 1.2),
    #     box=ref.box.box * 1.2,
    # )
    ref = System("input_files/hea.0.xyz")
    cur = System("input_files/hea.1.xyz")
    ws = WignerSeitzAnalysis(ref, True)
    res = ws.compute(cur)

    pipeline = import_file("input_files/hea.*.xyz")
    pipeline.modifiers.append(
        WignerSeitzAnalysisModifier(
            affine_mapping=ReferenceConfigurationModifier.AffineMapping.ToReference,
        )
    )

    data = pipeline.compute(1)

    assert data.attributes["WignerSeitz.vacancy_count"] == res["vacancy_count"]
    assert (
        data.attributes["WignerSeitz.interstitial_count"] == res["interstitial_count"]
    )
    assert np.allclose(data.particles["Occupancy"][...], res["site_occupancy"])

    pipeline = import_file("input_files/hea.*.xyz")
    pipeline.modifiers.append(
        WignerSeitzAnalysisModifier(
            affine_mapping=ReferenceConfigurationModifier.AffineMapping.ToReference,
            output_displaced=True,
        )
    )
    data = pipeline.compute(1)

    assert np.allclose(data.particles["Occupancy"][...], res["atom_occupancy"])
    assert np.allclose(data.particles["Site Index"][...], res["atom_site_index"])
    type2element = {t.id: t.name for t in data.particles.particle_type.types}
    ovi_type = np.array([type2element[i] for i in data.particles["Site Type"][...]])
    for i, j in zip(ovi_type, res["atom_site_type"]):
        assert i == j
