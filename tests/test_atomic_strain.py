from mdapy import System, AtomicStrain
from ovito.modifiers import AtomicStrainModifier
from ovito.pipeline import ReferenceConfigurationModifier
from ovito.io import import_file
import numpy as np


def test_atom_strain():
    ref = System("input_files/strain.0.xyz")
    cur = System("input_files/strain.1.xyz")

    AS = AtomicStrain(3.0, ref, max_neigh=30)
    AS.compute(cur)

    pipeline = import_file("input_files/strain.*.xyz")
    pipeline.modifiers.append(AtomicStrainModifier(cutoff=3.0))
    data = pipeline.compute(1)

    assert np.allclose(
        data.particles["Shear Strain"][...],
        cur.data["shear_strain"].to_numpy(allow_copy=False),
    ), "shear strain is wrong."
    assert np.allclose(
        data.particles["Volumetric Strain"][...],
        cur.data["volumetric_strain"].to_numpy(allow_copy=False),
    ), "volumetric strain is wrong."

    AS = AtomicStrain(3.0, ref, max_neigh=30, affine=True)
    AS.compute(cur)

    pipeline = import_file("input_files/strain.*.xyz")
    pipeline.modifiers.append(
        AtomicStrainModifier(
            cutoff=3.0,
            affine_mapping=ReferenceConfigurationModifier.AffineMapping.ToReference,
            reference_frame=0,
        )
    )
    data = pipeline.compute(1)

    assert np.allclose(
        data.particles["Shear Strain"][...],
        cur.data["shear_strain"].to_numpy(allow_copy=False),
    ), "shear strain is wrong."
    assert np.allclose(
        data.particles["Volumetric Strain"][...],
        cur.data["volumetric_strain"].to_numpy(allow_copy=False),
    ), "volumetric strain is wrong."
