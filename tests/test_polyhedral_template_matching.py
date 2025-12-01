from mdapy import System
from mdapy.build_lattice import build_crystal
import numpy as np
from ovito.modifiers import PolyhedralTemplateMatchingModifier
from mdapy.create_polycrystal import CreatePolycrystal


def test_single_cell():
    system = build_crystal("Al", "fcc", 4.05, nx=1, ny=1, nz=1)
    system.cal_polyhedral_template_matching()
    assert np.all(system.data["ptm"].to_numpy(allow_copy=False) == 1), (
        "fail at 1x1x1 FCC"
    )

    system = build_crystal("Al", "fcc", 4.05, nx=10, ny=10, nz=10)
    system.build_neighbor(4.0, max_neigh=50)
    system.cal_polyhedral_template_matching()
    assert np.all(system.data["ptm"].to_numpy(allow_copy=False) == 1), (
        "fail at 10x10x10 FCC"
    )

    system = build_crystal("Fe", "bcc", 2.83, nx=1, ny=1, nz=1)
    system.cal_polyhedral_template_matching()

    assert np.all(system.data["ptm"].to_numpy(allow_copy=False) == 3), (
        "fail at 1x1x1 BCC"
    )

    system = build_crystal("Ti", "hcp", 3.0, nx=1, ny=1, nz=1)
    system.cal_polyhedral_template_matching()

    assert np.all(system.data["ptm"].to_numpy(allow_copy=False) == 2), (
        "fail at 1x1x1 HCP"
    )

    system = System("input_files/Ti.poscar")
    system.cal_polyhedral_template_matching()

    assert np.all(system.data["ptm"].to_numpy(allow_copy=False) == 2), "fail at Ti HCP"

    system = build_crystal("C", "graphene", 1.42, nx=1, ny=1, nz=1)
    system.cal_polyhedral_template_matching(structure="all")

    assert np.all(system.data["ptm"].to_numpy(allow_copy=False) == 8), (
        "fail at 1x1x1 GRAPHENE"
    )

    system = build_crystal("C", "diamond", 3.5, nx=1, ny=1, nz=1)
    system.cal_polyhedral_template_matching(structure="all")

    assert np.all(system.data["ptm"].to_numpy(allow_copy=False) == 6), (
        "fail at 1x1x1 cubic diamond"
    )

    system = System("input_files/HexDiamond.xyz")
    system.cal_polyhedral_template_matching(structure="all")

    assert np.all(system.data["ptm"].to_numpy(allow_copy=False) == 7), (
        "fail at 1x1x1 hexgon diamond"
    )


def test_big():
    system1 = System("input_files/ISF.dump")
    unit = build_crystal("Al", "fcc", 4.05, nx=1, ny=1, nz=1)
    poly = CreatePolycrystal(unit, 100, 10, randomseed=1, add_graphene=True)
    system2 = poly.compute(False)

    for system in [system1, system2]:
        system.cal_polyhedral_template_matching(
            structure="all",
            return_atomic_distance=True,
            return_rmsd=True,
            return_orientation=True,
        )
        atom = system.to_ovito()
        modifier = PolyhedralTemplateMatchingModifier(
            output_interatomic_distance=True, output_orientation=True, output_rmsd=True
        )

        modifier.structures[
            PolyhedralTemplateMatchingModifier.Type.GRAPHENE
        ].enabled = True
        modifier.structures[PolyhedralTemplateMatchingModifier.Type.ICO].enabled = True
        modifier.structures[PolyhedralTemplateMatchingModifier.Type.SC].enabled = True
        modifier.structures[
            PolyhedralTemplateMatchingModifier.Type.CUBIC_DIAMOND
        ].enabled = True
        modifier.structures[
            PolyhedralTemplateMatchingModifier.Type.HEX_DIAMOND
        ].enabled = True

        atom.apply(modifier)
        assert np.allclose(
            atom.particles["Structure Type"][...],
            system.data["ptm"].to_numpy(allow_copy=False),
        ), "Structure Type is wrong"
        sele = (system.data["ptm"].to_numpy(allow_copy=False) != 0) & (
            system.data["ptm"].to_numpy(allow_copy=False) != 8
        )

        assert np.allclose(
            atom.particles["Interatomic Distance"][...][sele],
            system.data["interatomic_distance"].to_numpy(allow_copy=False)[sele],
        ), "interatomic_distance is wrong"

        assert np.allclose(
            atom.particles["RMSD"][...][sele],
            system.data["rmsd"].to_numpy(allow_copy=False)[sele],
            atol=1e-5,
        ), "RMSD is wrong"

        # x = atom.particles["Orientation"][...][:, 0][sele]
        # y = system.data["qx"].to_numpy(allow_copy=False)[sele]
        # m = abs(x - y)
        # ss = m > 0.1
        # print(system.data["ptm"].to_numpy(allow_copy=False)[sele][ss])
        # print(x[ss])
        # print(y[ss])

        assert np.allclose(
            atom.particles["Orientation"][...][:, 0][sele],
            system.data["qx"].to_numpy(allow_copy=False)[sele],
            atol=1e-5,
        ), "qx is wrong"

        assert np.allclose(
            atom.particles["Orientation"][...][:, 1][sele],
            system.data["qy"].to_numpy(allow_copy=False)[sele],
            atol=1e-5,
        ), "qy is wrong"

        assert np.allclose(
            atom.particles["Orientation"][...][:, 2][sele],
            system.data["qz"].to_numpy(allow_copy=False)[sele],
            atol=1e-5,
        ), "qz is wrong"

        assert np.allclose(
            atom.particles["Orientation"][...][:, 3][sele],
            system.data["qw"].to_numpy(allow_copy=False)[sele],
            atol=1e-5,
        ), "qw is wrong"
