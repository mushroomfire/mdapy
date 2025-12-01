from mdapy.build_lattice import build_crystal
import numpy as np
from mdapy.create_polycrystal import CreatePolycrystal
from ovito.data import DataCollection
from ovito.modifiers import CommonNeighborAnalysisModifier
from mdapy import System


def test_ids():
    system = build_crystal("Al", "fcc", 4.05, nx=1, ny=1, nz=1)
    system.cal_common_neighbor_analysis()
    assert np.all(system.data["cna"].to_numpy(allow_copy=False) == 1), (
        "fail at 1x1x1 FCC"
    )

    system = build_crystal("Al", "fcc", 4.05, nx=1, ny=1, nz=1)
    system.cal_common_neighbor_analysis(rc=4.05 / 2**0.5 + 0.2)
    assert np.all(system.data["cna"].to_numpy(allow_copy=False) == 1), (
        "fail at 1x1x1 FCC"
    )

    system = build_crystal("Al", "fcc", 4.05, nx=10, ny=10, nz=10)
    system.build_neighbor(4.0, max_neigh=50)
    system.cal_common_neighbor_analysis(rc=4.05 / 2**0.5 + 0.2)
    assert np.all(system.data["cna"].to_numpy(allow_copy=False) == 1), (
        "fail at 10x10x10 FCC"
    )

    system = build_crystal("Al", "fcc", 4.05, nx=10, ny=10, nz=10)
    system.build_neighbor(4.0, max_neigh=50)
    system.cal_common_neighbor_analysis()
    assert np.all(system.data["cna"].to_numpy(allow_copy=False) == 1), (
        "fail at 10x10x10 FCC"
    )

    system = build_crystal("Fe", "bcc", 2.83, nx=1, ny=1, nz=1)
    system.cal_common_neighbor_analysis()

    assert np.all(system.data["cna"].to_numpy(allow_copy=False) == 3), (
        "fail at 1x1x1 BCC"
    )

    system = build_crystal("Ti", "hcp", 3.0, nx=1, ny=1, nz=1)
    system.cal_common_neighbor_analysis()

    assert np.all(system.data["cna"].to_numpy(allow_copy=False) == 2), (
        "fail at 1x1x1 HCP"
    )

    system = System("input_files/Ti.poscar")
    system.cal_common_neighbor_analysis()

    assert np.all(system.data["cna"].to_numpy(allow_copy=False) == 2), "fail at Ti HCP"

    unit = build_crystal("Al", "fcc", 4.05)
    poly = CreatePolycrystal(unit, 100, 10, randomseed=1)
    system = poly.compute(False)
    system.cal_common_neighbor_analysis()

    atom: DataCollection = system.to_ovito()
    atom.apply(CommonNeighborAnalysisModifier())
    print(atom.particles["Structure Type"][...])
    assert np.allclose(
        system.data["cna"].to_numpy(allow_copy=False),
        atom.particles["Structure Type"][...],
    ), "fail at big system with ovito in acna mode"

    poly = CreatePolycrystal(unit, 100, 10, randomseed=3)
    system = poly.compute(False)
    system.cal_common_neighbor_analysis(rc=3.2, max_neigh=30)

    atom: DataCollection = system.to_ovito()
    atom.apply(
        CommonNeighborAnalysisModifier(
            cutoff=3.2, mode=CommonNeighborAnalysisModifier.Mode.FixedCutoff
        )
    )
    print(atom.particles["Structure Type"][...])
    assert np.allclose(
        system.data["cna"].to_numpy(allow_copy=False),
        atom.particles["Structure Type"][...],
    ), "fail at big system with ovitoin fcna mode"
