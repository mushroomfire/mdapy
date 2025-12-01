from mdapy import System
import freud
import numpy as np


def test_cutoffNeigh():
    box, points = freud.data.make_random_system(10, 100, seed=0)
    system = System(box=box.to_matrix(), pos=points)
    llist = [4, 6, 8]
    ql = freud.order.Steinhardt(l=llist)
    ql.compute((box, points), {"r_max": 3})
    system.cal_steinhardt_bond_orientation(llist, rc=3.0)

    for i, j in enumerate(llist):
        assert np.allclose(
            ql.particle_order[:, i], system.data[f"ql{j}"].to_numpy(allow_copy=False)
        ), f"q{j} is wrong in rc model."

    ql = freud.order.Steinhardt(l=llist, average=True)
    ql.compute((box, points), {"r_max": 3})
    system.cal_steinhardt_bond_orientation(llist, rc=3.0, average=True)

    for i, j in enumerate(llist):
        assert np.allclose(
            ql.particle_order[:, i], system.data[f"ql{j}"].to_numpy(allow_copy=False)
        ), f"q{j} is wrong in rc model with average."

    ql = freud.order.Steinhardt(l=llist, wl=True)
    ql.compute((box, points), {"r_max": 3})
    system.cal_steinhardt_bond_orientation(llist, rc=3.0, wl=True)

    for i, j in enumerate(llist):
        assert np.allclose(
            ql.particle_order[:, i],
            system.data[f"wl{j}"].to_numpy(allow_copy=False),
            atol=1e-5,
        ), f"wl{j} is wrong in rc model with wl."

    ql = freud.order.Steinhardt(l=llist, wl=True, wl_normalize=True)
    ql.compute((box, points), {"r_max": 3})
    system.cal_steinhardt_bond_orientation(llist, rc=3.0, wlhat=True)

    for i, j in enumerate(llist):
        assert np.allclose(
            ql.particle_order[:, i],
            system.data[f"wlh{j}"].to_numpy(allow_copy=False),
            atol=1e-5,
        ), f"wlh{j} is wrong in rc model with wlh."


def test_nnnNeigh():
    box, points = freud.data.make_random_system(10, 100, seed=0)
    system = System(box=box.to_matrix(), pos=points)
    llist = [4, 6, 8]
    ql = freud.order.Steinhardt(l=llist)
    ql.compute((box, points), {"num_neighbors": 12})
    system.cal_steinhardt_bond_orientation(llist, nnn=12)

    for i, j in enumerate(llist):
        assert np.allclose(
            ql.particle_order[:, i], system.data[f"ql{j}"].to_numpy(allow_copy=False)
        ), f"q{j} is wrong in nnn model."

    ql = freud.order.Steinhardt(l=llist, average=True)
    ql.compute((box, points), {"num_neighbors": 12})
    system.cal_steinhardt_bond_orientation(llist, nnn=12, average=True)

    for i, j in enumerate(llist):
        assert np.allclose(
            ql.particle_order[:, i], system.data[f"ql{j}"].to_numpy(allow_copy=False)
        ), f"q{j} is wrong in nnn model with average."

    ql = freud.order.Steinhardt(l=llist, wl=True)
    ql.compute((box, points), {"num_neighbors": 12})
    system.cal_steinhardt_bond_orientation(llist, nnn=12, wl=True)

    for i, j in enumerate(llist):
        assert np.allclose(
            ql.particle_order[:, i],
            system.data[f"wl{j}"].to_numpy(allow_copy=False),
            atol=1e-5,
        ), f"wl{j} is wrong in nnn model with wl."

    ql = freud.order.Steinhardt(l=llist, wl=True, wl_normalize=True)
    ql.compute((box, points), {"num_neighbors": 12})
    system.cal_steinhardt_bond_orientation(llist, nnn=12, wlhat=True)

    for i, j in enumerate(llist):
        assert np.allclose(
            ql.particle_order[:, i],
            system.data[f"wlh{j}"].to_numpy(allow_copy=False),
            atol=1e-5,
        ), f"wlh{j} is wrong in nnn model with wlh."


def test_vorNeigh():
    box, points = freud.data.make_random_system(10, 100, seed=0)
    system = System(box=box.to_matrix(), pos=points)
    vor = freud.locality.Voronoi()
    vor.compute((box, points))
    llist = [4, 6, 8]
    ql = freud.order.Steinhardt(l=llist)
    ql.compute((box, points), vor.nlist)
    system.cal_steinhardt_bond_orientation(llist, use_voronoi=True)

    for i, j in enumerate(llist):
        assert np.allclose(
            ql.particle_order[:, i], system.data[f"ql{j}"].to_numpy(allow_copy=False)
        ), f"q{j} is wrong in vor model."

    ql = freud.order.Steinhardt(l=llist, average=True)
    ql.compute((box, points), vor.nlist)
    system.cal_steinhardt_bond_orientation(llist, use_voronoi=True, average=True)

    for i, j in enumerate(llist):
        assert np.allclose(
            ql.particle_order[:, i], system.data[f"ql{j}"].to_numpy(allow_copy=False)
        ), f"q{j} is wrong in vor model with average."

    ql = freud.order.Steinhardt(l=llist, wl=True)
    ql.compute((box, points), vor.nlist)
    system.cal_steinhardt_bond_orientation(llist, use_voronoi=True, wl=True)

    for i, j in enumerate(llist):
        assert np.allclose(
            ql.particle_order[:, i],
            system.data[f"wl{j}"].to_numpy(allow_copy=False),
            atol=1e-5,
        ), f"wl{j} is wrong in vor model with wl."

    ql = freud.order.Steinhardt(l=llist, wl=True, wl_normalize=True)
    ql.compute((box, points), vor.nlist)
    system.cal_steinhardt_bond_orientation(llist, use_voronoi=True, wlhat=True)

    for i, j in enumerate(llist):
        assert np.allclose(
            ql.particle_order[:, i],
            system.data[f"wlh{j}"].to_numpy(allow_copy=False),
            atol=1e-5,
        ), f"wlh{j} is wrong in vor model with wlh."

    ql = freud.order.Steinhardt(l=llist, weighted=True)
    ql.compute((box, points), vor.nlist)
    system.cal_steinhardt_bond_orientation(llist, use_voronoi=True, use_weight=True)

    for i, j in enumerate(llist):
        assert np.allclose(
            ql.particle_order[:, i], system.data[f"ql{j}"].to_numpy(allow_copy=False)
        ), f"q{j} is wrong in vor model with weight."


def test_solidliquid():
    system = System("input_files/Mo.xyz")
    box = system.box.box
    points = system.data.select("x", "y", "z").to_numpy()

    ql = freud.order.SolidLiquid(6, q_threshold=0.7, solid_threshold=7)
    ql.compute((box, points), {"num_neighbors": 12})

    system.cal_steinhardt_bond_orientation([6], nnn=12, identify_liquid=True)

    assert np.allclose(system.data["nbond"], ql.num_connections), (
        "number of solid bond is different."
    )
