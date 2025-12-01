import mdapy as mp
from ovito.io import import_file
from ovito.modifiers import VoronoiAnalysisModifier
import numpy as np
import freud


class TestVoronoiVolume:
    def calculate_voronoi_volume(self, filename):
        system = mp.System(filename)
        system.cal_voronoi_volume()
        pipeline = import_file(filename)
        pipeline.modifiers.append(VoronoiAnalysisModifier())
        data = pipeline.compute()

        assert np.allclose(
            data.particles["Atomic Volume"][...], system.data["volume"].to_numpy(), 1e-6
        ), "Atomic volumes are different."
        assert np.allclose(
            data.particles["Cavity Radius"][...],
            system.data["cavity_radius"].to_numpy() * 0.5,
            1e-6,
        ), "Cavity radius is different."
        assert np.all(
            data.particles["Coordination"][...]
            == system.data["neighbor_number"].to_numpy()
        ), "Neighbor number is different."

    def test_box_big_rec(self):
        self.calculate_voronoi_volume("input_files/rec_box_big.xyz")

    def test_box_small_rec(self):
        self.calculate_voronoi_volume("input_files/rec_box_small.xyz")

    def test_box_big_tri(self):
        self.calculate_voronoi_volume("input_files/tri_box_big.xyz")

    def test_box_small_tri(self):
        self.calculate_voronoi_volume("input_files/tri_box_small.xyz")


def test_voronoi_neighbor():
    data = import_file("input_files/rec_box_big.xyz").compute()
    vor = freud.locality.Voronoi()
    vor.compute(data)
    system = mp.System(ovito_atom=data)
    system.build_voronoi_neighbor()
    for i in range(system.N):
        sele = vor.nlist.query_point_indices == i
        assert np.all(
            np.equal(vor.nlist.point_indices[sele], np.sort(system.voro_verlet_list[i]))
        ), f"atom {i} has wrong voronoi neighbor."
