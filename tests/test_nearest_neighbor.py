from mdapy import System
import numpy as np
from ovito.data import NearestNeighborFinder
from ovito.io import import_file


def test_nn(k: int = 12):
    for filename in [
        "input_files/HexDiamond.xyz",
        "input_files/tri_box_small.xyz",
        "input_files/rec_box_small.xyz",
        "input_files/rec_box_big.xyz",
        "input_files/AlCrNi.xyz",
        "input_files/tri_box_big.xyz",
    ]:
        atom = import_file(filename).compute()
        system = System(ovito_atom=atom)
        finder = NearestNeighborFinder(k, atom)
        ind, vec = finder.find_all()
        system.build_nearest_neighbor(k)
        ovidis = np.linalg.norm(vec, axis=-1)

        assert np.allclose(ovidis, system.distance_list[: system.N]), (
            f"distance list is different with Ovito for {filename}"
        )


# import freud
# aq = freud.locality.AABBQuery.from_system(atom)
# query_result = aq.query(aq.points, dict(mode='nearest', num_neighbors=k, exclude_ii=True))
# nlist = query_result.toNeighborList()
# x = nlist.distances[nlist.query_point_indices==0]
# y = nlist.point_indices[nlist.query_point_indices==0]
# sele = np.argsort(x)
