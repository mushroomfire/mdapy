import mdapy as mp
from ovito.io import import_file
from ovito.data import CutoffNeighborFinder


class TestNeighborCutoff:
    def __test_box(self, filename, r_max=5.0, max_neigh=None, small_box=False):
        system = mp.System(filename)
        system.build_neighbor(r_max, max_neigh)
        pipeline = import_file(filename)
        data = pipeline.compute()
        finder = CutoffNeighborFinder(r_max, data)

        for test_idx in [
            0,
            int(system.N / 4),
            int(system.N / 2),
            int(system.N / 4 * 3),
            system.N - 1,
        ]:
            ovito_list = [neigh.index for neigh in finder.find(test_idx)]
            mdapy_list = system.verlet_list[test_idx][system.verlet_list[test_idx] > -1]

            assert system.neighbor_number[test_idx] == len(ovito_list), (
                f"Atom {test_idx} neighbor number not equal!"
            )
            if not small_box:
                for i in mdapy_list:
                    assert i in ovito_list, f"Atom {test_idx} neighbor not equal!"

    def test_box_big_rec(
        self, filename="input_files/rec_box_big.xyz", r_max=5.0, max_neigh=50
    ):
        self.__test_box(filename, r_max, max_neigh)

    def test_box_big_rec_no_max_neigh(
        self, filename="input_files/rec_box_big.xyz", r_max=5.0, max_neigh=None
    ):
        self.__test_box(filename, r_max, max_neigh)

    def test_box_small_rec(
        self, filename="input_files/rec_box_small.xyz", r_max=5.0, max_neigh=50
    ):
        self.__test_box(filename, r_max, max_neigh, True)

    def test_box_small_rec_no_max_neigh(
        self, filename="input_files/rec_box_small.xyz", r_max=5.0, max_neigh=None
    ):
        self.__test_box(filename, r_max, max_neigh, True)

    def test_box_big_tri(
        self, filename="input_files/tri_box_big.xyz", r_max=5.0, max_neigh=50
    ):
        self.__test_box(filename, r_max, max_neigh)

    def test_box_big_tri_no_max_neigh(
        self, filename="input_files/tri_box_big.xyz", r_max=5.0, max_neigh=None
    ):
        self.__test_box(filename, r_max, max_neigh)

    def test_box_small_tri(
        self, filename="input_files/tri_box_small.xyz", r_max=5.0, max_neigh=50
    ):
        self.__test_box(filename, r_max, max_neigh, True)

    def test_box_small_tri_no_max_neigh(
        self, filename="input_files/tri_box_small.xyz", r_max=5.0, max_neigh=None
    ):
        self.__test_box(filename, r_max, max_neigh, True)
