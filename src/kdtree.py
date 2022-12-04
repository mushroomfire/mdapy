import numpy as np
from scipy.spatial import KDTree


class kdtree:
    """
    建立一个kdtree,用来搜索最近邻原子.
    """

    def __init__(self, pos, box, boundary):
        self.pos = pos
        self.box = box
        self.boundary = boundary
        self.init()

    def init(self):
        boxsize = np.array(
            [
                self.box[i][1] - self.box[i][0]
                if self.boundary[i] == 1
                else self.box[i][1] - self.box[i][0] + 50.0
                for i in range(3)
            ]
        )
        self.kdt = KDTree(self.pos - np.min(self.pos, axis=0), boxsize=boxsize)

    def query_nearest_neighbors(self, n):

        dis, index = self.kdt.query(
            self.pos - np.min(self.pos, axis=0), k=n + 1, workers=-1
        )

        return dis[:, 1:], index[:, 1:]


if __name__ == "__main__":
    from lattice_maker import LatticeMaker
    import taichi as ti
    from time import time

    # ti.init(ti.gpu, device_memory_GB=5.0)
    ti.init(ti.cpu)
    start = time()
    lattice_constant = 4.05
    x, y, z = 100, 100, 250
    FCC = LatticeMaker(lattice_constant, "FCC", x, y, z)
    FCC.compute()
    end = time()
    print(f"Build {FCC.pos.shape[0]} atoms FCC time: {end-start} s.")
    np.random.seed(10)
    noise = np.random.rand(*FCC.pos.shape)
    FCC.pos += noise / 10
    start = time()
    kdt = kdtree(FCC.pos, FCC.box, [1, 1, 1])
    end = time()
    print(f"Build kdtree time: {end-start} s.")

    start = time()
    dis, index = kdt.query_nearest_neighbors(12)
    end = time()
    print(f"Query kdtree time: {end-start} s.")

    print(dis[0])
    print(index[0])

    # FCC.write_data()
