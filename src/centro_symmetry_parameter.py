import taichi as ti
import numpy as np


@ti.data_oriented
class CentroSymmetryParameter:
    def __init__(self, pos, box, N, boundary, verlet_list, distance_list, neighbor_number):
        self.pos = pos
        self.box = box
        self.N = N
        self.boundary = np.array(boundary)
        self.verlet_list = verlet_list
        self.distance_list = distance_list
        self.neighbor_number = neighbor_number
        
    @ti.kernel
    def _get_pair(self, pair:ti.types.ndarray(), pos:ti.types.ndarray(), verlet_list:ti.types.ndarray(), box:ti.types.ndarray(),
                  sort_index:ti.types.ndarray(), boundary:ti.types.ndarray(), neighbor_number:ti.types.ndarray()):
        for i in range(pos.shape[0]):
            if neighbor_number[i] >= self.N:
                n = 0
                for j in range(self.N):
                    for k in range(j+1, self.N):
                        rij = ti.Vector([ti.float64(0.)]*3)
                        rik = ti.Vector([ti.float64(0.)]*3)

                        for m in ti.static(range(3)):
                            rij[m] = pos[verlet_list[i, sort_index[i,j]], m] - pos[i, m]
                            rik[m] = pos[verlet_list[i, sort_index[i,k]], m] - pos[i, m]
                            if boundary[m] == 1:
                                rik[m] -= box[m] * ti.round(rik[m]/box[m])
                                rij[m] -= box[m] * ti.round(rij[m]/box[m])

                        pair[i, n] = (rij+rik).norm_sqr()
                        n += 1
        
    def compute(self):
        try:
            import torch 
            print('Parallel computing by torch.')
            pair = torch.zeros((self.pos.shape[0], int(self.N*(self.N-1)/2)), dtype=torch.float64).numpy()
            sort_index = torch.argsort(torch.from_numpy(self.distance_list))[:,:self.N].numpy()
            self._get_pair(pair, self.pos, self.verlet_list, self.box, sort_index, self.boundary, self.neighbor_number)
            self.csp = torch.sum(torch.sort(torch.from_numpy(pair))[0][:,:int(self.N/2)], dim=1).numpy()
        except ImportError:
            pair = np.zeros((self.pos.shape[0], int(self.N*(self.N-1)/2)))
            sort_index = np.argpartition(self.distance_list, self.N)[:, :self.N]
            self._get_pair(pair, self.pos, self.verlet_list, self.box, sort_index, self.boundary, self.neighbor_number)
            pair.sort()
            self.csp = np.sum(pair[:, :int(self.N/2)] ,axis=1)


if __name__ == '__main__':
    from lattice_maker import LatticeMaker
    from neighbor import Neighbor
    from time import time
    ti.init(ti.gpu, device_memory_GB=5.0)
    # ti.init(ti.cpu)
    start = time()
    lattice_constant = 4.05
    x, y, z = 100, 100, 50
    FCC = LatticeMaker(lattice_constant, "FCC", x, y, z)
    FCC.compute()
    end = time()
    print(f"Build {FCC.pos.shape[0]} atoms FCC time: {end-start} s.")
    start = time()

    neigh = Neighbor(FCC.pos, FCC.box, 5.0, max_neigh=60)
    neigh.compute()
    end = time()
    print(f"Build neighbor time: {end-start} s.")
    start = time()
    CSP = CentroSymmetryParameter(FCC.pos, FCC.box[:,1], 12, [1, 1, 1], neigh.verlet_list, neigh.distance_list, neigh.neighbor_number)
    CSP.compute()
    csp = CSP.csp 
    end = time()
    print(f"Cal csp time: {end-start} s.")
    print(csp.min(), csp.max(), csp.mean())
        