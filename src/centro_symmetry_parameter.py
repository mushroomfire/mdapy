import taichi as ti
import numpy as np
from mdapy.src.kdtree import kdtree


@ti.data_oriented
class CentroSymmetryParameter:
    def __init__(self, N, pos, box, boundary=[1,1,1]):
        self.pos = pos
        self.box = box
        self.N = N
        self.boundary = np.array(boundary)
        
    @ti.kernel
    def _get_pair(self, pair:ti.types.ndarray(), pos:ti.types.ndarray(), verlet_list:ti.types.ndarray(), box:ti.types.ndarray(),
                  boundary:ti.types.ndarray()):
        for i in range(pos.shape[0]):
            n = 0
            for j in range(self.N):
                for k in range(j+1, self.N):
                    rij = ti.Vector([ti.float64(0.)]*3)
                    rik = ti.Vector([ti.float64(0.)]*3)

                    for m in ti.static(range(3)):
                        rij[m] = pos[verlet_list[i, j], m] - pos[i, m]
                        rik[m] = pos[verlet_list[i, k], m] - pos[i, m]
                        if boundary[m] == 1:
                            rik[m] -= box[m] * ti.round(rik[m]/box[m])
                            rij[m] -= box[m] * ti.round(rij[m]/box[m])

                    pair[i, n] = (rij+rik).norm_sqr()
                    n += 1
        
    def compute(self):
        
        assert self.pos.shape[0] > self.N 

        kdt = kdtree(self.pos, self.box, self.boundary)
        verlet_list = kdt.query_nearest_neighbors(self.N)[1]
        
        pair = np.zeros((self.pos.shape[0], int(self.N*(self.N-1)/2)))
        self._get_pair(pair, self.pos, verlet_list, self.box[:,1]-self.box[:,0], self.boundary)
        
        try:
            import torch 
            self.csp = torch.sum(torch.sort(torch.from_numpy(pair))[0][:,:int(self.N/2)], dim=1).numpy()
        except ImportError:
            pair.sort()
            self.csp = np.sum(pair[:, :int(self.N/2)] ,axis=1)


if __name__ == '__main__':
    from lattice_maker import LatticeMaker
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
    CSP = CentroSymmetryParameter(12, FCC.pos, FCC.box, [1, 1, 1])
    CSP.compute()
    csp = CSP.csp 
    end = time()
    print(f"Cal csp time: {end-start} s.")
    print(csp.min(), csp.max(), csp.mean())
        