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
    
    @ti.func
    def pbc(self, rij, box:ti.types.ndarray(), boundary:ti.types.ndarray()):
        for i in ti.static(range(3)):
            if boundary[i] == 1:
                box_length = box[i, 1] - box[i, 0]
                rij[i] = rij[i] - box_length * ti.round(rij[i] / box_length)
        return rij

    @ti.kernel
    def _get_csp(self, pair:ti.types.ndarray(), 
    pos:ti.types.ndarray(element_dim=1), 
    verlet_list:ti.types.ndarray(), 
    box:ti.types.ndarray(),
                  boundary:ti.types.ndarray(), 
                  loop_index:ti.types.ndarray(), 
                  csp:ti.types.ndarray()):

        # Get loop index
        num = 0
        ti.loop_config(serialize=True)
        for i in range(self.N):
            for j in range(i+1, self.N):
                loop_index[num, 0] = i
                loop_index[num, 1] = j
                num += 1
    
        for i, index in ti.ndrange(pair.shape[0], pair.shape[1]):
            j = loop_index[index, 0]
            k = loop_index[index, 1]
            rij = pos[verlet_list[i, j]] - pos[i]
            rik = pos[verlet_list[i, k]] - pos[i]
            rij = self.pbc(rij, box, boundary)
            rik = self.pbc(rik, box, boundary)
            pair[i, index] = (rij+rik).norm_sqr()

        # Select sort
        for i in range(pair.shape[0]):
            for j in range(int(self.N/2)):
                minIndex = j
                for k in range(j+1, pair.shape[1]):
                    if pair[i, k] < pair[i, minIndex]:
                        minIndex = k
                if minIndex != j:
                    pair[i, minIndex], pair[i, j] = pair[i, j], pair[i, minIndex]
                csp[i] += pair[i, j]
        
    def compute(self):
        # from time import time
        assert self.pos.shape[0] > self.N 
        # start = time()
        kdt = kdtree(self.pos, self.box, self.boundary)
        _, verlet_list = kdt.query_nearest_neighbors(self.N)
        # end = time()
        # print(f'kdtree time: {end-start} s.')
        # start = time()
        loop_index = np.zeros((int(self.N*(self.N-1)/2), 2), dtype=int)
        pair = np.zeros((self.pos.shape[0], int(self.N*(self.N-1)/2)))
        self.csp = np.zeros(self.pos.shape[0])
        self._get_csp(pair, self.pos, verlet_list, self.box, self.boundary, loop_index, self.csp)
        # end = time()
        # print(f'csp time: {end-start} s.')


if __name__ == '__main__':
    from lattice_maker import LatticeMaker
    from time import time
    # ti.init(ti.gpu, device_memory_GB=5.0)
    ti.init(ti.cpu)
    start = time()
    lattice_constant = 4.05
    x, y, z = 100, 100, 100
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
        