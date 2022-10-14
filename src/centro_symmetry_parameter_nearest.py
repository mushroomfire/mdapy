import taichi as ti
import numpy as np
import freud


@ti.data_oriented
class CentroSymmetryParameterNearest:
    def __init__(self, pos, box, N, boundary=[1,1,1]):
        self.pos = pos
        self.box = box
        self.N = N
        self.boundary = boundary
        
    def build_kdtree(self):    
        for i in range(3):
            if self.boundary[i] == 0:
                self.box[i] += 50.
        fbox = freud.box.Box(*self.box)
        kdt = freud.locality.AABBQuery(fbox, self.pos)
        nlist = kdt.query(self.pos, {'num_neighbors':self.N, 'exclude_ii':True}).toNeighborList()
        self.verlet = nlist[:, 1].reshape(-1, self.N)
    
    @ti.kernel
    def _get_pair(self, pair:ti.types.ndarray(), pos:ti.types.ndarray(), verlet:ti.types.ndarray(), box:ti.types.ndarray()):
        for i in range(pos.shape[0]):
            n = 0
            for j in range(self.N):
                for k in range(j+1, self.N):
                    rij = ti.Vector([ti.float64(0.)]*3)
                    rik = ti.Vector([ti.float64(0.)]*3)
                    
                    for m in ti.static(range(3)):
                        rij[m] = pos[verlet[i, j], m] - pos[i, m]
                        rij[m] -= box[m] * ti.round(rij[m]/box[m])
                        
                        rik[m] = pos[verlet[i, k], m] - pos[i, m]
                        rik[m] -= box[m] * ti.round(rik[m]/box[m])
                    
                    pair[i, n] = (rij+rik).norm_sqr()
                    n += 1

    def get_csp(self):
        pair = np.zeros((self.pos.shape[0], int(self.N*(self.N-1)/2)))
        self._get_pair(pair, self.pos, self.verlet, self.box)
        self.csp = np.sum(np.sort(pair)[:, :int(self.N/2)] ,axis=1)
        
    def compute(self):
        self.build_kdtree()
        self.get_csp()


if __name__ == '__main__':
    from lattice_maker import LatticeMaker
    ti.init(ti.gpu)
    FCC = LatticeMaker(4.05, "FCC", 10, 10, 10)
    FCC.compute()
    pos = FCC.pos.to_numpy().reshape(-1, 3)
    box = np.array([40.5]*3)
    CSP = CentroSymmetryParameterNearest(pos, box, 12)
    CSP.compute()
    csp = CSP.csp 
    print(csp.min(), csp.max(), csp.mean())
        