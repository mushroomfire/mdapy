import taichi as ti 
import numpy as np 


@ti.data_oriented
class PairDistribution:
    def __init__(self, rc, nbin, rho, verlet_list, distance_list):
        self.rc = rc
        self.nbin = nbin
        self.rho = rho
        self.verlet_list = verlet_list
        self.distance_list = distance_list 
        self.g = np.zeros(self.nbin, dtype=np.float32)
        self.N = self.distance_list.shape[0]
        
    @ti.kernel
    def _rdf(self, verlet_list:ti.types.ndarray(), distance_list:ti.types.ndarray(),
             g:ti.types.ndarray()):

        dr = self.rc/self.nbin
        for i in range(self.N):
            for jindex in range(verlet_list.shape[1]):
                j = verlet_list[i, jindex]
                if j > i and j > -1:
                    dis = distance_list[i, jindex]
                    if dis <= self.rc:
                        k = ti.floor(dis/dr, dtype=ti.i32)
                        if k > self.nbin-1:
                            k = self.nbin-1
                        g[k] += 2.
    
    def compute(self):
        r = np.linspace(0, self.rc, self.nbin+1)
        self._rdf(self.verlet_list, self.distance_list, self.g)
        const = 4.0 * np.pi * self.rho / 3.0
        self.g /= self.N*const*(r[1:]**3-r[:-1]**3)
        self.r = (r[1:]+r[:-1])/2
        
    def plot(self):
        
        from mdapy.plot.pltset import pltset, cm2inch
        import matplotlib.pyplot as plt
        pltset()
        plt.figure(figsize=(cm2inch(8), cm2inch(5)), dpi=150)
        plt.subplots_adjust(left=0.16,bottom=0.225,right=0.97,top=0.97)
        plt.plot(self.r, self.g, 'o-', ms=3)
        plt.xlabel('r ($\mathregular{\AA}$)')
        plt.ylabel('g(r)')
        plt.xlim(0, self.rc)
        plt.show()


if __name__ == '__main__':
    from lattice_maker import LatticeMaker
    from neighbor import Neighbor
    from time import time
    ti.init(ti.cpu)
    # ti.init(ti.cpu)
    start = time()
    lattice_constant = 4.05
    x, y, z = 100, 100, 50
    FCC = LatticeMaker(lattice_constant, "FCC", x, y, z)
    FCC.compute()
    end = time()
    print(f"Build {FCC.pos.shape[0]} atoms FCC time: {end-start} s.")
    #FCC.write_data()
    start = time()
    rc = 6.0
    neigh = Neighbor(FCC.pos, FCC.box, rc, max_neigh=80)
    neigh.compute()
    end = time()
    print(f"Build neighbor time: {end-start} s.")
    start = time()
    rho = FCC.pos.shape[0] / np.product(FCC.box[:,1]-FCC.box[:,0])
    gr = PairDistribution(rc, 200, rho, neigh.verlet_list, neigh.distance_list)
    gr.compute()
    end = time()
    print(f"Cal gr time: {end-start} s.")
    gr.plot()

        