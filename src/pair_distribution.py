import taichi as ti 
import numpy as np 
from mdapy.plot.pltset import pltset, cm2inch
import matplotlib.pyplot as plt

@ti.data_oriented
class PairDistribution:
    def __init__(self, rc, nbin, rho, verlet_list, distance_list, type_list=None):
        self.rc = rc
        self.nbin = nbin
        self.rho = rho
        self.verlet_list = verlet_list
        self.distance_list = distance_list 
        self.N = self.distance_list.shape[0]
        if type_list is not None:
            self.type_list = type_list-1
        else:
            self.type_list = np.zeros(self.N, dtype=int)
        self.Ntype = len(np.unique(self.type_list))
        pltset()
        
    @ti.kernel
    def _rdf(self, verlet_list:ti.types.ndarray(), distance_list:ti.types.ndarray(), type_list:ti.types.ndarray(),
             g:ti.types.ndarray(), concentrates:ti.types.ndarray()):

        dr = self.rc/self.nbin
        for i in range(self.N):
            i_type = type_list[i]
            for jindex in range(verlet_list.shape[1]):
                j = verlet_list[i, jindex]
                j_type = type_list[j]
                if j > -1:
                    if j > i:
                        dis = distance_list[i, jindex]
                        if dis <= self.rc:
                            k = ti.floor(dis/dr, dtype=ti.i32)
                            if k > self.nbin-1:
                                k = self.nbin-1
                            if j_type >= i_type:
                                g[i_type, j_type, k] += 2./concentrates[i_type]/concentrates[j_type]
                else:
                    break
    
    def compute(self):
        r = np.linspace(0, self.rc, self.nbin+1)
        
        concentrates = np.array([len(self.type_list[self.type_list==i]) for i in range(self.Ntype)])/self.N
        self.g = np.zeros((self.Ntype, self.Ntype, self.nbin), dtype=np.float64)
        self._rdf(self.verlet_list, self.distance_list, self.type_list, self.g, concentrates)
        const = 4.0 * np.pi * self.rho / 3.0
        self.g /= self.N*const*(r[1:]**3-r[:-1]**3)
        self.r = (r[1:]+r[:-1])/2
        self.g_total = np.zeros_like(self.r)
        for i in range(self.Ntype):
            for j in range(self.Ntype):
                if j == i:
                    self.g_total += concentrates[i]*concentrates[j]*self.g[i, j]
                else:
                    self.g_total += 2*concentrates[i]*concentrates[j]*self.g[i, j]
        
    def plot(self):
        fig = plt.figure(figsize=(cm2inch(8), cm2inch(5)), dpi=150)
        plt.subplots_adjust(left=0.16,bottom=0.225,right=0.97,top=0.97)
        plt.plot(self.r, self.g_total, 'o-', ms=3)
        plt.xlabel('r ($\mathregular{\AA}$)')
        plt.ylabel('g(r)')
        plt.xlim(0, self.rc)
        ax = plt.gca()
        plt.show()
        return fig, ax
    
    def plot_partial(self, elements_list=None):
        if elements_list is not None:
            assert len(elements_list) == self.Ntype
        fig = plt.figure(figsize=(cm2inch(8), cm2inch(5)), dpi=150)
        plt.subplots_adjust(left=0.16,bottom=0.225,right=0.97,top=0.97)
        
        for i in range(self.Ntype):
            for j in range(self.Ntype):
                if j >= i:
                    if elements_list is not None:
                        plt.plot(self.r, self.g[i, j], label = f'{elements_list[i]}-{elements_list[j]}')
                    else:
                        plt.plot(self.r, self.g[i, j], label = f'{i}-{j}')
        plt.legend(ncol=2, fontsize=6)
        plt.xlabel('r ($\mathregular{\AA}$)')
        plt.ylabel('g(r)')
        plt.xlim(0, self.rc)
        ax = plt.gca()
        plt.show()
        return fig, ax

if __name__ == '__main__':
    from lattice_maker import LatticeMaker
    from neighbor import Neighbor
    from time import time
    ti.init(ti.cpu)
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

        