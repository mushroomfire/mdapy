import taichi as ti

@ti.data_oriented
class AtomicEntropy():
    def __init__(self, box, distance_list, rc, sigma=0.25, use_local_density=False):
        self.box = box.to_numpy()
        self.distance_list = distance_list
        self.rc = rc 
        self.sigma = sigma
        self.use_local_density = use_local_density
        self.N, self.vol, self.global_density, self.nbins, self.rlist, self.rlist_sq, self.prefactor = self.initinput()
        self.entropy = ti.field(dtype=ti.f64, shape=(self.N))
        
        
    def initinput(self):
        N = self.distance_list.shape[0]
        box_l = self.box[:,1]-self.box[:,0]
        vol = box_l[0]*box_l[1]*box_l[2]
        global_density = N / vol
        nbins = ti.floor(self.rc / self.sigma) + 1
        interval = self.rc / (nbins-1.)
        
        rlist = ti.field(dtype=ti.f64, shape=(nbins))
        rlist_sq = ti.field(dtype=ti.f64, shape=(nbins))
        prefactor = ti.field(dtype=ti.f64, shape=(nbins))
        
        for i in range(nbins):
            rlist[i] = i*interval
            rlist_sq[i] = (i*interval)**2
            prefactor[i] = (i*interval)**2 * (4 * ti.math.pi * global_density * ti.sqrt(2 * ti.math.pi * self.sigma**2))

        prefactor[0] = prefactor[1]
        return N, vol, global_density, nbins, rlist, rlist_sq, prefactor
    
    @ti.kernel
    def compute(self):
        
        for i in range(self.N):
            g_m = ti.Vector([0.]*self.nbins)
            intergrad = ti.Vector([0.]*self.nbins)
            n_neigh = 0
            for j in ti.static(range(self.nbins)):
                for k in range(self.distance_list.shape[1]):
                    if self.distance_list[i, k] > -1:
                        g_m[j] += ti.exp(-(self.rlist[j] - self.distance_list[i, k])**2 / (2.0*self.sigma**2)) / self.prefactor[j]
                        n_neigh += 1
                        
            density = 0.
            if self.use_local_density:
                local_vol = 4/3 * ti.math.pi * self.rc**3
                density = n_neigh / local_vol
                g_m *= self.global_density / density
            else:
                density = self.global_density
                
            for j in ti.static(range(self.nbins)):
                if g_m[j] >= 1e-10:
                    intergrad[j] = (g_m[j] * ti.log(g_m[j]) - g_m[j] + 1.0)*self.rlist_sq[j]
                else:
                    intergrad[j] = self.rlist_sq[j]
                    
            sum_intergrad = 0.
            for j in ti.static(range(self.nbins-1)):
                sum_intergrad += (intergrad[j] + intergrad[j+1]) * (self.rlist[j+1] - self.rlist[j])
                
            self.entropy[i] = -ti.math.pi * density * sum_intergrad
            