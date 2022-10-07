import taichi as ti

@ti.data_oriented
class CentroSymmetryParameter():
    """
    计算材料中心对称参数,目前主要可以优化的地方在于2Dfield排序部分。
    """
    def __init__(self, pos, box, boundary, verlet_list, distance_list, N):
        self.N = N
        self.pos = pos
        self.box = box
        self.boundary = boundary
        self.verlet_list = verlet_list
        self.distance_list = distance_list
        self.pair = ti.field(ti.f64, shape=(self.pos.shape[0], int(self.N*(self.N-1)/2)))
        self.csp = ti.field(ti.f64, shape=(self.pos.shape[0]))
    
    @ti.func
    def pbc(self, rij):
        for i in ti.static(range(rij.n)):
            box_length = self.box[i][1] - self.box[i][0]
            if self.boundary[i] == 1:
                rij[i] = rij[i] - box_length * ti.round(rij[i] / box_length)
        return rij
    
    @ti.kernel
    def sort_2dfield_without_keys(self, values:ti.template()):
        """
        对values这个2Dfield排序。
        """
        for i in range(values.shape[0]):
            for j in range(1, values.shape[1]):
                for k in range(0, values.shape[1]-j):
                    if values[i,k] > values[i, k+1]:
                        values[i, k], values[i, k + 1] = values[i, k + 1], values[i, k]

    @ti.kernel
    def sort_2dfield_with_keys(self, values:ti.template(), keys:ti.template()):
        """
        对values这个2Dfield排序，基于values排序对keys排序。
        values.shape == keys.shape
        """
        for i in range(values.shape[0]):
            for j in range(1, values.shape[1]):
                for k in range(0, values.shape[1]-j):
                    if values[i,k] > values[i, k+1]:
                        values[i, k], values[i, k + 1] = values[i, k + 1], values[i, k]
                        keys[i, k], keys[i, k+1] = keys[i, k+1], keys[i, k]
               
    @ti.kernel
    def get_pair(self):
        for i in range(self.pos.shape[0]): 
            ncol = 0
            for j in range(self.N):
                j_index = self.verlet_list[i, j]
                for k in range(j+1, self.N):
                    k_index = self.verlet_list[i, k]
                    rij = self.pbc(self.pos[j_index] - self.pos[i])
                    rik = self.pbc(self.pos[k_index] - self.pos[i])
                    rijk = (rij+rik).norm_sqr()
                    self.pair[i, ncol] = rijk
                    ncol += 1
    
    @ti.kernel
    def get_csp(self):
        for i in range(self.pos.shape[0]):
            for j in range(int(self.N/2)):
                self.csp[i] += self.pair[i, j]
        
    def compute(self):
        self.sort_2dfield_with_keys(self.distance_list, self.verlet_list)
        self.get_pair()
        self.sort_2dfield_without_keys(self.pair)
        self.get_csp()
        
                                    