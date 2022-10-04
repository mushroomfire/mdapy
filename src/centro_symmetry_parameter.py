import taichi as ti
import numpy as np 

@ti.data_oriented
class CentroSymmetryParameter():
    
    def __init__(self, pos, box, boundary, verlet_list, distance_list, N):
        self.N = N
        self.pos = pos
        self.box = box
        self.boundary = boundary
        self.verlet_list = verlet_list
        self.distance_list = distance_list
        self.csp = ti.field(ti.f64, shape=(self.pos.shape[0]))
        self.sortindex = self.sort_dis()
    
    def sort_dis(self):
        dis = self.distance_list.to_numpy()
        dis[dis==-1.] = np.max(dis)+0.5
        sortindex_arr = np.argsort(dis)[:,:self.N]
        sortindex = ti.field(dtype=ti.i64, shape=(self.distance_list.shape[0], self.N))
        sortindex.from_numpy(sortindex_arr)
        return sortindex
    
    @ti.func
    def pbc(self, rij):
        for i in ti.static(range(rij.n)):
            box_length = self.box[i][1] - self.box[i][0]
            if self.boundary[i] == 1:
                rij[i] = rij[i] - box_length * ti.round(rij[i] / box_length)
        
        return rij
    
    @ti.func
    def sort_pair(self, pair):

        for i in ti.static(range(1, pair.n)):
            for j in ti.static(range(0, pair.n-i)):
                if pair[j] > pair[j+1]:
                    pair[j], pair[j + 1] = pair[j + 1], pair[j]
        return pair
               
    @ti.kernel
    def compute(self):
        for i in range(self.pos.shape[0]):  # self.pos.shape[0]
            pair = ti.Vector([0.]*int(self.N/2))
            # init pair
            for k in ti.static(range(int(self.N/2))):
                j_index = self.verlet_list[i, self.sortindex[i,0]]
                k_index = self.verlet_list[i, self.sortindex[i,k+1]] # 不能重复
                rij = self.pbc(self.pos[j_index] - self.pos[i])
                rik = self.pbc(self.pos[k_index] - self.pos[i])
                rijk = (rij+rik).norm_sqr()
                pair[k] = rijk  
            # get pair
            for j in range(self.N):
                j_index = self.verlet_list[i, self.sortindex[i,j]]
                for k in range(j+1, self.N):
                    k_index = self.verlet_list[i, self.sortindex[i,k]]
                    rij = self.pbc(self.pos[j_index] - self.pos[i])
                    rik = self.pbc(self.pos[k_index] - self.pos[i])
                    rijk = (rij+rik).norm_sqr()
                    if j == 0 and k < int(self.N/2)+1: # 排除初始数值
                        pass
                    else:
                        if rijk < pair.max():
                            pair = self.sort_pair(pair)
                            pair[pair.n-1] = rijk
            # print(i, pair)  for debug
            self.csp[i] += pair.sum()
                                    