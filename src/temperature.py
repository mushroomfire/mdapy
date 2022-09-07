import taichi as ti


@ti.data_oriented
class AtomicTemperature:
    def __init__(self, amass, vel, verlet_list, atype_list):
        self.amass = ti.field(dtype=ti.f64, shape=(amass.shape[0]))
        self.amass.from_numpy(amass)
        self.atype_list = ti.field(dtype=ti.i32, shape=(atype_list.shape[0]))
        self.atype_list.from_numpy(atype_list)
        self.vel = ti.Vector.field(vel.shape[1], dtype=ti.f64, shape=(vel.shape[0]))
        self.vel.from_numpy(vel * 100.0)
        self.verlet_list = verlet_list
        self.N = self.vel.shape[0]
        self.T = ti.field(dtype=ti.f64, shape=(self.N))

    @ti.kernel
    def compute(self):
        """
        kb = 8.617333262145 eV / K
        kb = 1.380649×10^−23 J/K
        dim = 3
        afu = 6.022140857×10^23 1/mol
        j2e = 6.24150913 × 10^18
        """
        kb = 1.380649e-23
        dim = 3
        afu = 6.022140857e23
        max_neigh = self.verlet_list.shape[1]
        for i in range(self.N):

            # 求 atom_i 的邻域的平均速度
            v_neigh = ti.Vector([0.0, 0.0, 0.0])
            n_neigh = 0
            for j_index in range(max_neigh):
                j = self.verlet_list[i, j_index]
                if j > -1:
                    v_neigh += self.vel[j]
                    n_neigh += 1
            v_mean = v_neigh / n_neigh

            # 求邻域去除平均速度后的动能
            ke_neigh = 0.0
            for j_index in range(max_neigh):
                j = self.verlet_list[i, j_index]
                if j > -1:
                    v_j = (self.vel[j] - v_mean).norm_sqr()
                    ke_neigh += (
                        0.5 * self.amass[self.atype_list[j] - 1] / afu / 1000.0 * v_j
                    )

            # 温度转换
            self.T[i] = ke_neigh * 2.0 / dim / n_neigh / kb
