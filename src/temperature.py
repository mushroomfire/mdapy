import taichi as ti


@ti.data_oriented
class AtomicTemperature:
    """
    描述: 根据原子的邻域来计算原子的平均温度.
    输入参数:
    required:
    amass : (N_type)的array,N_type为原子类型的数目,按照体系的类型顺序写入相应原子类型的相对原子质量.
    vel : (Nx3)的array,体系原子的速度,N为原子数.
    verlet_list : (N, max_neigh) ti.field, 系统的邻域列表,使用Neighbor类来建立.
    atype_list: (N)的array,每一个原子的原子类型.一般可以使用system.data['type'].values
    optional:
    units : str, 采用LAMMPS中的单位,支持metal和real单位,具体见https://docs.lammps.org/units.html.
            default : 'metal'

    输出参数:
    T : (N) ti.field,每一个原子的平均温度.
    """

    def __init__(self, amass, vel, verlet_list, atype_list, units="metal"):
        self.amass = ti.field(dtype=ti.f64, shape=(amass.shape[0]))
        self.amass.from_numpy(amass)
        self.atype_list = ti.field(dtype=ti.i32, shape=(atype_list.shape[0]))
        self.atype_list.from_numpy(atype_list)
        self.vel = ti.Vector.field(vel.shape[1], dtype=ti.f64, shape=(vel.shape[0]))
        if units == "metal":
            self.vel.from_numpy(vel * 100.0)
        elif units == "real":
            self.vel.from_numpy(vel * 100000.0)
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
            v_neigh = ti.Vector([ti.float64(0.0)] * 3)
            n_neigh = 0
            for j_index in range(max_neigh):
                j = self.verlet_list[i, j_index]
                if j > -1 and j != i:
                    v_neigh += self.vel[j]
                    n_neigh += 1
            v_neigh += self.vel[i]
            n_neigh += 1
            v_mean = v_neigh / n_neigh

            # 求邻域去除平均速度后的动能
            ke_neigh = ti.float64(0.0)
            for j_index in range(max_neigh):
                j = self.verlet_list[i, j_index]
                if j > -1 and j != i:
                    v_j = (self.vel[j] - v_mean).norm_sqr()
                    ke_neigh += (
                        0.5 * self.amass[self.atype_list[j] - 1] / afu / 1000.0 * v_j
                    )
            ke_neigh += (
                0.5
                * self.amass[self.atype_list[i] - 1]
                / afu
                / 1000.0
                * (self.vel[i] - v_mean).norm_sqr()
            )

            # 温度转换
            self.T[i] = ke_neigh * 2.0 / dim / n_neigh / kb
