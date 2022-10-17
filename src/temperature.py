import taichi as ti
import numpy as np


@ti.data_oriented
class AtomicTemperature:
    """
    描述: 根据原子的邻域来计算原子的平均温度.
    输入参数:
    required:
    amass : (N_type)的array,N_type为原子类型的数目,按照体系的类型顺序写入相应原子类型的相对原子质量.
    vel : (Nx3)的array,体系原子的速度,N为原子数.
    verlet_list : (N, max_neigh) array, 系统的邻域列表,使用Neighbor类来建立.
    atype_list: (N)的array,每一个原子的原子类型.一般可以使用system.data['type'].values
    optional:
    units : str, 采用LAMMPS中的单位,支持metal和real单位,具体见https://docs.lammps.org/units.html.
            default : 'metal'

    输出参数:
    T : (N) array,每一个原子的平均温度.
    """

    def __init__(
        self, amass, vel, verlet_list, distance_list, atype_list, rc, units="metal"
    ):
        self.amass = amass
        self.atype_list = atype_list
        self.units = units
        if self.units == "metal":
            self.vel = vel * 100.0
        elif self.units == "real":
            self.vel = vel * 100000.0
        self.verlet_list = verlet_list
        self.distance_list = distance_list
        self.rc = rc
        self.N = self.vel.shape[0]
        self.T = np.zeros(self.N)

    @ti.kernel
    def _compute(
        self,
        verlet_list: ti.types.ndarray(),
        distance_list: ti.types.ndarray(),
        vel: ti.types.ndarray(),
        amass: ti.types.ndarray(),
        atype_list: ti.types.ndarray(),
        T: ti.types.ndarray(),
    ):
        """
        kb = 8.617333262145 eV / K
        kb = 1.380649×10^−23 J/K
        dim = 3.
        afu = 6.022140857×10^23 1/mol
        j2e = 6.24150913 × 10^18
        """
        kb = 1.380649e-23
        dim = 3.0
        afu = 6.022140857e23
        max_neigh = verlet_list.shape[1]
        for i in range(self.N):

            # 求 atom_i 的邻域的质心速度
            v_neigh = ti.Vector([ti.float64(0.0)] * 3)
            n_neigh = 0
            mass_neigh = ti.float64(0.0)
            for j_index in range(max_neigh):
                j = verlet_list[i, j_index]
                disj = distance_list[i, j_index]
                if j > -1 and j != i and disj <= self.rc:
                    j_mass = amass[atype_list[j] - 1]
                    v_neigh += ti.Vector([vel[j, 0], vel[j, 1], vel[j, 2]]) * j_mass
                    n_neigh += 1
                    mass_neigh += j_mass
            v_neigh += ti.Vector([vel[i, 0], vel[i, 1], vel[i, 2]])
            n_neigh += 1
            mass_neigh += amass[atype_list[i] - 1]
            v_mean = v_neigh / mass_neigh

            # 求邻域去除平均速度后的动能
            ke_neigh = ti.float64(0.0)
            for j_index in range(max_neigh):
                j = verlet_list[i, j_index]
                disj = distance_list[i, j_index]
                if j > -1 and j != i and disj <= self.rc:
                    v_j = (
                        ti.Vector([vel[j, 0], vel[j, 1], vel[j, 2]]) - v_mean
                    ).norm_sqr()
                    ke_neigh += 0.5 * amass[atype_list[j] - 1] / afu / 1000.0 * v_j
            ke_neigh += (
                0.5
                * amass[atype_list[i] - 1]
                / afu
                / 1000.0
                * (ti.Vector([vel[i, 0], vel[i, 1], vel[i, 2]]) - v_mean).norm_sqr()
            )

            # 温度转换
            T[i] = ke_neigh * 2.0 / dim / n_neigh / kb

    def compute(self):
        self._compute(
            self.verlet_list,
            self.distance_list,
            self.vel,
            self.amass,
            self.atype_list,
            self.T,
        )


if __name__ == "__main__":

    def init_vel(N, T, Mass):
        Boltzmann_Constant = 8.617385e-5
        np.random.seed(10086)
        x1 = np.random.random(N * 3)
        x2 = np.random.random(N * 3)
        vel = (
            np.sqrt(T * Boltzmann_Constant / Mass)
            * np.sqrt(-2 * np.log(x1))
            * np.cos(2 * np.pi * x2)
        ).reshape(N, 3)
        vel -= vel.mean(axis=0)
        return vel

    from lattice_maker import LatticeMaker
    from neighbor import Neighbor
    from time import time

    ti.init(ti.gpu, device_memory_GB=5.0)
    # ti.init(ti.cpu)
    start = time()
    lattice_constant = 4.05
    x, y, z = 100, 100, 50
    FCC = LatticeMaker(lattice_constant, "FCC", x, y, z)
    FCC.compute()
    pos = FCC.pos.to_numpy().reshape(-1, 3)
    end = time()
    print(f"Build {pos.shape[0]} atoms FCC time: {end-start} s.")
    start = time()
    box = np.array(
        [
            [0.0, lattice_constant * x],
            [0.0, lattice_constant * y],
            [0.0, lattice_constant * z],
        ]
    )
    neigh = Neighbor(pos, box, 5.0, max_neigh=60)
    neigh.compute()
    end = time()
    print(f"Build neighbor time: {end-start} s.")

    vel = init_vel(pos.shape[0], 300.0, 12.0)
    start = time()
    T = AtomicTemperature(
        np.array([12.0]),
        vel * 100.0,
        neigh.verlet_list,
        neigh.distance_list,
        np.ones(pos.shape[0], dtype=int),
        5.0,
    )

    T.compute()
    end = time()
    print(f"Calculating T time: {end-start} s.")
    print(T.T.mean())
