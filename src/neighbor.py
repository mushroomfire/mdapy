import taichi as ti
import numpy as np


@ti.data_oriented
class Neighbor:

    """
    描述: 用于生成体系的领域列表,也就是距离小于截断半径的原子集合.
    输入参数:
    required:
    pos : (Nx3)的array,代表原子位置坐标,N为原子数目.
    box : (3x2)的array,目前代码仅支持3维正交盒子,第一列为盒子最小值,第二列为盒子最大值.
    rc : float,邻域的截断半径距离,单位为\AA.
    optional:
    boundary : (3x1)的list,表示模拟体系的边界条件,其中1代表周期性边界,0代表自由边界.
            default : [1, 1, 1]
    max_neigh : int,每一个原子的最大邻域原子数目,如果rc过大的话,需要提高这个数值,但是需要更大的内存分配.
            default : 50
    exclude : bool,代表邻域列表中是否包含自身.
            default : True (不包含自身)
    输出参数:
    verlet_list : (N,max_neigh) array,其中每一行代表第i个原子的邻域,其中的非负数为j原子索引.
    distance_list : (N,max_neigh) array,对应的ij原子的距离.
    neighbor_number : (N) array, i原子的邻域数目.
    """

    def __init__(self, pos, box, rc, boundary=[1, 1, 1], max_neigh=80, exclude=True):

        self.pos = pos
        self.box = ti.Vector.field(box.shape[1], dtype=ti.f64, shape=(box.shape[0]))
        self.box.from_numpy(box)
        # 定义几个常数
        self.exclude = exclude
        self.N = self.pos.shape[0]
        self.rc = rc
        self.bin_length = self.rc + 0.5
        self.origin = ti.Vector([box[0, 0], box[1, 0], box[2, 0]])
        self.ncel = ti.Vector(
            [
                i if i > 3 else 3
                for i in [
                    int(np.floor((box[i, 1] - box[i, 0]) / self.bin_length))
                    for i in range(box.shape[0])
                ]
            ]
        )  # 使小体系也可以计算
        self.boundary = ti.Vector(boundary)
        self.max_neigh = max_neigh
        # 邻域计算
        self.verlet_list = np.zeros((self.N, self.max_neigh), dtype=np.int32) - 1
        self.distance_list = (
            np.zeros((self.N, self.max_neigh), dtype=np.float64) + self.rc + 1.0
        )
        self.neighbor_number = np.zeros(self.N, dtype=np.int32)

    @ti.kernel
    def build_cell(
        self,
        pos: ti.types.ndarray(),
        atom_cell_list: ti.types.ndarray(),
        cell_id_list: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=True)  # 需要串行
        for i in range(self.N):
            r_i = ti.Vector([pos[i, 0], pos[i, 1], pos[i, 2]])
            icel, jcel, kcel = ti.floor(
                (r_i - self.origin) / self.bin_length, dtype=ti.i32
            )
            iicel, jjcel, kkcel = icel, jcel, kcel
            if icel < 0:
                iicel = 0
            elif icel > self.ncel[0] - 1:
                iicel = self.ncel[0] - 1
            if jcel < 0:
                jjcel = 0
            elif jcel > self.ncel[1] - 1:
                jjcel = self.ncel[1] - 1
            if kcel < 0:
                kkcel = 0
            elif kcel > self.ncel[2] - 1:
                kkcel = self.ncel[2] - 1

            atom_cell_list[i] = cell_id_list[iicel, jjcel, kkcel]
            cell_id_list[iicel, jjcel, kkcel] = i

    @ti.func
    def pbc(self, rij):
        for i in ti.static(range(rij.n)):
            if self.boundary[i] == 1:
                box_length = self.box[i][1] - self.box[i][0]
                rij[i] = rij[i] - box_length * ti.round(rij[i] / box_length)
        return rij

    @ti.kernel
    def build_verlet_list(
        self,
        pos: ti.types.ndarray(element_dim=1),
        atom_cell_list: ti.types.ndarray(),
        cell_id_list: ti.types.ndarray(),
        verlet_list: ti.types.ndarray(),
        distance_list: ti.types.ndarray(),
        neighbor_number: ti.types.ndarray(),
    ):
        for i in range(self.N):
            nindex = 0
            icel, jcel, kcel = ti.floor(
                (pos[i] - self.origin) / self.bin_length, dtype=ti.i32
            )
            iicel, jjcel, kkcel = icel, jcel, kcel  # 这一段用于确保所处正确的cell
            if icel < 0:
                iicel = 0
            elif icel > self.ncel[0] - 1:
                iicel = self.ncel[0] - 1
            if jcel < 0:
                jjcel = 0
            elif jcel > self.ncel[1] - 1:
                jjcel = self.ncel[1] - 1
            if kcel < 0:
                kkcel = 0
            elif kcel > self.ncel[2] - 1:
                kkcel = self.ncel[2] - 1
            for iiicel in range(iicel - 1, iicel + 2):
                for jjjcel in range(jjcel - 1, jjcel + 2):
                    for kkkcel in range(kkcel - 1, kkcel + 2):
                        iiiicel = iiicel
                        jjjjcel = jjjcel
                        kkkkcel = kkkcel
                        if iiicel < 0:
                            iiiicel += self.ncel[0]
                        elif iiicel > self.ncel[0] - 1:
                            iiiicel -= self.ncel[0]
                        if jjjcel < 0:
                            jjjjcel += self.ncel[1]
                        elif jjjcel > self.ncel[1] - 1:
                            jjjjcel -= self.ncel[1]
                        if kkkcel < 0:
                            kkkkcel += self.ncel[2]
                        elif kkkcel > self.ncel[2] - 1:
                            kkkkcel -= self.ncel[2]
                        j = cell_id_list[iiiicel, jjjjcel, kkkkcel]
                        while j > -1:
                            rij = self.pbc(pos[j] - pos[i])
                            rijdis = rij.norm()
                            if self.exclude:
                                if rijdis < self.rc and j != i:
                                    verlet_list[i, nindex] = j
                                    distance_list[i, nindex] = rijdis
                                    nindex += 1
                            else:
                                if rijdis < self.rc:
                                    verlet_list[i, nindex] = j
                                    distance_list[i, nindex] = rijdis
                                    nindex += 1
                            j = atom_cell_list[j]
            neighbor_number[i] = nindex

    @ti.kernel
    def build_verlet_list_small(
        self,
        pos: ti.types.ndarray(element_dim=1),
        verlet_list: ti.types.ndarray(),
        distance_list: ti.types.ndarray(),
        neighbor_number: ti.types.ndarray(),
    ):

        ti.loop_config(serialize=True)
        for i in range(self.N):
            nindex = 0
            for j in range(self.N):
                rij = self.pbc(pos[i] - pos[j])
                rijdis = rij.norm()
                if self.exclude:
                    if rijdis < self.rc and j != i:
                        verlet_list[i, nindex] = j
                        distance_list[i, nindex] = rijdis
                        nindex += 1
                else:
                    if rijdis < self.rc:
                        verlet_list[i, nindex] = j
                        distance_list[i, nindex] = rijdis
                        nindex += 1
            neighbor_number[i] = nindex

    def compute(self):
        if self.N > 1000:
            atom_cell_list = np.zeros(self.N, dtype=np.int32)
            cell_id_list = (
                np.zeros((self.ncel[0], self.ncel[1], self.ncel[2]), dtype=np.int32) - 1
            )
            self.build_cell(self.pos, atom_cell_list, cell_id_list)
            self.build_verlet_list(
                self.pos,
                atom_cell_list,
                cell_id_list,
                self.verlet_list,
                self.distance_list,
                self.neighbor_number,
            )
        else:
            self.build_verlet_list_small(
                self.pos, self.verlet_list, self.distance_list, self.neighbor_number
            )
        max_neighbor_number = self.neighbor_number.max()
        assert (
            max_neighbor_number <= self.max_neigh
        ), f"Neighbor number exceeds max_neigh, which should be larger than {max_neighbor_number}!"


if __name__ == "__main__":
    from lattice_maker import LatticeMaker
    from time import time

    # ti.init(
    #     ti.gpu, device_memory_GB=5.0, packed=True, offline_cache=True
    # )  # , offline_cache=True)
    ti.init(ti.cpu)
    start = time()
    lattice_constant = 4.05
    x, y, z = 100, 100, 100
    FCC = LatticeMaker(lattice_constant, "FCC", x, y, z)
    FCC.compute()
    end = time()
    print(f"Build {FCC.pos.shape[0]} atoms FCC time: {end-start} s.")
    start = time()
    neigh = Neighbor(FCC.pos, FCC.box, 3.0, max_neigh=13, exclude=True)
    print(neigh.ncel)
    neigh.compute()
    end = time()
    print(f"Build neighbor time: {end-start} s.")
    print(neigh.verlet_list[0])
    print(neigh.distance_list[0])
    print(neigh.neighbor_number[0])
