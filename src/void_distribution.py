import taichi as ti
import numpy as np
from mdapy.src.neighbor import Neighbor
from mdapy.src.cluser_analysis import ClusterAnalysis


@ti.data_oriented
class VoidDistribution:
    """
    计算固体中空洞的数目,体积,空间分布.
    input :
    pos : ndarray
    box : (3x2) ndarray
    cell_length : float, 应该要比晶胞长度大一点
    boundary : list, [1, 1, 1]
    out_void : bool, False 是否导出void的坐标
    head : dump文件的head, 不指定会默认生成
    out_name : void坐标文件的名字, 默认为void.dump
    """

    def __init__(
        self,
        pos,
        box,
        cell_length,
        boundary=[1, 1, 1],
        out_void=False,
        head=None,
        out_name="void.dump",
    ):

        self.pos = pos
        self.N = self.pos.shape[0]
        self.box = box
        self.cell_length = cell_length
        self.boundary = boundary
        self.out_void = out_void
        self.head = head
        self.out_name = out_name
        self.origin = ti.Vector([box[0, 0], box[1, 0], box[2, 0]])
        self.ncel = ti.Vector(
            [
                i if i > 3 else 3
                for i in [
                    int(np.floor((box[i, 1] - box[i, 0]) / self.cell_length))
                    for i in range(box.shape[0])
                ]
            ]
        )

    @ti.kernel
    def fill_cell(
        self,
        pos: ti.types.ndarray(element_dim=1),
        cell_id_list: ti.types.ndarray(),
        id_list: ti.types.ndarray(),
    ):
        for i in range(self.N):
            r_i = pos[i]
            icel, jcel, kcel = ti.floor(
                (r_i - self.origin) / self.cell_length, dtype=ti.i32
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
            id_list[iicel, jjcel, kkcel] = 1

        # ti.loop_config(serialize=True)
        for iicel, jjcel, kkcel in ti.ndrange(self.ncel[0], self.ncel[1], self.ncel[2]):
            num = 0
            for i, j, k in ti.ndrange(
                (iicel - 1, iicel + 2), (jjcel - 1, jjcel + 2), (kkcel - 1, kkcel + 2)
            ):
                ii, jj, kk = i, j, k
                if i < 0:
                    ii += self.ncel[0]
                elif i > self.ncel[0] - 1:
                    ii = self.ncel[0] - 1
                if j < 0:
                    jj += self.ncel[1]
                elif j > self.ncel[1] - 1:
                    jj = self.ncel[1] - 1
                if k < 0:
                    kk += self.ncel[2]
                elif k > self.ncel[2] - 1:
                    kk = self.ncel[2] - 1
                num += id_list[ii, jj, kk]
            if num < 26:  # 20 is empty
                cell_id_list[iicel, jjcel, kkcel] = 1  # is void
            # else:
            #     cell_id_list[iicel, jjcel, kkcel] = 0  # is point defect

    def write_void_pos(self, void_data):

        if self.head is None:
            boundary_str = ["pp" if i == 1 else "ff" for i in self.boundary]
            head = [
                "ITEM: TIMESTEP\n",
                "0\n",
                "ITEM: NUMBER OF ATOMS\n",
                f"{void_data.shape[0]}\n",
                "ITEM: BOX BOUNDS {} {} {}\n".format(*boundary_str),
                f"{self.box[0, 0]} {self.box[0, 1]}\n",
                f"{self.box[1, 0]} {self.box[1, 1]}\n",
                f"{self.box[2, 0]} {self.box[2, 1]}\n",
                "ITEM: ATOMS id type x y z cluster_id\n",
            ]
        else:
            head = self.head.copy()
            head[3] = f"{void_data.shape[0]}\n"
            head[-1] = "ITEM: ATOMS id type x y z cluster_id\n"

        with open(self.out_name, "w") as op:
            op.write("".join(head))
            np.savetxt(op, void_data, fmt="%d %d %f %f %f %d", delimiter=" ")

    def compute(self):
        cell_id_list = np.zeros(
            (self.ncel[0], self.ncel[1], self.ncel[2]), dtype=np.int32
        )
        id_list = np.zeros((self.ncel[0], self.ncel[1], self.ncel[2]), dtype=np.int32)
        self.fill_cell(self.pos, cell_id_list, id_list)

        if 1 in cell_id_list:
            void_pos = np.argwhere(cell_id_list == 1) * self.cell_length
            void_pos += self.origin.to_numpy()
            # np.array([self.cell_length] * 3) - self.origin.to_numpy()

            neigh = Neighbor(void_pos, self.box, self.cell_length * 1.1, self.boundary)
            neigh.compute()
            cluster = ClusterAnalysis(
                self.cell_length * 1.1, neigh.verlet_list, neigh.distance_list
            )
            cluster.compute()
            self.void_number = cluster.cluster_number
            self.void_volume = void_pos.shape[0] * self.cell_length**3

            if self.out_void:
                void_data = np.c_[
                    np.arange(void_pos.shape[0]) + 1,
                    np.ones(void_pos.shape[0]),
                    void_pos,
                    cluster.particleClusters,
                ]
                self.write_void_pos(void_data)
        else:
            self.void_number = 0
            self.void_volume = 0.0


if __name__ == "__main__":

    from lattice_maker import LatticeMaker
    from time import time

    # ti.init(ti.gpu, device_memory_GB=5.0)
    ti.init(ti.cpu)
    start = time()
    lattice_constant = 4.05
    x, y, z = 50, 50, 50
    FCC = LatticeMaker(lattice_constant, "FCC", x, y, z)
    FCC.compute()
    end = time()
    print(f"Build {FCC.pos.shape[0]} atoms FCC time: {end-start} s.")
    pos = FCC.pos.copy()

    pos = pos[np.sum(np.square(pos - np.array([50, 50, 50])), axis=1) > 100]
    pos = pos[np.sum(np.square(pos - np.array([100, 100, 100])), axis=1) > 100]
    pos = pos[np.sum(np.square(pos - np.array([150, 150, 150])), axis=1) > 400]
    pos = pos[np.sum(np.square(pos - np.array([50, 150, 50])), axis=1) > 400]

    # FCC.pos = pos
    # FCC.N = pos.shape[0]
    # FCC.write_data()
    print("Generate four voids.")

    start = time()
    void = VoidDistribution(pos, FCC.box, lattice_constant + 1.0, out_void=False)
    void.compute()
    end = time()
    print("void number is:", void.void_number)
    print("void volume is:", void.void_volume)
    print("real valume is:", 4 / 3 * np.pi * (2 * 10**3 + 2 * 20**3))
    print(f"Calculate void time: {end-start} s.")
