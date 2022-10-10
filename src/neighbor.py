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
    verlet_list : (N,max_neigh) ti.field,其中每一行代表第i个原子的邻域,其中的非负数为j原子索引.
    distance_list : (N,max_neigh) ti.field,对应的ij原子的距离.
    neighbor_number : (N) ti.field, i原子的邻域数目.
    """

    def __init__(self, pos, box, rc, boundary=[1, 1, 1], max_neigh=80, exclude=True):

        # 转换pos, box为ti.field, 此处64位精度很重要！！！
        self.pos = ti.Vector.field(pos.shape[1], dtype=ti.f64, shape=(pos.shape[0]))
        self.pos.from_numpy(pos)
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
                int(np.floor((box[i, 1] - box[i, 0]) / self.bin_length))
                for i in range(box.shape[0])
            ]
        )
        self.boundary = ti.Vector(boundary)
        self.max_neigh = max_neigh
        # 邻域计算
        self.verlet_list = ti.field(dtype=ti.i32, shape=(self.N, self.max_neigh))
        self.distance_list = ti.field(dtype=ti.f64, shape=(self.N, self.max_neigh))
        self.neighbor_number = ti.field(dtype=ti.i32, shape=(self.N))
        self.atom_cell_list = ti.field(dtype=ti.i32, shape=(self.N))
        self.cell_id_list = ti.field(
            dtype=ti.i32, shape=(self.ncel[0], self.ncel[1], self.ncel[2])
        )

    @ti.kernel
    def initinput(self):
        for I in ti.grouped(self.verlet_list):
            self.verlet_list[I] = -1
        for I in ti.grouped(self.distance_list):
            self.distance_list[I] = self.rc + 1.0
        for I in ti.grouped(self.cell_id_list):
            self.cell_id_list[I] = -1
        for I in ti.grouped(self.atom_cell_list):
            self.atom_cell_list[I] = 0

    @ti.kernel
    def build_cell(self):
        ti.loop_config(serialize=True)  # 需要串行
        for i in range(self.N):
            icel, jcel, kcel = ti.floor(
                (self.pos[i] - self.origin) / self.bin_length, dtype=ti.i32
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
            self.atom_cell_list[i] = self.cell_id_list[iicel, jjcel, kkcel]
            self.cell_id_list[iicel, jjcel, kkcel] = i

    @ti.func
    def pbc(self, rij):
        for i in ti.static(range(rij.n)):
            box_length = self.box[i][1] - self.box[i][0]
            if self.boundary[i] == 1:
                rij[i] = rij[i] - box_length * ti.round(rij[i] / box_length)
        return rij

    @ti.kernel
    def build_verlet_list(self):
        for i in range(self.N):
            nindex = 0
            icel, jcel, kcel = ti.floor(
                (self.pos[i] - self.origin) / self.bin_length, dtype=ti.i32
            )
            for iicel in range(icel - 1, icel + 2):
                for jjcel in range(jcel - 1, jcel + 2):
                    for kkcel in range(kcel - 1, kcel + 2):
                        iiicel = iicel
                        jjjcel = jjcel
                        kkkcel = kkcel
                        if iicel < 0:
                            iiicel += self.ncel[0]
                        elif iicel > self.ncel[0] - 1:
                            iiicel -= self.ncel[0]
                        if jjcel < 0:
                            jjjcel += self.ncel[1]
                        elif jjcel > self.ncel[1] - 1:
                            jjjcel -= self.ncel[1]
                        if kkcel < 0:
                            kkkcel += self.ncel[2]
                        elif kkcel > self.ncel[2] - 1:
                            kkkcel -= self.ncel[2]
                        j = self.cell_id_list[iiicel, jjjcel, kkkcel]
                        while j > -1:
                            rij = self.pbc(self.pos[i] - self.pos[j])
                            rijdis = rij.norm()
                            if self.exclude:
                                if rijdis < self.rc and j != i:
                                    self.verlet_list[i, nindex] = j
                                    self.distance_list[i, nindex] = rijdis
                                    nindex += 1
                            else:
                                if rijdis < self.rc:
                                    self.verlet_list[i, nindex] = j
                                    self.distance_list[i, nindex] = rijdis
                                    nindex += 1
                            j = self.atom_cell_list[j]
            self.neighbor_number[i] = nindex

    def check_boundary(self):
        @ti.kernel
        def panduan(arr: ti.template(), max_neigh: int) -> int:
            jishu = 0
            for I in arr:
                if arr[I] >= max_neigh:
                    jishu += 1
            return jishu

        assert (
            panduan(self.neighbor_number, self.max_neigh) == 0
        ), "Neighbor number exceeds max_neigh, which should be increased!"

    def compute(self):
        self.initinput()
        self.build_cell()
        self.build_verlet_list()
        self.check_boundary()
