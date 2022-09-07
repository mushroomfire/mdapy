import taichi as ti
import numpy as np


@ti.data_oriented
class Neighbor:
    def __init__(self, pos, box, rc, boundary=[1, 1, 1], max_neigh=50, exclude=True):

        # 转换pos, box为ti.field
        self.pos_ti = ti.Vector.field(pos.shape[1], dtype=ti.f64, shape=(pos.shape[0]))
        self.pos_ti.from_numpy(pos)
        self.box_ti = ti.Vector.field(box.shape[1], dtype=ti.f64, shape=(box.shape[0]))
        self.box_ti.from_numpy(box)
        # 定义几个常数
        self.exclude = exclude
        self.N = self.pos_ti.shape[0]
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
        # 邻域计算
        self.verlet_list_ti = ti.field(dtype=ti.i32, shape=(self.N, max_neigh))
        self.distance_list_ti = ti.field(dtype=ti.f32, shape=(self.N, max_neigh))
        self.atom_cell_list_ti = ti.field(dtype=ti.i32, shape=(self.N))
        self.cell_id_list_ti = ti.field(
            dtype=ti.i32, shape=(self.ncel[0], self.ncel[1], self.ncel[2])
        )

    @ti.kernel
    def initinput(self):
        for I in ti.grouped(self.verlet_list_ti):
            self.verlet_list_ti[I] = -1
        for I in ti.grouped(self.distance_list_ti):
            self.distance_list_ti[I] = -1.0
        for I in ti.grouped(self.cell_id_list_ti):
            self.cell_id_list_ti[I] = -1
        for I in ti.grouped(self.atom_cell_list_ti):
            self.atom_cell_list_ti[I] = 0

    @ti.kernel
    def build_cell_ti(self):
        ti.loop_config(serialize=True)  # 需要串行
        for i in range(self.N):
            icel, jcel, kcel = ti.floor(
                (self.pos_ti[i] - self.origin) / self.bin_length, dtype=ti.i32
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
            self.atom_cell_list_ti[i] = self.cell_id_list_ti[iicel, jjcel, kkcel]
            self.cell_id_list_ti[iicel, jjcel, kkcel] = i

    @ti.func
    def pbc_ti(self, rij):
        for i in ti.static(range(3)):
            box_length = self.box_ti[i][1] - self.box_ti[i][0]
            if self.boundary[i] == 1:
                rij[i] = rij[i] - box_length * ti.round(rij[i] / box_length)
        return rij

    @ti.kernel
    def build_verlet_list_cell_ti(self):
        for i in range(self.N):
            nindex = 0
            icel, jcel, kcel = ti.floor(
                (self.pos_ti[i] - self.origin) / self.bin_length, dtype=ti.i32
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
                        j = self.cell_id_list_ti[iiicel, jjjcel, kkkcel]
                        while j > -1:
                            rij = self.pbc_ti(self.pos_ti[i] - self.pos_ti[j])
                            rijdis = rij.norm()
                            if self.exclude:
                                if rijdis < self.rc and j != i:
                                    self.verlet_list_ti[i, nindex] = j
                                    self.distance_list_ti[i, nindex] = rijdis
                                    nindex += 1
                            else:
                                if rijdis < self.rc:
                                    self.verlet_list_ti[i, nindex] = j
                                    self.distance_list_ti[i, nindex] = rijdis
                                    nindex += 1
                            j = self.atom_cell_list_ti[j]

    def compute(self):
        self.initinput()
        self.build_cell_ti()
        self.build_verlet_list_cell_ti()
        # return self.verlet_list_ti.to_numpy(), self.distance_list_ti.to_numpy()
