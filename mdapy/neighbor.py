# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import taichi as ti
import numpy as np


@ti.data_oriented
class Neighbor:
    """This module is used to cerate neighbor of atoms. Using linked-cell list method make fast neighbor finding possible.

    Args:
        pos (np.ndarray): (Nparticles x 3) particles positions.

        box (np.ndarray): (3 x 2) system box, must be rectangle.

        rc (float): cutoff ditances.

        boundary (list, optional): boundary conditions, 1 is periodic and 0 is free boundary. Defaults to [1, 1, 1].

        max_neigh (int, optional): a given maximum neighbor number per atoms. Defaults to 80.

        exclude (bool, optional): whether include atom self, True means no including. Defaults to True.

    Returns:
        verlet_list (np.ndarray): (Nparticles x max_neigh) verlet_list[i, j] means j atom is a neighbor of i atom if j > -1.

        distance_list (np.ndarray): (Nparticles x max_neigh) distance_list[i, j] means distance between i and j atom.

        neighbor_number (np.ndarray): (Nparticles) neighbor atoms number.
    """

    def __init__(self, pos, box, rc, boundary=[1, 1, 1], max_neigh=80, exclude=True):

        self.pos = pos
        self.box = ti.Vector.field(box.shape[1], dtype=ti.f64, shape=(box.shape[0]))
        self.box.from_numpy(box)
        # define several const variables
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
        )  # calculate small system
        self.boundary = ti.Vector(boundary)
        self.max_neigh = max_neigh
        # neighbor related array
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
        ti.loop_config(serialize=True)  # serial for loop
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
            iicel, jjcel, kkcel = icel, jcel, kcel  # make sure correct cell
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
