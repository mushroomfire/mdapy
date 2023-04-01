# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import taichi as ti
import numpy as np

if __name__ == "__main__":
    from neigh._neigh import _build_cell
else:
    from _neigh import _build_cell

vec3f32 = ti.types.vector(3, ti.f32)
vec3f64 = ti.types.vector(3, ti.f64)


@ti.data_oriented
class Neighbor:
    """This module is used to cerate neighbor of atoms within a given cutoff distance.
    Using linked-cell list method makes fast neighbor finding possible.

    Args:
        pos (np.ndarray): (:math:`N_p, 3`) particles positions.
        box (np.ndarray): (:math:`3, 2`) system box, must be rectangle.
        rc (float): cutoff distance.
        boundary (list, optional): boundary conditions, 1 is periodic and 0 is free boundary. Defaults to [1, 1, 1].
        max_neigh (int, optional): a given maximum neighbor number per atoms. Defaults to 80.
        exclude (bool, optional): whether include atom self, True means no including. Defaults to True.

    Outputs:
        - **verlet_list** (np.ndarray) - (:math:`N_p, max\_neigh`) verlet_list[i, j] means j atom is a neighbor of i atom if j > -1.
        - **distance_list** (np.ndarray) - (:math:`N_p, max\_neigh`) distance_list[i, j] means distance between i and j atom.
        - **neighbor_number** (np.ndarray) - (:math:`N_p`) neighbor atoms number.

    Examples:
        >>> import mdapy as mp

        >>> mp.init()

        >>> import numpy as np

        >>> FCC = mp.LatticeMaker(3.615, 'FCC', 10, 10, 10) # Create a FCC structure

        >>> FCC.compute() # Get atom positions

        >>> neigh = mp.Neighbor(FCC.pos, FCC.box,
                                5., max_neigh=80) # Initialize Neighbor class.

        >>> neigh.compute() # Calculate particle neighbor information.

        >>> neigh.verlet_list # Check the neighbor index.

        >>> neigh.distance_list # Check the neighbor distance.

        >>> neigh.neighbor_number # Check the neighbor atom number.
    """

    def __init__(self, pos, box, rc, boundary=[1, 1, 1], max_neigh=80, exclude=True):

        assert pos.dtype in [
            np.float64,
            np.float32,
        ], "Dtype of pos must in [float64, float32]."
        self.pos = pos
        self.box = box
        self.boundary = ti.Vector([boundary[i] for i in range(3)], int)
        self.N = self.pos.shape[0]
        self.rc = rc
        self.bin_length = self.rc + 0.5
        self._corigin = np.ascontiguousarray(box[:, 0])
        if self.pos.dtype == np.float64:
            self.origin = vec3f64([box[i, 0] for i in range(3)])
            self.box_length = vec3f64([box[i, 1] - box[i, 0] for i in range(3)])
        elif self.pos.dtype == np.float32:
            self.origin = vec3f32([box[i, 0] for i in range(3)])
            self.box_length = vec3f32([box[i, 1] - box[i, 0] for i in range(3)])

        self._cncel = np.array(
            [
                i if i > 3 else 3
                for i in [
                    int(np.floor((box[i, 1] - box[i, 0]) / self.bin_length))
                    for i in range(box.shape[0])
                ]
            ]
        )
        self.ncel = ti.Vector([self._cncel[0], self._cncel[1], self._cncel[2]], int)
        self.max_neigh = max_neigh
        self.exclude = exclude
        self.verlet_list = np.zeros((self.N, self.max_neigh), dtype=np.int32) - 1
        self.distance_list = (
            np.zeros((self.N, self.max_neigh), dtype=pos.dtype) + self.rc + 1.0
        )
        self.neighbor_number = np.zeros(self.N, dtype=np.int32)
        self._if_computed = False

    @ti.func
    def _pbc(self, rij):

        for m in ti.static(range(3)):
            if self.boundary[m]:
                dx = rij[m]
                x_size = self.box_length[m]
                h_x_size = x_size * 0.5
                if dx > h_x_size:
                    dx = dx - x_size
                if dx <= -h_x_size:
                    dx = dx + x_size
                rij[m] = dx
        return rij

    @ti.kernel
    def _build_verlet_list(
        self,
        pos: ti.types.ndarray(dtype=ti.math.vec3),
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
                            rij = self._pbc(pos[j] - pos[i])
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
    def _build_verlet_list_small(
        self,
        pos: ti.types.ndarray(dtype=ti.math.vec3),
        verlet_list: ti.types.ndarray(),
        distance_list: ti.types.ndarray(),
        neighbor_number: ti.types.ndarray(),
    ):

        ti.loop_config(serialize=True)
        for i in range(self.N):
            nindex = 0
            for j in range(self.N):
                rij = self._pbc(pos[i] - pos[j])
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
        """Do the real neighbor calculation."""

        if self.N > 1000:
            atom_cell_list = np.zeros(self.N, dtype=np.int32)
            cell_id_list = (
                np.zeros((self.ncel[0], self.ncel[1], self.ncel[2]), dtype=np.int32) - 1
            )
            _build_cell(
                self.pos,
                atom_cell_list,
                cell_id_list,
                self._corigin,
                self._cncel,
                self.bin_length,
            )
            self._build_verlet_list(
                self.pos,
                atom_cell_list,
                cell_id_list,
                self.verlet_list,
                self.distance_list,
                self.neighbor_number,
            )
        else:
            self._build_verlet_list_small(
                self.pos, self.verlet_list, self.distance_list, self.neighbor_number
            )
        max_neighbor_number = self.neighbor_number.max()
        assert (
            max_neighbor_number <= self.max_neigh
        ), f"Neighbor number exceeds max_neigh, which should be larger than {max_neighbor_number}!"
        self._if_computed = True

    @ti.kernel
    def _partition_select_sort(
        self, indices: ti.types.ndarray(), keys: ti.types.ndarray(), N: int
    ):
        """This function sorts N-th minimal value in keys.

        Args:
            indices (ti.types.ndarray): indices.
            keys (ti.types.ndarray): values to be sorted.
            N (int): number of sorted values.
        """
        for i in range(indices.shape[0]):
            for j in range(N):
                minIndex = j
                for k in range(j + 1, indices.shape[1]):
                    if keys[i, k] < keys[i, minIndex]:
                        minIndex = k
                if minIndex != j:
                    keys[i, minIndex], keys[i, j] = keys[i, j], keys[i, minIndex]
                    indices[i, minIndex], indices[i, j] = (
                        indices[i, j],
                        indices[i, minIndex],
                    )

    def sort_verlet_by_distance(self, N: int):
        """This function sorts the first N-th verlet_list by distance_list.

        Args:
            N (int): number of sorted values
        """
        if not self._if_computed:
            self.compute()
        self._partition_select_sort(self.verlet_list, self.distance_list, N)


if __name__ == "__main__":
    from lattice_maker import LatticeMaker
    from time import time

    # ti.init(ti.gpu, device_memory_GB=4.0)
    ti.init(ti.cpu)
    start = time()
    lattice_constant = 3.615
    x, y, z = 100, 100, 125
    FCC = LatticeMaker(lattice_constant, "FCC", x, y, z)
    FCC.compute()
    end = time()
    print(f"Build {FCC.pos.shape[0]} atoms FCC time: {end-start} s.")
    start = time()
    neigh = Neighbor(
        FCC.pos,
        np.array(FCC.box, np.float32),
        5.0,
        max_neigh=50,
        exclude=True,
    )
    neigh.compute()
    end = time()
    print(neigh.ncel)
    print(f"Build neighbor time: {end-start} s.")
    # print(neigh.verlet_list[0])
    # print(neigh.distance_list[0])
    # print(neigh.neighbor_number.max())

    # neigh.sort_verlet_by_distance(12)
    # print(neigh.verlet_list[0])
    # print(neigh.distance_list[0])
