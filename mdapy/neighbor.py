# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import taichi as ti
import numpy as np

try:
    from _neigh import (
        _build_cell_rec,
        _build_cell_rec_with_jishu,
        _build_cell_tri,
        _build_cell_tri_with_jishu,
    )
except Exception:
    from neigh._neigh import (
        _build_cell_rec,
        _build_cell_rec_with_jishu,
        _build_cell_tri,
        _build_cell_tri_with_jishu,
    )


@ti.data_oriented
class Neighbor:
    """This module is used to cerate neighbor of atoms within a given cutoff distance.
    Using linked-cell list method makes fast neighbor finding possible.

    Args:
        pos (np.ndarray): (:math:`N_p, 3`) particles positions.
        box (np.ndarray): (:math:`4, 3`) system box, must be rectangle.
        rc (float): cutoff distance.
        boundary (list, optional): boundary conditions, 1 is periodic and 0 is free boundary. Defaults to [1, 1, 1].
        max_neigh (int, optional): maximum neighbor number. If not given, will estimate atomatically. Default to None.

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
                                5.) # Initialize Neighbor class.

        >>> neigh.compute() # Calculate particle neighbor information.

        >>> neigh.verlet_list # Check the neighbor index.

        >>> neigh.distance_list # Check the neighbor distance.

        >>> neigh.neighbor_number # Check the neighbor atom number.
    """

    def __init__(self, pos, box, rc, boundary=[1, 1, 1], max_neigh=None):

        if pos.dtype != np.float64:
            self.pos = pos.astype(np.float64)
        else:
            self.pos = pos
        if box.dtype != np.float64:
            box = box.astype(np.float64)
        if box.shape == (3, 2):
            self.box = np.zeros((4, 3), dtype=box.dtype)
            self.box[0, 0], self.box[1, 1], self.box[2, 2] = box[:, 1] - box[:, 0]
            self.box[-1] = box[:, 0]
        elif box.shape == (4, 3):
            self.box = box
        assert self.box[0, 1] == 0
        assert self.box[0, 2] == 0
        assert self.box[1, 2] == 0
        self.rc = rc
        self.boundary = ti.Vector([int(boundary[i]) for i in range(3)])
        self.bin_length = self.rc + 0.01
        self.max_neigh = max_neigh
        self.box_length = ti.Vector([np.linalg.norm(box[i]) for i in range(3)])

        self.N = self.pos.shape[0]
        for i, direc in zip(range(3), ["x", "y", "z"]):
            if self.boundary[i] == 1:
                assert (
                    self.box_length[i] / 2 > self.bin_length
                ), f"box length along {direc} direction with periodic boundary should be two times larger than rc."

        self.ncel = ti.Vector(
            [
                max(int(np.floor(np.linalg.norm(box[i]) / self.bin_length)), 3)
                for i in range(3)
            ]
        )
        self.if_computed = False

        self.rec = True
        if self.box[1, 0] != 0 or self.box[2, 0] != 0 or self.box[2, 1] != 0:
            self.rec = False

    @ti.func
    def _pbc_rec(self, rij):
        for m in ti.static(range(3)):
            if self.boundary[m] == 1:
                dx = rij[m]
                x_size = self.box_length[m]
                h_x_size = x_size * 0.5
                if dx > h_x_size:
                    dx = dx - x_size
                if dx <= -h_x_size:
                    dx = dx + x_size
                rij[m] = dx
        return rij

    @ti.func
    def _pbc(self, rij, box: ti.types.ndarray(element_dim=1)) -> ti.math.vec3:
        nz = rij[2] / box[2][2]
        ny = (rij[1] - nz * box[2][1]) / box[1][1]
        nx = (rij[0] - ny * box[1][0] - nz * box[2][0]) / box[0][0]
        n = ti.Vector([nx, ny, nz])
        for i in ti.static(range(3)):
            if self.boundary[i] == 1:
                if n[i] > 0.5:
                    n[i] -= 1
                elif n[i] < -0.5:
                    n[i] += 1
        return n[0] * box[0] + n[1] * box[1] + n[2] * box[2]

    @ti.kernel
    def _build_verlet_list(
        self,
        pos: ti.types.ndarray(element_dim=1),
        atom_cell_list: ti.types.ndarray(),
        cell_id_list: ti.types.ndarray(),
        verlet_list: ti.types.ndarray(),
        distance_list: ti.types.ndarray(),
        neighbor_number: ti.types.ndarray(),
        box: ti.types.ndarray(element_dim=1),
    ):
        rcsq = self.rc * self.rc

        for i in range(pos.shape[0]):
            nindex = 0
            icel, jcel, kcel = 0, 0, 0
            rij = ti.Vector([0.0, 0.0, 0.0], ti.f64)
            if ti.static(self.rec):
                icel, jcel, kcel = ti.floor(
                    (pos[i] - box[3]) / self.bin_length, dtype=int
                )
            else:
                r = pos[i] - box[3]
                nz = r[2] / box[2][2]
                ny = (r[1] - nz * box[2][1]) / box[1][1]
                nx = (r[0] - ny * box[1][0] - nz * box[2][0]) / box[0][0]
                icel = ti.floor((nx * box[0]).norm() / self.bin_length, int)
                jcel = ti.floor((ny * box[1]).norm() / self.bin_length, int)
                kcel = ti.floor((nz * box[2]).norm() / self.bin_length, int)
            # iicel, jjcel, kkcel = icel, jcel, kcel  # make sure correct cell
            if icel < 0:
                icel = 0
            elif icel > self.ncel[0] - 1:
                icel = self.ncel[0] - 1
            if jcel < 0:
                jcel = 0
            elif jcel > self.ncel[1] - 1:
                jcel = self.ncel[1] - 1
            if kcel < 0:
                kcel = 0
            elif kcel > self.ncel[2] - 1:
                kcel = self.ncel[2] - 1
            for iicel in range(icel - 1, icel + 2):
                for jjcel in range(jcel - 1, jcel + 2):
                    for kkcel in range(kcel - 1, kcel + 2):
                        j = cell_id_list[
                            iicel % self.ncel[0],
                            jjcel % self.ncel[1],
                            kkcel % self.ncel[2],
                        ]
                        while j > -1:
                            if ti.static(self.rec):
                                rij = self._pbc_rec(pos[j] - pos[i])
                            else:
                                rij = self._pbc(pos[j] - pos[i], box)
                            rijdis_sq = rij[0] ** 2 + rij[1] ** 2 + rij[2] ** 2
                            if rijdis_sq <= rcsq and j != i:
                                verlet_list[i, nindex] = j
                                distance_list[i, nindex] = ti.sqrt(rijdis_sq)
                                nindex += 1
                            j = atom_cell_list[j]
            neighbor_number[i] = nindex

    def compute(self):
        """Do the real neighbor calculation."""
        atom_cell_list = np.zeros(self.N, dtype=np.int32)
        cell_id_list = np.full(
            (self.ncel[0], self.ncel[1], self.ncel[2]), -1, dtype=np.int32
        )
        need_check = True
        if self.max_neigh is None:
            max_neigh_list = np.zeros_like(cell_id_list)
            # flag 0 indicate double, in the future we may support single!
            if self.rec:
                _build_cell_rec_with_jishu(
                    self.pos,
                    atom_cell_list,
                    cell_id_list,
                    np.ascontiguousarray(self.box[-1]),
                    np.array([i for i in self.ncel]),
                    self.bin_length,
                    max_neigh_list,
                    0,
                )
            else:
                _build_cell_tri_with_jishu(
                    self.pos,
                    atom_cell_list,
                    cell_id_list,
                    self.box,
                    np.array([i for i in self.ncel]),
                    self.bin_length,
                    max_neigh_list,
                    0,
                )
            self.max_neigh = np.partition(max_neigh_list.flatten(), -4)[-4:].sum()
            need_check = False
        else:
            if self.rec:
                _build_cell_rec(
                    self.pos,
                    atom_cell_list,
                    cell_id_list,
                    np.ascontiguousarray(self.box[-1]),
                    np.array([i for i in self.ncel]),
                    self.bin_length,
                    0,
                )

            else:
                _build_cell_tri(
                    self.pos,
                    atom_cell_list,
                    cell_id_list,
                    self.box,
                    np.array([i for i in self.ncel]),
                    self.bin_length,
                    0,
                )

        self.verlet_list = np.full((self.N, self.max_neigh), -1, dtype=np.int32)
        self.distance_list = np.full(
            (self.N, self.max_neigh), self.rc + 1.0, dtype=self.pos.dtype
        )
        self.neighbor_number = np.zeros(self.N, dtype=np.int32)

        self._build_verlet_list(
            self.pos,
            atom_cell_list,
            cell_id_list,
            self.verlet_list,
            self.distance_list,
            self.neighbor_number,
            self.box,
        )

        if need_check:
            max_neigh_number = self.neighbor_number.max()
            assert (
                max_neigh_number <= self.max_neigh
            ), f"Increase the max_neigh larger than {max_neigh_number}."
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

    ti.init()
    start = time()
    lattice_constant = 3.615
    x, y, z = 250, 100, 100
    FCC = LatticeMaker(lattice_constant, "FCC", x, y, z)
    FCC.compute()
    end = time()
    print(f"Build {FCC.pos.shape[0]} atoms FCC time: {end-start} s.")
    for arch, tarch in zip(["cpu", "gpu"], [ti.cpu, ti.gpu]):
        ti.init(
            tarch, offline_cache=True, device_memory_fraction=0.9, default_fp=ti.f64
        )
        start = time()
        neigh = Neighbor(FCC.pos, FCC.box, 5.0, max_neigh=43)
        neigh.compute()
        end = time()
        print(f"Arch: {arch}. Build neighbor time: {end-start} s.")
        print(neigh.verlet_list[0, :5])
        print(neigh.distance_list[0, :5])
        print(neigh.verlet_list.shape)
        # print(neigh.distance_list.dtype)
    # print(neigh.verlet_list.shape[1])
    # print(neigh.neighbor_number.max())
    # print(neigh.neighbor_number.min())

    # neigh.sort_verlet_by_distance(12)
    # print(neigh.verlet_list[0])
    # print(neigh.distance_list[0])
