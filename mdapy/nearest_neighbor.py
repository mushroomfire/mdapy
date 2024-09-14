# Copyright (c) 2022-2024, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import numpy as np
from scipy.spatial import KDTree
import taichi as ti

try:
    from box import init_box, _pbc
    from neigh._neigh import build_cell_tri
    from tool_function import _check_repeat_nearest
except Exception:
    from .box import init_box, _pbc
    from _neigh import build_cell_tri
    from .tool_function import _check_repeat_nearest


@ti.kernel
def _wrap_pos(
    pos: ti.types.ndarray(), box: ti.types.ndarray(), boundary: ti.types.ndarray()
):
    """This function is used to wrap particle positions into box considering periodic boundarys.

    Args:
        pos (ti.types.ndarray): (Nx3) particle position.

        box (ti.types.ndarray): (3x2) system box.

        boundary (ti.types.ndarray): boundary conditions, 1 is periodic and 0 is free boundary.
    """
    boxlength = ti.Vector([box[j, 1] - box[j, 0] for j in range(3)])
    for i in range(pos.shape[0]):
        for j in ti.static(range(3)):
            if boundary[j] == 1:
                while pos[i, j] < box[j, 0]:
                    pos[i, j] += boxlength[j]
                while pos[i, j] >= box[j, 1]:
                    pos[i, j] -= boxlength[j]


class kdtree:
    """This class is a wrapper of `kdtree of scipy <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html>`_
    and helful to obtain the certain nearest atom neighbors considering the periodic/free boundary.
    If you want to access the atom neighbor within a spherical
    distance, the Neighbor class is suggested.

    Args:
        pos (np.ndarray): (:math:`N_p, 3`), particles positions.
        box (np.ndarray): (:math:`3, 2`), system box.
        boundary (list): boundary conditions, 1 is periodic and 0 is free boundary. Such as [1, 1, 1].

    Outputs:
        - **shift_pos** (np.ndarray) - (:math:`N_p, 3`), shifted position, making the lower cornor is zero.
        - **kdt** (scipy.spatial.KDTree) - a KDTree class.

    Examples:

        >>> import mdapy as mp

        >>> mp.init()

        >>> FCC = mp.LatticeMaker(3.615, 'FCC', 10, 10, 10) # Create a FCC structure

        >>> FCC.compute() # Get atom positions

        >>> kdt = mp.kdtree(FCC.pos, FCC.box, [1, 1, 1]) # Build a kdtree.

        >>> dis, index = kdt.query_nearest_neighbors(12) # Query the 12 nearest neighbors per atom.
    """

    def __init__(self, pos, box, boundary):
        if_wrap = False
        lower, upper = np.min(pos, axis=0), np.max(pos, axis=0)

        for i in range(3):
            if lower[i] < box[i, 0] or upper[i] > box[i, 1]:
                if_wrap = True
                break
        if if_wrap:
            new_pos = pos.copy()
            _wrap_pos(new_pos, box, np.array(boundary, int))
            self.shift_pos = new_pos - np.min(new_pos, axis=0)
        else:
            self.shift_pos = pos - lower
        self.box = box
        self.boundary = boundary
        self._init()

    def _init(self):
        boxsize = np.array(
            [
                (
                    self.box[i][1] - self.box[i][0]
                    if self.boundary[i] == 1
                    else self.box[i][1] - self.box[i][0] + 50.0
                )
                for i in range(3)
            ]
        )

        self.kdt = KDTree(self.shift_pos, boxsize=boxsize)

    def query_nearest_neighbors(self, K, workers=-1):
        """Query the :math:`n` nearest atom neighbors.

        Args:
            K (int): number of neighbors to query.
            worker (int): maximum cores used. Defaults to -1, indicating use all aviliable cores. Only works for scipy backend.

        Returns:
            tuple: (distance, index), distance of atom :math:`i` to its neighbor atom :math:`j`, and the index of atom :math:`j`.
        """

        dis, index = self.kdt.query(self.shift_pos, k=K + 1, workers=workers)

        return np.ascontiguousarray(dis[:, 1:]), np.ascontiguousarray(index[:, 1:])


@ti.data_oriented
class NearestNeighbor:
    """This class is used to query the nearest neighbor with fixed number. For rectangle box, this
    class is a wrapper of `kdtree of scipy <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html>`_
    and helful to obtain the certain nearest atom neighbors considering the periodic/free boundary.
    If you want to access the atom neighbor within a spherical
    distance, the Neighbor class is suggested.
    For triclinic box, this class use cell-list to find the nearest neighbors.

    Args:
        pos (np.ndarray): (:math:`N_p, 3`), particles positions.
        box (np.ndarray): (:math:`4, 3` or :math:`3, 2`), system box.
        boundary (list): boundary conditions, 1 is periodic and 0 is free boundary. Defaults to [1, 1, 1].

    Examples:

        >>> import mdapy as mp

        >>> mp.init()

        >>> FCC = mp.LatticeMaker(3.615, 'FCC', 10, 10, 10) # Create a FCC structure

        >>> FCC.compute() # Get atom positions

        >>> kdt = mp.kdtree(FCC.pos, FCC.box, [1, 1, 1]) # Build a kdtree.

        >>> dis, index = kdt.query_nearest_neighbors(12) # Query the 12 nearest neighbors per atom.
    """

    def __init__(self, pos, box, boundary=[1, 1, 1]):
        self.box, self.inverse_box, self.rec = init_box(box)
        repeat = _check_repeat_nearest(pos, self.box, boundary)
        assert (
            sum(repeat) == 3
        ), f"The atom number < 100 or shorest box length < 1 nm, which should be repeated by {repeat} to make sure the results correct."

        if pos.dtype != np.float64:
            pos = pos.astype(np.float64)
        self.pos = pos
        self.boundary = ti.Vector([int(boundary[i]) for i in range(3)])
        self.N = self.pos.shape[0]
        if self.rec:
            box = np.zeros((3, 2))
            box[:, 0] = self.box[-1]
            box[:, 1] = (
                np.array([self.box[0, 0], self.box[1, 1], self.box[2, 2]]) + box[:, 0]
            )
            self._kdt = kdtree(self.pos, box, self.boundary)

    @ti.kernel
    def _build_verlet_list(
        self,
        pos: ti.types.ndarray(dtype=ti.math.vec3),
        atom_cell_list: ti.types.ndarray(),
        cell_id_list: ti.types.ndarray(),
        init_delta: int,
        verlet_list: ti.types.ndarray(),
        distance_list: ti.types.ndarray(),
        box: ti.types.ndarray(element_dim=1),
        inverse_box: ti.types.ndarray(element_dim=1),
        K: int,
    ):
        for i in range(self.N):
            r = pos[i] - box[3]
            n = r[0] * inverse_box[0] + r[1] * inverse_box[1] + r[2] * inverse_box[2]
            icel = ti.floor((n[0] * box[0]).norm() / self.bin_length, int)
            jcel = ti.floor((n[1] * box[1]).norm() / self.bin_length, int)
            kcel = ti.floor((n[2] * box[2]).norm() / self.bin_length, int)
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
            nindex = 0
            delta = init_delta
            while nindex < K:
                for iicel in range(icel - delta, icel + delta + 1):
                    for jjcel in range(jcel - delta, jcel + delta + 1):
                        for kkcel in range(kcel - delta, kcel + delta + 1):
                            j = cell_id_list[
                                iicel % self.ncel[0],
                                jjcel % self.ncel[1],
                                kkcel % self.ncel[2],
                            ]
                            while j > -1:
                                rij = _pbc(
                                    pos[j] - pos[i], self.boundary, box, inverse_box
                                )
                                rijdis = rij.norm()
                                if rijdis < delta * self.bin_length and j != i:
                                    if nindex < K:
                                        verlet_list[i, nindex] = j
                                        distance_list[i, nindex] = rijdis
                                    else:
                                        max_value = distance_list[i, 0]
                                        max_index = 0
                                        for m in range(1, K):
                                            if distance_list[i, m] > max_value:
                                                max_value = distance_list[i, m]
                                                max_index = m
                                        if distance_list[i, max_index] > rijdis:
                                            distance_list[i, max_index] = rijdis
                                            verlet_list[i, max_index] = j
                                    nindex += 1
                                j = atom_cell_list[j]

                if nindex < K:
                    nindex = 0
                    delta += 1

            for j in range(K):
                minIndex = j
                for k in range(j + 1, K):
                    if distance_list[i, k] < distance_list[i, minIndex]:
                        minIndex = k
                if minIndex != j:
                    distance_list[i, minIndex], distance_list[i, j] = (
                        distance_list[i, j],
                        distance_list[i, minIndex],
                    )
                    verlet_list[i, minIndex], verlet_list[i, j] = (
                        verlet_list[i, j],
                        verlet_list[i, minIndex],
                    )

    def query_nearest_neighbors(self, K, workers=-1):
        """Query the :math:`n` nearest atom neighbors.

        Args:
            K (int): number of neighbors to query.
            worker (int): maximum cores used. Defaults to -1, indicating use all aviliable cores. Only works for scipy backend.

        Returns:
            tuple: (distance, index), distance of atom :math:`i` to its neighbor atom :math:`j`, and the index of atom :math:`j`.
        """
        assert K < 25
        if sum(self.boundary) == 0:
            assert self.pos.shape[0] >= K, f"Atom number should be larger than {K}."

        if self.rec:
            distance_list, verlet_list = self._kdt.query_nearest_neighbors(
                K, workers=workers
            )
        else:
            self.bin_length = 2.0
            vol = np.inner(self.box[0], np.cross(self.box[1], self.box[2]))
            pho = self.N / vol
            init_delta = int(
                ((K / pho / (4 / 3 * ti.math.pi)) ** (1 / 3) / self.bin_length)
            )
            self.ncel = ti.Vector(
                [
                    int(np.ceil(np.linalg.norm(self.box[i]) / self.bin_length))
                    for i in range(3)
                ]
            )
            atom_cell_list = np.zeros(self.N, dtype=np.int32)
            cell_id_list = np.full(
                (self.ncel[0], self.ncel[1], self.ncel[2]), -1, dtype=np.int32
            )
            build_cell_tri(
                self.pos,
                atom_cell_list,
                cell_id_list,
                self.box,
                self.inverse_box,
                np.array([i for i in self.ncel]),
                self.bin_length,
            )
            verlet_list = np.zeros((self.N, K), int)
            distance_list = np.zeros_like(verlet_list, float)
            self._build_verlet_list(
                self.pos,
                atom_cell_list,
                cell_id_list,
                init_delta,
                verlet_list,
                distance_list,
                self.box,
                self.inverse_box,
                K,
            )
        return distance_list, verlet_list


if __name__ == "__main__":
    from lattice_maker import LatticeMaker
    import taichi as ti
    from time import time

    import mdapy as mp

    ti.init(ti.cpu)
    system = mp.System(r"D:\Package\MyPackage\mdapy-tutorial\frame\Ti.data")
    system.replicate(20, 20, 20)
    # print(system)
    start = time()
    kdt = NearestNeighbor(system.pos, system.box, system.boundary)
    end = time()
    print(f"Build kdtree time: {end-start} s.")
    start = time()
    dis, index = kdt.query_nearest_neighbors(16)
    end = time()
    print(f"Query time: {end-start} s.")
    print(dis[0])
    print(index[0])
    # start = time()
    # lattice_constant = 4.05
    # x, y, z = 100, 100, 100
    # FCC = LatticeMaker(lattice_constant, "FCC", x, y, z)
    # FCC.compute()
    # end = time()
    # print(f"Build {FCC.pos.shape[0]} atoms FCC time: {end-start} s.")

    # start = time()
    # kdt = NearestNeighbor(FCC.pos, FCC.box, [1, 1, 1])
    # # kdt = kdtree(FCC.pos, FCC.box, [1, 1, 1])
    # end = time()
    # print(f"Build kdtree time: {end-start} s.")
    # start = time()
    # dis, index = kdt.query_nearest_neighbors(12)
    # end = time()
    # print(f"Query kdtree time: {end-start} s.")
    # # start = time()
    # # dis, index = kdt.query_nearest_neighbors(16)
    # # end = time()
    # # print(f"Query kdtree time: {end-start} s.")

    # print(dis[0])
    # print(index[0])
    # # print(FCC.box)
    # print(FCC.pos[399] - FCC.pos[0])
