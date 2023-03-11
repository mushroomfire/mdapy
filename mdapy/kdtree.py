# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import numpy as np
from scipy.spatial import KDTree
import taichi as ti


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
    One can install `pyfnntw <https://github.com/cavemanloverboy/FNNTW>`_ to accelerate this module.
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
            self.shift_pos = np.ascontiguousarray(pos - lower)  # for pyfnntw backend.
        self.box = box
        self.boundary = boundary
        self._use_pyfnntw = False
        self._init()

    def _init(self):
        boxsize = np.array(
            [
                self.box[i][1] - self.box[i][0]
                if self.boundary[i] == 1
                else self.box[i][1] - self.box[i][0] + 50.0
                for i in range(3)
            ]
        )
        try:
            import pyfnntw

            if self.shift_pos.dtype == np.float64:
                if self.shift_pos.shape[0] < 10**6:
                    self.kdt = pyfnntw.Treef64(
                        self.shift_pos,
                        leafsize=32,
                        par_split_level=1,
                        boxsize=boxsize,
                    )
                elif self.shift_pos.shape[0] < 10**7:
                    self.kdt = pyfnntw.Treef64(
                        self.shift_pos,
                        leafsize=32,
                        par_split_level=2,
                        boxsize=boxsize,
                    )
                else:
                    self.kdt = pyfnntw.Treef64(
                        self.shift_pos,
                        leafsize=32,
                        par_split_level=4,
                        boxsize=boxsize,
                    )
            elif self.shift_pos.dtype == np.float32:
                if self.shift_pos.shape[0] < 10**6:
                    self.kdt = pyfnntw.Treef32(
                        self.shift_pos,
                        leafsize=32,
                        par_split_level=1,
                        boxsize=boxsize,
                    )
                elif self.shift_pos.shape[0] < 10**7:
                    self.kdt = pyfnntw.Treef32(
                        self.shift_pos,
                        leafsize=32,
                        par_split_level=2,
                        boxsize=boxsize,
                    )
                else:
                    self.kdt = pyfnntw.Treef32(
                        self.shift_pos,
                        leafsize=32,
                        par_split_level=4,
                        boxsize=boxsize,
                    )
            self._use_pyfnntw = True
        except Exception:
            self.kdt = KDTree(self.shift_pos, leafsize=32, boxsize=boxsize)

    def query_nearest_neighbors(self, n, workers=-1):
        """Query the :math:`n` nearest atom neighbors.

        Args:
            n (int): number of neighbors to query.
            worker (int): maximum cores used. Defaults to -1, indicating use all aviliable cores. Only works for scipy backend.

        Returns:
            tuple: (distance, index), distance of atom :math:`i` to its neighbor atom :math:`j`, and the index of atom :math:`j`.
        """
        if self._use_pyfnntw:
            dis, index = self.kdt.query(self.shift_pos, n + 1)
        else:
            dis, index = self.kdt.query(self.shift_pos, k=n + 1, workers=workers)
        return np.ascontiguousarray(dis[:, 1:]), np.ascontiguousarray(index[:, 1:])


if __name__ == "__main__":
    from lattice_maker import LatticeMaker
    import taichi as ti
    from time import time

    ti.init(ti.cpu)
    start = time()
    lattice_constant = 4.05
    x, y, z = 250, 100, 100
    FCC = LatticeMaker(lattice_constant, "FCC", x, y, z)
    FCC.compute()
    end = time()
    print(f"Build {FCC.pos.shape[0]} atoms FCC time: {end-start} s.")
    FCC.pos -= 0.2
    print(np.min(FCC.pos, axis=0))
    start = time()
    kdt = kdtree(FCC.pos, FCC.box, [1, 1, 1])
    end = time()
    print(f"Build kdtree time: {end-start} s.")
    print(np.min(FCC.pos, axis=0))
    start = time()
    dis, index = kdt.query_nearest_neighbors(12)
    end = time()
    print(f"Query kdtree time: {end-start} s.")

    print(dis[0])
    print(index[0])
