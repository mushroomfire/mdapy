# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import numpy as np
from scipy.spatial import KDTree


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

        self.shift_pos = pos - np.min(pos, axis=0)
        self.box = box
        self.boundary = boundary
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
        self.kdt = KDTree(self.shift_pos, boxsize=boxsize)

    def query_nearest_neighbors(self, n, workers=-1):
        """Query the :math:`n` nearest atom neighbors.

        Args:
            n (int): number of neighbors to query.

            workers (int, optional): maximum CPU cores to use in calculation. Defaults to -1, indicating using all available CPU cores.

        Returns:
            tuple: (distance, index), distance of atom :math:`i` to its neighbor atom :math:`j`, and the index of atom :math:`j`.
        """
        dis, index = self.kdt.query(self.shift_pos, k=n + 1, workers=workers)

        return np.ascontiguousarray(dis[:, 1:]), np.ascontiguousarray(index[:, 1:])


if __name__ == "__main__":
    from lattice_maker import LatticeMaker
    import taichi as ti
    from time import time

    # ti.init(ti.gpu, device_memory_GB=5.0)
    ti.init(ti.cpu)
    start = time()
    lattice_constant = 4.05
    x, y, z = 10, 10, 10
    FCC = LatticeMaker(lattice_constant, "FCC", x, y, z)
    FCC.compute()
    end = time()
    print(f"Build {FCC.pos.shape[0]} atoms FCC time: {end-start} s.")
    np.random.seed(10)
    noise = np.random.rand(*FCC.pos.shape)
    FCC.pos += noise / 10
    start = time()
    kdt = kdtree(FCC.pos, FCC.box, [1, 1, 1])
    end = time()
    print(f"Build kdtree time: {end-start} s.")

    start = time()
    dis, index = kdt.query_nearest_neighbors(12)
    end = time()
    print(f"Query kdtree time: {end-start} s.")

    print(dis[0])
    print(index[0])

    # FCC.write_data()
