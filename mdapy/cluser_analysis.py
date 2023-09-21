# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import numpy as np

try:
    from tool_function import _check_repeat_cutoff
    from replicate import Replicate
    from neighbor import Neighbor
    from cluster import _cluster_analysis
except Exception:
    from .tool_function import _check_repeat_cutoff
    from .replicate import Replicate
    from .neighbor import Neighbor
    import _cluster_analysis


class ClusterAnalysis:
    """This class is used to divide atoms connected within a given cutoff distance into a cluster.
    It is helpful to recognize the reaction products or fragments under shock loading.

    Args:
        rc (float): cutoff distance.
        verlet_list (np.ndarray, optional): (:math:`N_p, max\_neigh`) verlet_list[i, j] means j atom is a neighbor of i atom if j > -1. Defaults to None.
        distance_list (np.ndarray, optional): (:math:`N_p, max\_neigh`) distance_list[i, j] means distance between i and j atom. Defaults to None.
        pos (np.ndarray, optional): (:math:`N_p, 3`) particles positions. Defaults to None.
        box (np.ndarray, optional): (:math:`3, 2`) or (:math:`4, 3`) system box. Defaults to None.
        boundary (list, optional): boundary conditions, 1 is periodic and 0 is free boundary. Such as [1, 1, 1]. Defaults to None.

    Outputs:
        - **particleClusters** (np.ndarray) - (:math:`N_p`) cluster ID per atoms.
        - **cluster_number** (int) - cluster number.

    .. note:: This class use the `same method as in Ovito <https://www.ovito.org/docs/current/reference/pipelines/modifiers/cluster_analysis.html#particles-modifiers-cluster-analysis>`_.

    Examples:
        >>> import mdapy as mp

        >>> mp.init()

        >>> FCC = mp.LatticeMaker(3.615, 'FCC', 10, 10, 10) # Create a FCC structure

        >>> FCC.compute() # Get atom positions

        >>> FCC.pos = FCC.pos[((FCC.pos[:, 0]>1*3.615) & (FCC.pos[:, 0]<4*3.615))
                            | ((FCC.pos[:,0]>6*3.615) & (FCC.pos[:,0]< 9*3.615))] # Do a slice operation.

        >>> neigh = mp.Neighbor(FCC.pos, FCC.box,
                                4., max_neigh=50) # Initialize Neighbor class.

        >>> neigh.compute() # Calculate particle neighbor information.

        >>> Clus = mp.ClusterAnalysis(4., neigh.verlet_list, neigh.distance_list) # Initilize Cluster class.

        >>> Clus.compute() # Do cluster calculation.

        >>> Clus.cluster_number # Check cluster number, should be 2 here.

        >>> Clus.particleClusters # Check atom in which cluster.

        >>> Clus.get_size_of_cluster(1) # Obtain the atom number in cluster 1.

    """

    def __init__(
        self,
        rc,
        verlet_list=None,
        distance_list=None,
        pos=None,
        box=None,
        boundary=None,
    ):
        self.rc = rc
        self.old_N = None

        self.verlet_list = verlet_list
        self.distance_list = distance_list
        if verlet_list is None or distance_list is None:
            assert pos is not None
            assert box is not None
            assert boundary is not None
            repeat = _check_repeat_cutoff(box, boundary, self.rc)

            if pos.dtype != np.float64:
                pos = pos.astype(np.float64)
            if box.dtype != np.float64:
                box = box.astype(np.float64)
            if sum(repeat) == 3:
                self.pos = pos
                if box.shape == (3, 2):
                    self.box = np.zeros((4, 3), dtype=box.dtype)
                    self.box[0, 0], self.box[1, 1], self.box[2, 2] = (
                        box[:, 1] - box[:, 0]
                    )
                    self.box[-1] = box[:, 0]
                elif box.shape == (4, 3):
                    self.box = box
            else:
                self.old_N = pos.shape[0]
                repli = Replicate(pos, box, *repeat)
                repli.compute()
                self.pos = repli.pos
                self.box = repli.box

            assert self.box[0, 1] == 0
            assert self.box[0, 2] == 0
            assert self.box[1, 2] == 0
            self.boundary = [int(boundary[i]) for i in range(3)]
        self.is_computed = False

    def compute(self):
        """Do the real cluster analysis."""
        if self.verlet_list is None or self.distance_list is None:
            neigh = Neighbor(self.pos, self.box, self.rc, self.boundary)
            neigh.compute()
            self.verlet_list, self.distance_list = (
                neigh.verlet_list,
                neigh.distance_list,
            )
        N = self.verlet_list.shape[0]
        self.particleClusters = np.zeros(N, dtype=np.int32) - 1
        self.cluster_number = _cluster_analysis._get_cluster(
            self.verlet_list, self.distance_list, self.rc, self.particleClusters
        )
        # print(f"Cluster number is {self.cluster_number}.")
        if self.old_N is not None:
            self.particleClusters = np.ascontiguousarray(
                self.particleClusters[: self.old_N]
            )
        self.is_computed = True

    def get_size_of_cluster(self, cluster_id):
        """This method can obtain the number of atoms in cluster :math:`cluster\_id`.

        Args:
            cluster_id (int): cluster ID, it should be larger than 0 and lower than total :math:`cluster\_number`.

        Returns:
            int: the number of atoms in given cluster.
        """
        if not self.is_computed:
            self.compute()
        assert (
            0 < cluster_id <= self.cluster_number
        ), f"cluster_id should be in the range of  [1, cluster_number {self.cluster_number}]."
        return len(self.particleClusters[self.particleClusters == cluster_id])


if __name__ == "__main__":
    from lattice_maker import LatticeMaker
    from time import time
    from neighbor import Neighbor
    import taichi as ti

    # ti.init(ti.gpu, device_memory_GB=5.0)
    ti.init(ti.cpu)
    start = time()
    lattice_constant = 4.05
    x, y, z = 10, 10, 10
    FCC = LatticeMaker(lattice_constant, "FCC", x, y, z)
    FCC.compute()
    end = time()
    print(f"Build {FCC.pos.shape[0]} atoms FCC time: {end-start} s.")

    # start = time()
    # neigh = Neighbor(FCC.pos, FCC.box, 3.0, max_neigh=20)
    # neigh.compute()
    # # print(neigh.neighbor_number.max())
    # end = time()
    # print(f"Build neighbor time: {end-start} s.")

    start = time()
    Cls = ClusterAnalysis(3.0, pos=FCC.pos, box=FCC.box, boundary=[1, 1, 1])
    Cls.compute()
    end = time()
    print(f"Cal cluster time: {end-start} s.")

    print("Cluster id:", Cls.particleClusters)

    print("Number of cluster", Cls.cluster_number)

    print("Cluster size of 1:", Cls.get_size_of_cluster(1))
