# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import numpy as np
import taichi as ti

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


@ti.kernel
def filter_by_type(
    verlet_list: ti.types.ndarray(),
    distance_list: ti.types.ndarray(),
    neighbor_number: ti.types.ndarray(),
    type_list: ti.types.ndarray(),
    type1: ti.types.ndarray(),
    type2: ti.types.ndarray(),
    r: ti.types.ndarray(),
):
    for i in range(verlet_list.shape[0]):
        n_neighbor = neighbor_number[i]
        for jj in range(n_neighbor):
            j = verlet_list[i, jj]
            for k in range(type1.shape[0]):
                if (
                    type1[k] == type_list[i]
                    and type2[k] == type_list[j]
                    and distance_list[i, jj] > r[k]
                ):
                    verlet_list[i, jj] = -1


class ClusterAnalysis:
    """This class is used to divide atoms connected within a given cutoff distance into a cluster.
    It is helpful to recognize the reaction products or fragments under shock loading.

    Args:
        rc (float | dict): cutoff distance. One can also assign multi cutoff for different elemental pair, such as {'1-1':1.5, '1-6':1.7}. The unassigned elemental pair will default use the maximum cutoff distance.
        verlet_list (np.ndarray, optional): (:math:`N_p, max\_neigh`) verlet_list[i, j] means j atom is a neighbor of i atom if j > -1. Defaults to None.
        distance_list (np.ndarray, optional): (:math:`N_p, max\_neigh`) distance_list[i, j] means distance between i and j atom. Defaults to None.
        neighbor_number (np.ndarray, optional): (:math:`N_p`) neighbor number per atoms. Defaults to None.
        pos (np.ndarray, optional): (:math:`N_p, 3`) particles positions. Defaults to None.
        box (np.ndarray, optional): (:math:`3, 2`) or (:math:`4, 3`) system box. Defaults to None.
        boundary (list, optional): boundary conditions, 1 is periodic and 0 is free boundary. Such as [1, 1, 1]. Defaults to None.
        type_list (np.ndarray, optional): (:math:`N_p`) atom type. It is needed only if rc is a Dict.

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

        >>> Clus = mp.ClusterAnalysis(4., neigh.verlet_list, neigh.distance_list, neigh.neighbor_number) # Initilize Cluster class.

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
        neighbor_number=None,
        pos=None,
        box=None,
        boundary=None,
        type_list=None,
    ):
        self.rc = rc
        if isinstance(rc, float) or isinstance(rc, int):
            self.max_rc = self.rc
        elif isinstance(rc, dict):
            assert type_list is not None, "Need type_list for multi cutoff mode."
            self.max_rc = max([i for i in self.rc.values()])
        else:
            raise "rc should be a positive number, or a dict like {'1-1':1.5, '1.2':1.3}"
        self.old_N = None
        self.verlet_list = verlet_list
        self.distance_list = distance_list
        self.neighbor_number = neighbor_number
        if verlet_list is None or distance_list is None or neighbor_number is None:
            assert pos is not None
            assert box is not None
            assert boundary is not None
            repeat = _check_repeat_cutoff(box, boundary, self.max_rc)

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
        self.type_list = type_list
        self.is_computed = False

    def _filter_verlet(self, verlet_list, distance_list, neighbor_number):
        type1, type2, r = [], [], []
        for key, value in self.rc.items():
            left, right = key.split("-")
            type1.append(left)
            type2.append(right)
            r.append(value)
            if left != right:
                type1.append(right)
                type2.append(left)
                r.append(value)
        type1 = np.array(type1, int)
        type2 = np.array(type2, int)
        r = np.array(r, float)
        filter_by_type(
            verlet_list, distance_list, neighbor_number, self.type_list, type1, type2, r
        )

    def compute(self):
        """Do the real cluster analysis."""
        distance_list, neighbor_number = self.distance_list, self.neighbor_number
        if self.verlet_list is not None and isinstance(self.rc, dict):
            verlet_list = self.verlet_list.copy()
        else:
            verlet_list = self.verlet_list
        if (
            self.verlet_list is None
            or self.distance_list is None
            or self.neighbor_number is None
        ):
            neigh = Neighbor(self.pos, self.box, self.max_rc, self.boundary)
            neigh.compute()
            verlet_list, distance_list, neighbor_number = (
                neigh.verlet_list,
                neigh.distance_list,
                neigh.neighbor_number,
            )

        if isinstance(self.rc, dict):
            self._filter_verlet(verlet_list, distance_list, neighbor_number)

        N = verlet_list.shape[0]
        self.particleClusters = np.full(N, -1, dtype=np.int32)
        if isinstance(self.rc, dict):
            self.cluster_number = _cluster_analysis._get_cluster_by_bond(
                verlet_list, neighbor_number, self.particleClusters
            )
        else:
            self.cluster_number = _cluster_analysis._get_cluster(
                verlet_list,
                distance_list,
                neighbor_number,
                self.max_rc,
                self.particleClusters,
            )

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
    from time import time
    from neighbor import Neighbor
    import taichi as ti
    import mdapy as mp

    ti.init(ti.cpu)
    system = mp.System(r"example\CoCuFeNiPd-4M.dump")

    start = time()
    # {'1-1':2.5, '1-2':1.6, '2-2':1.5}
    Cls = ClusterAnalysis(
        rc=2.9,
        pos=system.pos,
        box=system.box,
        boundary=system.boundary,
        type_list=system.data["type"].to_numpy(),
    )
    Cls.compute()
    end = time()
    print(f"Cal cluster time: {end-start} s.")
    print("Cluster id:", Cls.particleClusters)
    print("Number of cluster", Cls.cluster_number)

    print("Cluster size of 1:", Cls.get_size_of_cluster(1))
