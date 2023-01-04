# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import numpy as np
from .cluster import _cluster_analysis


class ClusterAnalysis:
    """This class is used to divide atoms connected within a given cutoff distance into a cluster.
    It is helpful to recognize the reaction products or fragments under shock loading.

    Args:
        rc (float): cutoff distance.
        verlet_list (np.ndarray): (:math:`N_p, max\_neigh`) verlet_list[i, j] means j atom is a neighbor of i atom if j > -1.
        distance_list (np.ndarray): (:math:`N_p, max\_neigh`) distance_list[i, j] means distance between i and j atom.

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

    def __init__(self, rc, verlet_list, distance_list):

        self.rc = rc
        self.verlet_list = verlet_list
        self.distance_list = distance_list
        self.is_computed = False

    def compute(self):
        """Do the real cluster analysis."""
        N = self.verlet_list.shape[0]
        self.particleClusters = np.zeros(N, dtype=np.int32) - 1
        self.cluster_number = _cluster_analysis.get_cluster(
            self.verlet_list, self.distance_list, self.rc, self.particleClusters
        )
        # print(f"Cluster number is {self.cluster_number}.")
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
