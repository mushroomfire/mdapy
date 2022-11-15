import numpy as np
from .cluster import _cluster_analysis


class ClusterAnalysis:
    """
    代码参考Ovito的cluster_analysis模块,用于将粒子分类到不同的相互连通的group，也就是一个cluster中
    """

    def __init__(self, rc, verlet_list, distance_list):
        self.rc = rc
        self.verlet_list = verlet_list
        self.distance_list = distance_list
        self.is_computed = False

    def compute(self):
        N = self.verlet_list.shape[0]
        self.particleClusters = np.zeros(N, dtype=np.int32) - 1
        self.cluster_number = _cluster_analysis.get_cluster(
            self.verlet_list, self.distance_list, self.rc, self.particleClusters
        )
        # print(f"Cluster number is {self.cluster_number}.")
        self.is_computed = True

    def get_size_of_cluster(self, cluster_id):
        if not self.is_computed:
            self.compute()
        assert (
            0 < cluster_id <= self.cluster_number
        ), f"cluster_id should be in the range of  [1, cluster_number {self.cluster_number}]."
        return len(self.particleClusters[self.particleClusters == cluster_id])
