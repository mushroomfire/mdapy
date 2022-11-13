import numpy as np


try:
    from .cluster import ClusterAnalysisComputeF

    _compute = ClusterAnalysisComputeF.compute
except ImportError:
    import numba as nb

    @nb.njit
    def _compute(verlet_list, distance_list, rc):
        cluster = 0
        N = verlet_list.shape[0]
        particleClusters = np.zeros(N, dtype=np.int32) - 1
        for seedParticleIndex in range(N):
            if particleClusters[seedParticleIndex] != -1:
                continue
            toProcess = []
            toProcess.append(seedParticleIndex)
            cluster += 1
            while True:  # 保证程序至少执行一次
                if len(toProcess) > 0:
                    currentParticle = toProcess[0]
                    del toProcess[0]
                    n = 0  # 领域计数,保证孤立原子也有cluster_id
                    for j in range(verlet_list.shape[1]):
                        neighborIndex = verlet_list[currentParticle, j]
                        if neighborIndex > -1:
                            if distance_list[currentParticle, j] <= rc:
                                n += 1
                                if particleClusters[neighborIndex] == -1:
                                    particleClusters[neighborIndex] = cluster
                                    toProcess.append(neighborIndex)
                        else:
                            break
                    if n == 0:
                        particleClusters[currentParticle] = cluster
                else:
                    break
        return particleClusters, cluster


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
        self.particleClusters, self.cluster_number = _compute(
            self.verlet_list, self.distance_list, self.rc
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
