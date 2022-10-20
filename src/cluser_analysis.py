import numpy as np
import numba as nb


@nb.njit
def _compute(N, verlet_list, particleClusters):
    cluster = 0

    for seedParticleIndex in range(N):
        if particleClusters[seedParticleIndex] != -1:
            continue
        toProcess = []
        toProcess.append(seedParticleIndex)
        cluster += 1
        while len(toProcess) > 0:
            currentParticle = toProcess[0]
            del toProcess[0]
            for j in range(verlet_list.shape[1]):
                neighborIndex = verlet_list[currentParticle, j]
                if neighborIndex > -1:
                    if particleClusters[neighborIndex] == -1:
                        particleClusters[neighborIndex] = cluster
                        toProcess.append(neighborIndex)
    return cluster


class ClusterAnalysis:
    def __init__(self, N, verlet_list):
        self.N = N
        self.verlet_list = verlet_list
        self.particleClusters = np.zeros(self.N, dtype=np.int32) - 1
        self.is_computed = False

    def compute(self):
        self.cluster_number = _compute(self.N, self.verlet_list, self.particleClusters)
        print(f"Cluster number is {self.cluster_number}.")
        self.is_computed = True

    def get_size_of_cluster(self, cluster_id):
        if not self.is_computed:
            self.compute()
        assert (
            0 < cluster_id <= self.cluster_number
        ), f"cluster_id should be in the range of  [1, cluster_number {self.cluster_number}]."
        return len(self.particleClusters[self.particleClusters == cluster_id])
