# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import taichi as ti
import numpy as np


@ti.data_oriented
class AtomicEntropy:
    def __init__(
        self,
        vol,
        verlet_list,
        distance_list,
        rc=5.0,
        sigma=0.2,
        use_local_density=False,
        compute_average=False,
        average_rc=None,
    ):
        self.vol = vol
        self.verlet_list = verlet_list
        self.distance_list = distance_list
        self.rc = rc
        self.sigma = sigma
        self.use_local_density = use_local_density
        self.compute_average = compute_average
        if average_rc is None:
            self.average_rc = rc
        else:
            assert average_rc <= rc, "Average rc should not be larger than rc!"
            self.average_rc = average_rc

    @ti.kernel
    def _compute(
        self,
        distance_list: ti.types.ndarray(),
        rlist: ti.types.ndarray(),
        rlist_sq: ti.types.ndarray(),
        prefactor: ti.types.ndarray(),
        entropy: ti.types.ndarray(),
    ):
        for i in range(self.N):
            g_m = ti.Vector([ti.float64(0.0)] * self.nbins)
            intergrad = ti.Vector([ti.float64(0.0)] * self.nbins)
            n_neigh = 0
            for j in ti.static(range(self.nbins)):
                for k in range(distance_list.shape[1]):
                    if distance_list[i, k] <= self.rc:
                        g_m[j] += (
                            ti.exp(
                                -((rlist[j] - distance_list[i, k]) ** 2)
                                / (2.0 * self.sigma**2)
                            )
                            / prefactor[j]
                        )
                        if j == 0:
                            n_neigh += 1

            density = ti.float64(0.0)
            if self.use_local_density:
                local_vol = 4 / 3 * ti.math.pi * self.rc**3
                density = n_neigh / local_vol
                g_m *= self.global_density / density
            else:
                density = self.global_density

            for j in ti.static(range(self.nbins)):
                if g_m[j] >= 1e-10:
                    intergrad[j] = (g_m[j] * ti.log(g_m[j]) - g_m[j] + 1.0) * rlist_sq[
                        j
                    ]
                else:
                    intergrad[j] = rlist_sq[j]

            sum_intergrad = ti.float64(0.0)
            for j in ti.static(range(self.nbins - 1)):
                sum_intergrad += (intergrad[j] + intergrad[j + 1]) * (
                    rlist[j + 1] - rlist[j]
                )

            entropy[i] = -ti.math.pi * density * sum_intergrad

    @ti.kernel
    def _compute_average(
        self,
        entropy: ti.types.ndarray(),
        verlet_list: ti.types.ndarray(),
        distance_list: ti.types.ndarray(),
        entropy_average: ti.types.ndarray(),
    ):

        for i in range(verlet_list.shape[0]):
            neigh_num = 1
            entropy_average[i] = entropy[i]
            for j in range(verlet_list.shape[1]):
                if verlet_list[i, j] > -1:
                    if distance_list[i, j] <= self.average_rc:
                        neigh_num += 1
                        entropy_average[i] += entropy[verlet_list[i, j]]
                else:
                    break
            entropy_average[i] /= neigh_num

    def compute(self):

        self.N = self.distance_list.shape[0]
        self.entropy = np.zeros(self.N)
        self.global_density = self.N / self.vol
        self.nbins = int(np.floor(self.rc / self.sigma) + 1)
        rlist = np.linspace(0.0, self.rc, self.nbins)
        rlist_sq = rlist**2
        prefactor = rlist_sq * (
            4 * np.pi * self.global_density * np.sqrt(2 * np.pi * self.sigma**2)
        )
        prefactor[0] = prefactor[1]
        self._compute(self.distance_list, rlist, rlist_sq, prefactor, self.entropy)
        if self.compute_average:
            self.entropy_average = np.zeros(self.N)
            self._compute_average(
                self.entropy, self.verlet_list, self.distance_list, self.entropy_average
            )


if __name__ == "__main__":
    from lattice_maker import LatticeMaker
    from neighbor import Neighbor
    from time import time

    # ti.init(ti.gpu, device_memory_GB=5.0)
    ti.init(ti.cpu)
    start = time()
    lattice_constant = 4.05
    x, y, z = 100, 100, 50
    FCC = LatticeMaker(lattice_constant, "FCC", x, y, z)
    FCC.compute()
    end = time()
    print(f"Build {FCC.pos.shape[0]} atoms FCC time: {end-start} s.")
    start = time()
    neigh = Neighbor(FCC.pos, FCC.box, 5.0, max_neigh=60)
    neigh.compute()
    end = time()
    print(f"Build neighbor time: {end-start} s.")

    start = time()
    vol = np.product(FCC.box[:, 1] - FCC.box[:, 0])
    Entropy = AtomicEntropy(
        vol,
        neigh.verlet_list,
        neigh.distance_list,
        neigh.rc,
        sigma=0.2,
        use_local_density=False,
        compute_average=True,
        average_rc=4.0,
    )
    Entropy.compute()
    entropy = Entropy.entropy
    entropy_ave = Entropy.entropy_average
    end = time()
    print(f"Cal entropy time: {end-start} s.")
    print(entropy)
    print(entropy_ave)
