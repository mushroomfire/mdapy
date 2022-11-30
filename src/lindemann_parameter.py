import taichi as ti
import numpy as np
from mdapy.plot.pltset import pltset, cm2inch
import matplotlib.pyplot as plt


@ti.data_oriented
class LindemannParameter:
    """
    Using Welford to updated the varience and mean of rij, see https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford
    Input:
    pos_list : np.ndarray (Nframes, Natoms, 3)

    Output:
    lindemann_atom : np.ndarray (Nframes, Natoms)
    lindemann_frame : np.ndarray (Nframes)
    lindemann_trj : float
    """

    def __init__(self, pos_list) -> None:
        self.pos_list = pos_list
        # self.parallel = parallel
        self.if_compute = False

    @ti.kernel
    def _compute_serial(
        self,
        pos_list: ti.types.ndarray(element_dim=1, dtype=float),
        pos_mean: ti.types.ndarray(dtype=float),
        pos_variance: ti.types.ndarray(dtype=float),
        lindemann_frame: ti.types.ndarray(dtype=float),
        lindemann_atom: ti.types.ndarray(dtype=float),
    ):

        Nframes, Natoms = pos_list.shape
        ti.loop_config(serialize=True)
        for frame in range(Nframes):
            for i in range(Natoms):
                for j in range(i + 1, Natoms):
                    rij = pos_list[frame, i] - pos_list[frame, j]
                    rijdis = rij.norm()
                    mean = pos_mean[i, j]
                    var = pos_variance[i, j]
                    delta = rijdis - mean
                    pos_mean[i, j] = mean + delta / (frame + 1)
                    pos_variance[i, j] = var + delta * (rijdis - pos_mean[i, j])
                    pos_mean[j, i] = pos_mean[i, j]
                    pos_variance[j, i] = pos_variance[i, j]

            lindemann_index = ti.float64(0.0)
            for i in range(Natoms):
                for j in range(Natoms):
                    # if i != j:
                    if pos_variance[i, j] > 0:
                        ldm = ti.sqrt(pos_variance[i, j] / Nframes) / pos_mean[i, j]
                        lindemann_index += ldm
                        lindemann_atom[frame, i] += ldm / (Natoms - 1)

            lindemann_index /= Natoms * (Natoms - 1)  # (Natoms-1)/2
            lindemann_frame[frame] = lindemann_index

    # @ti.kernel
    # def _compute_parallel(
    #     self,
    #     pos_list: ti.types.ndarray(element_dim=1, dtype=float),
    #     pos_mean: ti.types.ndarray(dtype=float),
    #     pos_variance: ti.types.ndarray(dtype=float),
    #     lindemann_frame: ti.types.ndarray(dtype=float),
    #     lindemann_atom: ti.types.ndarray(dtype=float),
    # ):

    #     Nframes, Natoms = pos_list.shape
    #     for frame, i in ti.ndrange(Nframes, Natoms):
    #         for j in range(i + 1, Natoms):
    #             rij = pos_list[frame, i] - pos_list[frame, j]
    #             rijdis = rij.norm()
    #             pos_mean[frame, i, j] = rijdis
    #             pos_mean[frame, j, i] = rijdis
    #             pos_variance[frame, i, j] = rijdis**2
    #             pos_variance[frame, j, i] = pos_variance[frame, i, j]

    #     ti.loop_config(serialize=True)  # Must run in serial
    #     for frame in range(1, Nframes):
    #         for i in range(Natoms):
    #             for j in range(i + 1, Natoms):
    #                 pos_mean[frame, i, j] += pos_mean[frame - 1, i, j]
    #                 pos_variance[frame, i, j] += pos_variance[frame - 1, i, j]
    #                 pos_mean[frame, j, i] = pos_mean[frame, i, j]
    #                 pos_variance[frame, j, i] = pos_variance[frame, i, j]

    #     for frame, i in ti.ndrange((1, Nframes), Natoms):
    #         for j in range(i + 1, Natoms):
    #             pos_mean[frame, i, j] /= frame + 1
    #             pos_variance[frame, i, j] /= frame + 1
    #             pos_mean[frame, j, i] = pos_mean[frame, i, j]
    #             pos_variance[frame, j, i] = pos_variance[frame, i, j]

    #     for frame in range(Nframes):
    #         lindemann_index = ti.float64(0.0)
    #         for i in range(Natoms):
    #             for j in range(Natoms):
    #                 if i != j:
    #                     ldm = (
    #                         ti.sqrt(
    #                             pos_variance[frame, i, j] - pos_mean[frame, i, j] ** 2
    #                         )
    #                         / pos_mean[frame, i, j]
    #                     )
    #                     lindemann_index += ldm
    #                     lindemann_atom[frame, i] += ldm / (Natoms - 1)

    #         lindemann_index /= Natoms * (Natoms - 1)  # (Natoms-1)/2
    #         lindemann_frame[frame] = lindemann_index

    def compute(self):
        # if self.parallel:
        #     Nframes, Natoms = self.pos_list.shape[:2]
        #     pos_mean = np.zeros((Nframes, Natoms, Natoms))
        #     pos_variance = np.zeros_like(pos_mean)
        #     self.lindemann_frame = np.zeros(Nframes)
        #     self.lindemann_atom = np.zeros((Nframes, Natoms))
        #     self._compute_parallel(
        #         self.pos_list,
        #         pos_mean,
        #         pos_variance,
        #         self.lindemann_frame,
        #         self.lindemann_atom,
        #     )
        #     self.lindemann_trj = self.lindemann_frame[-1]
        # else:
        Nframes, Natoms = self.pos_list.shape[:2]
        pos_mean = np.zeros((Natoms, Natoms))
        pos_variance = np.zeros_like(pos_mean)
        self.lindemann_frame = np.zeros(Nframes)
        self.lindemann_atom = np.zeros((Nframes, Natoms))
        self._compute_serial(
            self.pos_list,
            pos_mean,
            pos_variance,
            self.lindemann_frame,
            self.lindemann_atom,
        )
        self.lindemann_trj = self.lindemann_frame[-1]
        self.if_compute = True

    def plot(self):
        pltset()
        if not self.if_compute:
            self.compute()
        fig = plt.figure(figsize=(cm2inch(10), cm2inch(7)), dpi=150)
        plt.subplots_adjust(left=0.16, bottom=0.16, right=0.95, top=0.97)
        plt.plot(self.lindemann_frame, "o-")
        plt.legend()
        plt.xlabel("$\mathregular{N_{frames}}$")
        plt.ylabel("Lindemann index")
        plt.xlim(0, self.lindemann_frame.shape[0])

        ax = plt.gca()
        plt.show()
        return fig, ax


if __name__ == "__main__":
    from time import time

    ti.init(ti.cpu, offline_cache=True)
    Nframe, Nparticles = 200, 500
    pos_list = np.cumsum(
        np.random.choice([-1.0, 0.0, 1.0], size=(Nframe, Nparticles, 3)), axis=0
    )
    start = time()
    LDM = LindemannParameter(pos_list)
    LDM.compute()
    end = time()
    print(f"LDM_trj: {LDM.lindemann_trj}, LDM costs: {end-start} s.")
    print(LDM.lindemann_frame[:10])
    print(LDM.lindemann_atom.mean(axis=-1))
    # start = time()
    # LDM = LindemannParameter(pos_list, parallel=True)
    # LDM.compute()
    # end = time()
    # print(f"LDM_trj: {LDM.lindemann_trj}, LDM parallel costs: {end-start} s.")
    # print(LDM.lindemann_frame)
    # LDM.plot()
