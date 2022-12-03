import taichi as ti
import numpy as np
from mdapy.plot.pltset import pltset, cm2inch
import matplotlib.pyplot as plt


@ti.data_oriented
class LindemannParameter:
    """
    Using Welford method to updated the varience and mean of rij
    see
    1. https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford
    2. https://zhuanlan.zhihu.com/p/408474710

    Input:
    pos_list : np.ndarray (Nframes, Natoms, 3)
    only_globle : bool, only calculate globle lindemann index, fast, parallel and relatively low memory
    need_frame : bool, calculate lindemann index per frame, serial compute is conducted, relatively low memory
    need_atoms : bool, calculate atomic contribution on lindemann index, slow and need very high memory

    Output:
    lindemann_atom : np.ndarray (Nframes, Natoms)
    lindemann_frame : np.ndarray (Nframes)
    lindemann_trj : float
    """

    def __init__(
        self, pos_list, only_globle=True, need_frame=False, need_atom=False
    ) -> None:
        self.pos_list = pos_list
        self.only_globle = only_globle
        self.need_frame = need_frame
        self.need_atom = need_atom
        self.if_compute = False

    @ti.kernel
    def _compute_globle(
        self,
        pos_list: ti.types.ndarray(element_dim=1),
        pos_mean: ti.types.ndarray(),
        pos_variance: ti.types.ndarray(),
    ) -> float:

        Nframes, Natoms = pos_list.shape
        factor = pos_mean.shape[0]
        for frame in range(Nframes):
            num = 0
            for i in range(Natoms):
                for j in range(i + 1, Natoms):
                    rijdis = (pos_list[frame, i] - pos_list[frame, j]).norm()
                    pos_mean[num] += rijdis
                    pos_variance[num] += rijdis**2
                    num += 1

        lin_index = ti.float64(0.0)
        for m in range(factor):
            rij_squared_mean = pos_variance[m] / Nframes
            rij_mean = pos_mean[m] / Nframes
            delta = rij_squared_mean - rij_mean**2
            lin_index += ti.sqrt(delta) / rij_mean
        return lin_index / factor

    @ti.kernel
    def _compute_frame(
        self,
        pos_list: ti.types.ndarray(element_dim=1),
        pos_mean: ti.types.ndarray(),
        pos_variance: ti.types.ndarray(),
        lindemann_frame: ti.types.ndarray(),
    ):

        Nframes, Natoms = pos_list.shape
        factor = pos_mean.shape[0]
        ti.loop_config(serialize=True)
        for frame in range(Nframes):
            i = 0
            for atom1 in range(Natoms):
                for atom2 in range(atom1 + 1, Natoms):
                    rijdis = (pos_list[frame, atom1] - pos_list[frame, atom2]).norm()
                    pos_mean[i] += rijdis
                    pos_variance[i] += rijdis**2
                    i += 1
            lin_index = ti.float64(0.0)
            for m in range(factor):
                rij_squared_mean = pos_variance[m] / (frame + 1)
                rij_mean = pos_mean[m] / (frame + 1)
                delta = rij_squared_mean - rij_mean**2
                if delta > 0:
                    lin_index += ti.sqrt(delta) / rij_mean
            lindemann_frame[frame] = lin_index / factor

    @ti.kernel
    def _compute_all(
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
                        ldm = ti.sqrt(pos_variance[i, j] / (frame + 1)) / pos_mean[i, j]
                        lindemann_index += ldm
                        lindemann_atom[frame, i] += ldm / (Natoms - 1)

            lindemann_index /= Natoms * (Natoms - 1)  # (Natoms-1)/2
            lindemann_frame[frame] = lindemann_index

    def compute(self):
        Nframes, Natoms = self.pos_list.shape[:2]

        if self.only_globle:
            pos_mean = np.zeros(int(Natoms * (Natoms - 1) / 2), dtype=np.float64)
            pos_variance = np.zeros_like(pos_mean, dtype=pos_mean.dtype)
            self.lindemann_trj = self._compute_globle(
                self.pos_list, pos_mean, pos_variance
            )

        elif self.need_frame:
            pos_mean = np.zeros(int(Natoms * (Natoms - 1) / 2), dtype=np.float64)
            pos_variance = np.zeros_like(pos_mean, dtype=pos_mean.dtype)
            self.lindemann_frame = np.zeros(Nframes)
            self._compute_frame(
                self.pos_list, pos_mean, pos_variance, self.lindemann_frame
            )
            self.lindemann_trj = self.lindemann_frame[-1]
            self.if_compute = True
        elif self.need_atom:
            pos_mean = np.zeros((Natoms, Natoms))
            pos_variance = np.zeros_like(pos_mean)
            self.lindemann_frame = np.zeros(Nframes)
            self.lindemann_atom = np.zeros((Nframes, Natoms))
            self._compute_all(
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
            raise Exception("One should compute lidemann_frame first!")
        fig = plt.figure(figsize=(cm2inch(10), cm2inch(7)), dpi=150)
        plt.subplots_adjust(left=0.16, bottom=0.16, right=0.95, top=0.97)
        plt.plot(self.lindemann_frame, "o-")
        plt.xlabel("$\mathregular{N_{frames}}$")
        plt.ylabel("Lindemann index")
        # plt.xlim(0, self.lindemann_frame.shape[0])

        ax = plt.gca()
        plt.show()
        return fig, ax


if __name__ == "__main__":
    from time import time

    ti.init(ti.cpu, offline_cache=True)
    Nframe, Nparticles = 1000, 500
    pos_list = np.cumsum(
        np.random.choice([-1.0, 0.0, 1.0], size=(Nframe, Nparticles, 3)), axis=0
    )
    start = time()
    LDM = LindemannParameter(pos_list, only_globle=False, need_frame=True)
    LDM.compute()
    end = time()
    print(f"LDM_trj: {LDM.lindemann_trj}, LDM costs: {end-start} s.")
    print(LDM.lindemann_frame[:10])
    LDM.plot()

    start = time()
    LDM = LindemannParameter(
        pos_list, only_globle=False, need_frame=False, need_atom=True
    )
    LDM.compute()
    end = time()
    print(f"LDM_trj: {LDM.lindemann_trj}, LDM costs: {end-start} s.")
    print(LDM.lindemann_frame[:10])
    LDM.plot()

    start = time()
    LDM = LindemannParameter(
        pos_list, only_globle=True, need_frame=False, need_atom=False
    )
    LDM.compute()
    end = time()
    print(f"LDM_trj: {LDM.lindemann_trj}, LDM costs: {end-start} s.")

    # print(LDM.lindemann_frame[:10])
    # print(LDM.lindemann_atom.mean(axis=-1)[:10])
