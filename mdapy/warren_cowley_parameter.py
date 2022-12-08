# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
from .plotset import pltset, cm2inch


@ti.data_oriented
class WarrenCowleyParameter:
    """
    Calculate the average WCP parameter for system, to characteristic the short range order in alloy.
    input:
    verlet_list : ndarray
    neighbor : ndarray
    type_list : ndarray
    return :
    WCP : ndarray
    """

    def __init__(self, verlet_list, neighbor_number, type_list):
        self.verlet_list = verlet_list
        self.neighbor_number = neighbor_number
        self.type_list = type_list - 1
        pltset()

    @ti.kernel
    def _get_wcp(
        self,
        verlet_list: ti.types.ndarray(),
        neighbor_number: ti.types.ndarray(),
        type_list: ti.types.ndarray(),
        Ntype: int,
        Zmn: ti.types.ndarray(),
        Zm: ti.types.ndarray(),
        Alpha_n: ti.types.ndarray(),
    ):
        N = type_list.shape[0]
        for i in range(N):
            Alpha_n[type_list[i]] += 1
            for m_type in range(Ntype):
                N_i_neigh = neighbor_number[i]
                if type_list[i] == m_type:
                    for jj in range(N_i_neigh):
                        j = verlet_list[i, jj]
                        Zmn[type_list[i], type_list[j]] += 1
                    Zm[m_type] += N_i_neigh

        ti.loop_config(serialize=True)
        for i in range(Ntype):
            Alpha_n[i] /= N

    def compute(self):
        Ntype = len(np.unique(self.type_list))
        Zmn = np.zeros((Ntype, Ntype), dtype=np.int32)
        Zm = np.zeros(Ntype, dtype=np.int32)
        Alpha_n = np.zeros(Ntype)
        self._get_wcp(
            self.verlet_list,
            self.neighbor_number,
            self.type_list,
            Ntype,
            Zmn,
            Zm,
            Alpha_n,
        )
        self.WCP = 1 - Zmn / (Alpha_n * Zm)

    def plot(self, elements_list=None, vmin=-2, vmax=1, cmap="GnBu"):
        fig, ax = plt.subplots(figsize=(cm2inch(8), cm2inch(8)), dpi=150)
        plt.subplots_adjust(bottom=0.1, top=0.9, left=0.15, right=0.82)
        h = ax.imshow(self.WCP[::-1], vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_xticks(np.arange(self.WCP.shape[0]))
        ax.set_yticks(np.arange(self.WCP.shape[0]))
        if elements_list is not None:
            ax.set_xticklabels(elements_list)
            ax.set_yticklabels(elements_list[::-1])

        for i in range(self.WCP.shape[0]):
            for j in range(self.WCP.shape[1]):
                if self.WCP[i, j] == 0:
                    name = "0.00"
                else:
                    name = f"{np.round(self.WCP[::-1][i, j], 2)}"
                ax.text(j, i, name, ha="center", va="center", color="k")

        ax.set_xlabel("Central element")
        ax.set_ylabel("Neighboring element")

        baraxes = fig.add_axes([0.83, 0.165, 0.03, 0.67])
        bar = fig.colorbar(h, ax=ax, cax=baraxes)
        bar.set_ticks([vmin, 0, vmax], fontsize=8)
        bar.set_label("WCP")
        plt.show()
        return fig, ax


if __name__ == "__main__":
    from mdapy import System
    from neighbor import Neighbor
    from time import time

    # ti.init(ti.gpu, device_memory_GB=2.0)
    ti.init(ti.cpu, offline_cache=True)

    # file = open("./example/CoCuFeNiPd-4M.data").readlines()
    # box = np.array([i.split()[:2] for i in file[6:9]], dtype=float)
    # data = np.array([i.split() for i in file[12:]], dtype=float)
    # pos = data[:, 2:]
    # type_list = data[:, 1].astype(int)
    system = System("./example/CoCuFeNiPd-4M.dump")
    start = time()
    neigh = Neighbor(system.pos, system.box, 3.0, max_neigh=30)
    neigh.compute()
    end = time()
    print(f"Build neighbor time: {end-start} s.")
    start = time()
    wcp = WarrenCowleyParameter(
        neigh.verlet_list, neigh.neighbor_number, system.data["type"].values
    )
    wcp.compute()
    end = time()
    print(f"Cal WCP time: {end-start} s.")
    # print("WCP matrix is:")
    # print(wcp.WCP)
    wcp.plot(["Co", "Cu", "Fe", "Ni", "Pd"])
