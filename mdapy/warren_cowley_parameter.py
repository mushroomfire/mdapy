# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import taichi as ti
import numpy as np
import matplotlib.pyplot as plt


try:
    from plotset import set_figure
    from tool_function import _check_repeat_cutoff
    from replicate import Replicate
    from neighbor import Neighbor
except Exception:
    from .plotset import set_figure
    from .tool_function import _check_repeat_cutoff
    from .replicate import Replicate
    from .neighbor import Neighbor


@ti.data_oriented
class WarrenCowleyParameter:
    """This class is used to calculate the Warren Cowley parameter (WCP), which is useful to
    analyze the short-range order (SRO) in the 1st-nearest neighbor shell in alloy system and is given by:

    .. math:: WCP_{mn} = 1 - Z_{mn}/(X_n Z_{m}),

    where :math:`Z_{mn}` is the number of :math:`n`-type atoms around :math:`m`-type atoms,
    :math:`Z_m` is the total number of atoms around :math:`m`-type atoms, and :math:`X_n` is
    the atomic concentration of :math:`n`-type atoms in the alloy.

    .. note:: If you use this class in publication, you should also cite the original paper:
      `X‐Ray Measurement of Order in Single Crystals of Cu3Au <https://doi.org/10.1063/1.1699415>`_.

    Args:
        type_list (np.ndarray): (:math:`N_p`) atom type.
        verlet_list (np.ndarray, optional): (:math:`N_p, max\_neigh`) verlet_list[i, j] means j atom is a neighbor of i atom if j > -1.
        neighbor_number (np.ndarray, optional): (:math:`N_p`) neighbor atoms number.
        pos (np.ndarray, optional): (:math:`N_p, 3`) particles positions. Defaults to None.
        box (np.ndarray, optional): (:math:`3, 2`) or (:math:`4, 3`) system box. Defaults to None.
        boundary (list, optional): boundary conditions, 1 is periodic and 0 is free boundary. Such as [1, 1, 1]. Defaults to None.


    Outputs:
        - **WCP** (np.ndarray) - (:math:`N_{type}, N_{type}`) WCP for all pair elements

    Examples:
        >>> import mdapy as mp

        >>> mp.init()

        >>> system = mp.System("./example/CoCuFeNiPd-4M.dump") # Read a alloy DUMP file.

        >>> neigh = mp.Neighbor(system.pos, system.box, 3.0, max_neigh=30) # Initialize Neighbor class.

        >>> neigh.compute() # Calculate particle neighbor information.

        >>> wcp = mp.WarrenCowleyParameter(
            system.data["type"].values, neigh.verlet_list, neigh.neighbor_number
            ) # Initilize the WCP class.

        >>> wcp.compute() # Calculate the WCP.

        >>> wcp.WCP # Check the WCP matrix.

        This should get the same results with that in paper: `Simultaneously enhancing the ultimate strength and ductility of high-entropy alloys via short-range ordering <https://doi.org/10.1038/s41467-021-25264-5>`_.

        >>> wcp.plot(["Co", "Cu", "Fe", "Ni", "Pd"]) # Plot the results.
    """

    def __init__(
        self,
        type_list,
        verlet_list=None,
        neighbor_number=None,
        rc=None,
        pos=None,
        box=None,
        boundary=None,
    ):
        self.type_list = type_list - 1
        self.verlet_list = verlet_list
        self.neighbor_number = neighbor_number
        if verlet_list is None or neighbor_number is None:
            assert rc is not None
            assert pos is not None
            assert box is not None
            assert boundary is not None
            self.rc = rc
            repeat = _check_repeat_cutoff(box, boundary, self.rc, 4)

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
                repli = Replicate(pos, box, *repeat, type_list=type_list)
                repli.compute()
                self.pos = repli.pos
                self.box = repli.box
                self.type_list = repli.type_list - 1

            assert self.box[0, 1] == 0
            assert self.box[0, 2] == 0
            assert self.box[1, 2] == 0
            self.boundary = [int(boundary[i]) for i in range(3)]

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
        """Do the real WCP calculation."""
        if self.verlet_list is None or self.neighbor_number is None:
            neigh = Neighbor(self.pos, self.box, self.rc, self.boundary)
            neigh.compute()
            self.verlet_list, self.neighbor_number = (
                neigh.verlet_list,
                neigh.neighbor_number,
            )
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

        self.WCP = 1 - Zmn / (np.expand_dims(Alpha_n, 0) * np.expand_dims(Zm, 1))

        self.WCP = (self.WCP + self.WCP.T) / 2

    def plot(self, elements_list=None, vmin=-2, vmax=1, cmap="GnBu"):
        """Plot the WCP matrix.

        Args:
            elements_list (list, optional): elements list, such as ['Al', 'Ni', 'Co'].
            vmin (int, optional): vmin for colorbar. Defaults to -2.
            vmax (int, optional): vmax for colorbar. Defaults to 1.
            cmap (str, optional): cmap name. Defaults to "GnBu".

        Returns:
            tuple: (fig, ax) matplotlib figure and axis class.
        """
        fig, ax = set_figure(
            figsize=(8, 8), bottom=0.1, top=0.9, left=0.15, right=0.82, use_pltset=True
        )

        h = ax.imshow(self.WCP[::-1], vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_xticks(np.arange(self.WCP.shape[0]))
        ax.set_yticks(np.arange(self.WCP.shape[0]))
        if elements_list is not None:
            ax.set_xticklabels(elements_list)
            ax.set_yticklabels(elements_list[::-1])
        else:
            ax.set_yticklabels(np.arange(self.WCP.shape[0])[::-1])

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
        bar.set_ticks(ticks=[vmin, 0, vmax])
        bar.set_label("WCP")
        plt.show()
        return fig, ax


if __name__ == "__main__":
    from mdapy import System
    from neighbor import Neighbor
    from time import time
    from lattice_maker import LatticeMaker

    # ti.init(ti.gpu, device_memory_GB=2.0)
    ti.init(ti.cpu, offline_cache=True)
    # system = LatticeMaker(3.615, "FCC", 1, 1, 1)
    # system.compute()\
    # system = System(r"F:\HEARes\HEA-Paper\SFE\MDMC\relax.0.data")
    system = System(r"./example/CoCuFeNiPd-4M.dump")
    # start = time()
    # neigh = Neighbor(system.pos, system.box, 3.0, max_neigh=30)
    # neigh.compute()
    # end = time()
    # print(f"Build neighbor time: {end-start} s.")
    start = time()
    # system.data["type"].values
    wcp = WarrenCowleyParameter(
        system.data["type"].to_numpy(),
        None,
        None,
        3.0,
        system.pos,
        system.box,
        [1, 1, 1],
    )
    wcp.compute()
    end = time()
    print(f"Cal WCP time: {end-start} s.")
    print("WCP matrix is:")
    print(wcp.WCP)
    wcp.plot()
    # fig, ax = wcp.plot(["Co", "Cu", "Fe", "Ni", "Pd"])
    # fig.savefig('wcp.png', dpi=300, bbox_inches='tight', transparent=True)
