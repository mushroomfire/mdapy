# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    from rdf._rdf import _rdf, _rdf_single_species
    from plotset import pltset, cm2inch
else:
    from _rdf import _rdf, _rdf_single_species
    from .plotset import pltset, cm2inch


class PairDistribution:
    """This class is used to calculate the radiul distribution function (RDF),which
    reflects the probability of finding an atom at distance r. The seperate pair-wise
    combinations of particle types can also be computed:

    .. math:: g(r) = c_{\\alpha}^2 g_{\\alpha \\alpha}(r) + 2 c_{\\alpha} c_{\\beta} g_{\\alpha \\beta}(r) + c_{\\beta}^2 g_{\\beta \\beta}(r),

    where :math:`c_{\\alpha}` and :math:`c_{\\beta}` denote the concentration of two atom types in system
    and :math:`g_{\\alpha \\beta}(r)=g_{\\beta \\alpha}(r)`.

    Args:
        rc (float): cutoff distance.
        nbin (int): number of bins.
        rho (float): system density.
        verlet_list (np.ndarray): (:math:`N_p, max\_neigh`) verlet_list[i, j] means j atom is a neighbor of i atom if j > -1.
        distance_list (np.ndarray): (:math:`N_p, max\_neigh`) distance_list[i, j] means distance between i and j atom.
        neighbor_number (np.ndarray): (:math:`N_p`) neighbor atoms number.
        type_list (np.ndarray, optional): (:math:`N_p`) atom type list. If not given, all atoms types are set as 1.

    Outputs:
        - **r** (np.ndarray) - (nbin), distance.
        - **g_total** (np.ndarray) - (nbin), global RDF.
        - **Ntype** (int) - number of species.
        - **g** (np.ndarray): (:math:`N_{type}, N_{type}, nbin`), partial RDF.

    Examples:
        >>> import mdapy as mp

        >>> mp.init()

        >>> FCC = mp.LatticeMaker(3.615, 'FCC', 10, 10, 10) # Create a FCC structure.

        >>> FCC.compute() # Get atom positions.

        >>> neigh = mp.Neighbor(FCC.pos, FCC.box,
                                5., max_neigh=80) # Initialize Neighbor class.

        >>> neigh.compute() # Calculate particle neighbor information.

        >>> rho = FCC.pos.shape[0] / np.product(FCC.box[:, 1] - FCC.box[:, 0]) # Get the system density.

        >>> gr = mp.PairDistribution(neigh.rc, 200, rho,
                                  neigh.verlet_list, neigh.distance_list, neigh.neighbor_number,
                                  np.ones(FCC.N, dtype=int)) # Initilize the RDF class.

        >>> gr.compute() # Calculate the RDF.

        >>> gr.g_total # Check global RDF.

        >>> gr.g # Check partial RDF, here is the same due to only one species.

        >>> gr.r # Check the distance.

        >>> gr.plot() # Plot global RDF.

        >>> gr.plot_partial() # Plot partial RDF.

    """

    def __init__(
        self, rc, nbin, rho, verlet_list, distance_list, neighbor_number, type_list=None
    ):

        self.rc = rc
        self.nbin = nbin
        self.rho = rho
        self.verlet_list = verlet_list
        self.distance_list = distance_list
        self.neighbor_number = neighbor_number
        self.N = self.distance_list.shape[0]
        if type_list is not None:
            self.type_list = type_list - 1
        else:
            self.type_list = np.zeros(self.N, dtype=int)
        self.Ntype = len(np.unique(self.type_list))
        pltset()

    def compute(self):
        """Do the real RDF calculation."""
        r = np.linspace(0, self.rc, self.nbin + 1)
        const = 4.0 * np.pi * self.rho / 3.0
        if self.Ntype > 1:
            concentrates = (
                np.array(
                    [
                        len(self.type_list[self.type_list == i])
                        for i in range(self.Ntype)
                    ]
                )
                / self.N
            )
            self.g = np.zeros((self.Ntype, self.Ntype, self.nbin), dtype=np.float64)
            _rdf(
                self.verlet_list,
                self.distance_list,
                self.neighbor_number,
                self.type_list,
                self.g,
                concentrates,
                self.rc,
                self.nbin,
            )

            self.g /= self.N * const * (r[1:] ** 3 - r[:-1] ** 3)
            self.r = (r[1:] + r[:-1]) / 2
            self.g_total = np.zeros_like(self.r)
            for i in range(self.Ntype):
                for j in range(self.Ntype):
                    if j == i:
                        self.g_total += concentrates[i] * concentrates[j] * self.g[i, j]
                    else:
                        self.g_total += (
                            2 * concentrates[i] * concentrates[j] * self.g[i, j]
                        )
        else:
            self.g_total = np.zeros(self.nbin)
            _rdf_single_species(
                self.verlet_list,
                self.distance_list,
                self.neighbor_number,
                self.g_total,
                self.rc,
                self.nbin,
            )
            self.g_total /= self.N * const * (r[1:] ** 3 - r[:-1] ** 3)
            self.r = (r[1:] + r[:-1]) / 2
            self.g = np.zeros((self.Ntype, self.Ntype, self.nbin), dtype=np.float64)
            self.g[0, 0] = self.g_total

    def plot(self):
        """Plot the global RDF.

        Returns:
            tuple: (fig, ax) matplotlib figure and axis class.
        """
        fig = plt.figure(figsize=(cm2inch(8), cm2inch(5)), dpi=150)
        plt.subplots_adjust(left=0.16, bottom=0.225, right=0.97, top=0.97)
        plt.plot(self.r, self.g_total, "o-", ms=3)
        plt.xlabel("r ($\mathregular{\AA}$)")
        plt.ylabel("g(r)")
        plt.xlim(0, self.rc)
        ax = plt.gca()
        plt.show()
        return fig, ax

    def plot_partial(self, elements_list=None):
        """Plot the partial RDF.

        Args:
            elements_list (list, optional): species element names list, such as ['Al', 'Ni'].

        Returns:
            tuple: (fig, ax) matplotlib figure and axis class.
        """
        if elements_list is not None:
            assert len(elements_list) == self.Ntype
        fig = plt.figure(figsize=(cm2inch(8), cm2inch(5)), dpi=150)
        plt.subplots_adjust(left=0.16, bottom=0.225, right=0.97, top=0.97)
        lw = 1.0
        if self.Ntype > 3:
            colorlist = plt.cm.get_cmap("tab20").colors[::2]
        else:
            colorlist = [i["color"] for i in plt.rcParams["axes.prop_cycle"]]
        n = 0
        for i in range(self.Ntype):
            for j in range(self.Ntype):
                if j >= i:
                    if n > len(colorlist) - 1:
                        n = 0
                    if elements_list is not None:
                        plt.plot(
                            self.r,
                            self.g[i, j],
                            c=colorlist[n],
                            lw=lw,
                            label=f"{elements_list[i]}-{elements_list[j]}",
                        )
                    else:
                        plt.plot(
                            self.r,
                            self.g[i, j],
                            c=colorlist[n],
                            lw=lw,
                            label=f"{i}-{j}",
                        )
                    n += 1
        plt.legend(ncol=2, fontsize=6)
        plt.xlabel("r ($\mathregular{\AA}$)")
        plt.ylabel("g(r)")
        plt.xlim(0, self.rc)
        ax = plt.gca()
        plt.show()
        return fig, ax


if __name__ == "__main__":
    import taichi as ti
    from lattice_maker import LatticeMaker
    from neighbor import Neighbor
    from time import time

    ti.init(ti.gpu, device_memory_GB=4.0)
    # ti.init(ti.cpu)
    start = time()
    lattice_constant = 3.615
    x, y, z = 50, 50, 50
    FCC = LatticeMaker(lattice_constant, "FCC", x, y, z)
    FCC.compute()
    end = time()
    print(f"Build {FCC.pos.shape[0]} atoms FCC time: {end-start} s.")
    # FCC.write_data()
    start = time()
    rc = 5.0
    neigh = Neighbor(FCC.pos, FCC.box, rc, max_neigh=50)
    neigh.compute()
    end = time()
    print(f"Build neighbor time: {end-start} s.")
    start = time()
    rho = FCC.pos.shape[0] / np.product(FCC.box[:, 1] - FCC.box[:, 0])
    # type_list = np.r_[
    #     np.ones(int(FCC.N / 2), dtype=int), np.ones(int(FCC.N / 2), dtype=int) + 1
    # ]
    type_list = np.ones(FCC.N, int)
    gr = PairDistribution(
        rc,
        200,
        rho,
        neigh.verlet_list,
        neigh.distance_list,
        neigh.neighbor_number,
        type_list,
    )
    gr.compute()
    end = time()
    print(f"Cal gr time: {end-start} s.")
    gr.plot()
    gr.plot_partial()
