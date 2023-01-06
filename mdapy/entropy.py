# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import taichi as ti
import numpy as np


@ti.data_oriented
class AtomicEntropy:
    """This class is used to calculate the entropy fingerprint, which is useful to distinguish
    between ordered and disordered environments, including liquid and solid-like environments,
    or glassy and crystalline-like environments. The potential application could identificate grain boundaries
    or a solid cluster emerging from the melt. One of the advantages of this parameter is that no a priori
    information of structure is required.

    This parameter for atom :math:`i` is computed using the following formula:

    .. math:: s_S^i=-2\\pi\\rho k_B \\int\\limits_0^{r_m} \\left [ g(r) \\ln g(r) - g(r) + 1 \\right ] r^2 dr,

    where :math:`r` is a distance, :math:`g(r)` is the radial distribution function,
    and :math:`\\rho` is the density of the system.
    The :math:`g(r)` computed for each atom :math:`i` can be noisy and therefore it can be smoothed using:

    .. math:: g_m^i(r) = \\frac{1}{4 \\pi \\rho r^2} \\sum\\limits_{j} \\frac{1}{\\sqrt{2 \\pi \\sigma^2}} e^{-(r-r_{ij})^2/(2\\sigma^2)},

    where the sum over :math:`j` goes through the neighbors of atom :math:`i` and :math:`\\sigma` is
    a parameter to control the smoothing. The average of the parameter over the neighbors of atom :math:`i`
    is calculated according to:

    .. math:: \\left< s_S^i \\right>  = \\frac{\\sum_j s_S^j + s_S^i}{N + 1},

    where the sum over :math:`j` goes over the neighbors of atom :math:`i` and :math:`N` is the number of neighbors.
    The average version always provides a sharper distinction between order and disorder environments.

    .. note:: If you use this module in publication, you should also cite the original paper.
      `Entropy based fingerprint for local crystalline order <https://doi.org/10.1063/1.4998408>`_

    .. note:: This class uses the `same algorithm with LAMMPS <https://docs.lammps.org/compute_entropy_atom.html>`_.

    .. tip:: Suggestions for FCC, the :math:`rc = 1.4a` and :math:`average\_rc = 0.9a` and
      for BCC, the :math:`rc = 1.8a` and :math:`average\_rc = 1.2a`, where the :math:`a`
      is the lattice constant.

    Args:
        vol (float): system volume.
        verlet_list (np.ndarray): (:math:`N_p, max\_neigh`) verlet_list[i, j] means j atom is a neighbor of i atom if j > -1.
        distance_list (np.ndarray): (:math:`N_p, max\_neigh`) distance_list[i, j] means distance between i and j atom.
        rc (float, optional): cutoff distance. Defaults to 5.0.
        sigma (float, optional): smoothing parameter. Defaults to 0.2.
        use_local_density (bool, optional): whether use local atomic volume. Defaults to False.
        compute_average (bool, optional): whether compute the average version. Defaults to False.
        average_rc (float, optional): cutoff distance for averaging operation, if not given, it is equal to rc. This parameter should be lower than rc.

    Outputs:
        - **entropy** (np.ndarray) - (:math:`N_p`), atomic entropy.
        - **entropy_average** (np.ndarray) - (:math:`N_p`), averaged atomic entropy if compute_average is True.

    Examples:
        >>> import mdapy as mp

        >>> mp.init()

        >>> import numpy as np

        >>> FCC = mp.LatticeMaker(3.615, 'FCC', 10, 10, 10) # Create a FCC structure

        >>> FCC.compute() # Get atom positions

        >>> neigh = mp.Neighbor(FCC.pos, FCC.box,
                                3.615*1.4, max_neigh=80) # Initialize Neighbor class.

        >>> neigh.compute() # Calculate particle neighbor information.

        >>> vol = np.product(FCC.box[:, 1] - FCC.box[:, 0]) # Calculate system volume.

        >>> Entropy = mp.AtomicEntropy(
                vol,
                neigh.verlet_list,
                neigh.distance_list,
                neigh.rc,
                sigma=0.2,
                use_local_density=False,
                compute_average=True,
                average_rc=3.615*0.9,
            ) # Initilize the entropy class.

        >>> Entropy.compute() # Calculate the atomic entropy.

        >>> Entropy.entropy # Check atomic entropy.

        >>> Entropy.entropy_average # Check averaged atomic entropy.
    """

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
        g_m: ti.types.ndarray(),
        intergrad: ti.types.ndarray(),
    ):
        for i in range(self.N):
            n_neigh = 0
            for j in range(self.nbins):
                for k in range(distance_list.shape[1]):
                    if distance_list[i, k] <= self.rc:
                        g_m[i, j] += (
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
                for j in range(self.nbins):
                    g_m[i, j] *= self.global_density / density
            else:
                density = self.global_density

            for j in range(self.nbins):
                if g_m[i, j] >= 1e-10:
                    intergrad[i, j] = (
                        g_m[i, j] * ti.log(g_m[i, j]) - g_m[i, j] + 1.0
                    ) * rlist_sq[j]
                else:
                    intergrad[i, j] = rlist_sq[j]

            sum_intergrad = ti.float64(0.0)
            for j in range(self.nbins - 1):
                sum_intergrad += (intergrad[i, j] + intergrad[i, j + 1]) * (
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
        """Do the real entropy calculation."""
        self.N = self.distance_list.shape[0]
        self.entropy = np.zeros(self.N)
        self.global_density = self.N / self.vol
        self.nbins = int(np.floor(self.rc / self.sigma) + 1)
        g_m = np.zeros((self.N, self.nbins))
        intergrad = np.zeros_like(g_m)
        rlist = np.linspace(0.0, self.rc, self.nbins)
        rlist_sq = rlist**2
        prefactor = rlist_sq * (
            4 * np.pi * self.global_density * np.sqrt(2 * np.pi * self.sigma**2)
        )
        prefactor[0] = prefactor[1]
        self._compute(
            self.distance_list, rlist, rlist_sq, prefactor, self.entropy, g_m, intergrad
        )
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
    neigh = Neighbor(FCC.pos, FCC.box, lattice_constant * 1.4, max_neigh=60)
    neigh.compute()
    end = time()
    print(f"Build neighbor time: {end-start} s.")
    print(neigh.neighbor_number.max())

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
        average_rc=lattice_constant * 0.9,
    )
    Entropy.compute()
    entropy = Entropy.entropy
    entropy_ave = Entropy.entropy_average
    end = time()
    print(f"Cal entropy time: {end-start} s.")
    print(entropy)
    print(entropy_ave)
