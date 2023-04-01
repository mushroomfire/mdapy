# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import taichi as ti
import numpy as np

if __name__ == "__main__":
    from kdtree import kdtree
else:
    from .kdtree import kdtree


@ti.data_oriented
class AcklandJonesAnalysis:
    """This class applies Ackland Jones Analysis (AJA) method to identify the lattice structure.

    The AJA method can recgonize the following structure:

    1. Other
    2. FCC
    3. HCP
    4. BCC
    5. ICO

    .. note:: If you use this module in publication, you should also cite the original paper.
      `Ackland G J, Jones A P. Applications of local crystal structure measures in experiment
      and simulation[J]. Physical Review B, 2006, 73(5): 054104. <https://doi.org/10.1103/PhysRevB.73.054104>`_.

    .. hint:: The module uses the `legacy algorithm in LAMMPS <https://docs.lammps.org/compute_ackland_atom.html>`_.

    Args:
        pos (np.ndarray): (:math:`N_p, 3`) particles positions.
        box (np.ndarray): (:math:`3, 2`) system box.
        boundary (list, optional): boundary conditions, 1 is periodic and 0 is free boundary. Defaults to [1, 1, 1].
        verlet_list (np.ndarray, optional): (:math:`N_p`, >=14), first 14 neighbors is sorted, if not given, use kdtree to obtain it. Defaults to None.
        distance_list (np.ndarray, optional): (:math:`N_p`, >=14), first 14 neighbors is sorted, if not given, use kdtree to obtain it. Defaults to None.

    Outputs:
        - **aja** (np.ndarray) - (:math:`N_p`) AJA value per atoms.
        - **structure** (list) - the corresponding structure to each aja value.

    Examples:

        >>> import mdapy as mp

        >>> mp.init()

        >>> FCC = mp.LatticeMaker(3.615, 'FCC', 10, 10, 10) # Create a FCC structure

        >>> FCC.compute() # Get atom positions

        >>> AJA = mp.AcklandJonesAnalysis(FCC.pos, FCC.box, [1, 1, 1]) # Initialize AJA class

        >>> AJA.compute() # Calculate the aja per atoms

        >>> AJA.aja # Check the aja value
    """

    def __init__(
        self, pos, box, boundary=[1, 1, 1], verlet_list=None, distance_list=None
    ):

        self.pos = pos
        self.box = box
        self.boundary = np.array(boundary)
        self.verlet_list = verlet_list
        self.distance_list = distance_list
        self.structure = ["other", "fcc", "hcp", "bcc", "ico"]

    @ti.func
    def _pbc(self, rij, box: ti.types.ndarray(), boundary: ti.types.ndarray()):
        for i in ti.static(range(3)):
            if boundary[i] == 1:
                box_length = box[i, 1] - box[i, 0]
                rij[i] = rij[i] - box_length * ti.round(rij[i] / box_length)
        return rij

    @ti.kernel
    def _compute(
        self,
        pos: ti.types.ndarray(dtype=ti.math.vec3),
        box: ti.types.ndarray(),
        boundary: ti.types.ndarray(),
        verlet_list: ti.types.ndarray(),
        distance_list: ti.types.ndarray(),
        aja: ti.types.ndarray(),
    ):

        for i in range(pos.shape[0]):
            r0_sq = ti.f64(0.0)
            for j in range(6):
                r0_sq += distance_list[i, j] ** 2
            r0_sq /= 6.0
            N0, N1 = 0, 0
            for j in range(14):
                rij_sq = distance_list[i, j] ** 2
                if rij_sq < 1.55 * r0_sq:
                    N1 += 1
                    if rij_sq < 1.45 * r0_sq:
                        N0 += 1
            alpha = ti.Vector([0, 0, 0, 0, 0, 0, 0, 0])
            for j in range(N0):
                for k in range(j + 1, N0):
                    rij = pos[verlet_list[i, j]] - pos[i]
                    rik = pos[verlet_list[i, k]] - pos[i]
                    rij = self._pbc(rij, box, boundary)
                    rik = self._pbc(rik, box, boundary)
                    cos_theta = rij.dot(rik) / (
                        distance_list[i, j] * distance_list[i, k]
                    )
                    if cos_theta < -0.945:
                        alpha[0] += 1
                    elif cos_theta < -0.915:
                        alpha[1] += 1
                    elif cos_theta < -0.755:
                        alpha[2] += 1
                    elif cos_theta < -0.195:
                        alpha[3] += 1
                    elif cos_theta < 0.195:
                        alpha[4] += 1
                    elif cos_theta < 0.245:
                        alpha[5] += 1
                    elif cos_theta < 0.795:
                        alpha[6] += 1
                    else:
                        alpha[7] += 1
            sigma_cp = ti.abs(1.0 - alpha[6] / 24.0)
            s56m4 = alpha[5] + alpha[6] - alpha[4]
            sigma_bcc = sigma_cp + 1.0
            if s56m4 != 0:
                sigma_bcc = 0.35 * alpha[4] / s56m4
            sigma_fcc = 0.61 * (ti.abs(alpha[0] + alpha[1] - 6) + alpha[2]) / 6.0
            sigma_hcp = (
                ti.abs(alpha[0] - 3.0)
                + ti.abs(alpha[0] + alpha[1] + alpha[2] + alpha[3] - 9)
            ) / 12

            if alpha[0] == 7:
                sigma_bcc = 0.0
            elif alpha[0] == 6:
                sigma_fcc = 0.0
            elif alpha[0] <= 3:
                sigma_hcp = 0.0

            if alpha[7] > 0:
                aja[i] = 0
            elif alpha[4] < 3:
                if N1 > 13 or N1 < 11:
                    aja[i] = 0  # Other
                else:
                    aja[i] = 4  # ICO
            elif sigma_bcc <= sigma_cp:
                if N1 < 11:
                    aja[i] = 0  # Other
                else:
                    aja[i] = 3  # BCC
            elif N1 > 12 or N1 < 11:
                aja[i] = 0  # Other
            else:
                if sigma_fcc < sigma_hcp:
                    aja[i] = 1  # FCC
                else:
                    aja[i] = 2  # HCP

    def compute(self):
        """Do the real AJA calculation."""
        verlet_list, distance_list = self.verlet_list, self.distance_list
        if verlet_list is None or distance_list is None:
            kdt = kdtree(self.pos, self.box, self.boundary)
            distance_list, verlet_list = kdt.query_nearest_neighbors(14)
        self.aja = np.zeros(self.pos.shape[0], dtype=int)
        self._compute(
            self.pos, self.box, self.boundary, verlet_list, distance_list, self.aja
        )


if __name__ == "__main__":
    from lattice_maker import LatticeMaker
    from time import time
    from neighbor import Neighbor

    # ti.init(ti.gpu, device_memory_GB=5.0)
    ti.init(ti.cpu)
    start = time()
    lattice_constant = 4.05
    x, y, z = 100, 100, 100
    FCC = LatticeMaker(lattice_constant, "FCC", x, y, z)
    FCC.compute()
    end = time()
    print(f"Build {FCC.pos.shape[0]} atoms FCC time: {end-start} s.")

    start = time()
    AJA = AcklandJonesAnalysis(FCC.pos, FCC.box, [1, 1, 1])
    AJA.compute()
    aja = AJA.aja
    end = time()
    print(f"Cal aja kdt time: {end-start} s.")
    for i in range(5):
        print(AJA.structure[i], sum(aja == i))

    Neigh = Neighbor(FCC.pos, FCC.box, 4.05, max_neigh=30)
    Neigh.compute()
    print(Neigh.neighbor_number.min())

    start = time()
    Neigh.sort_verlet_by_distance(14)
    end = time()
    print(f"taichi sort time: {end-start} s.")

    start = time()
    AJA = AcklandJonesAnalysis(
        FCC.pos, FCC.box, [1, 1, 1], Neigh.verlet_list, Neigh.distance_list
    )
    AJA.compute()
    aja = AJA.aja
    end = time()
    print(f"Cal aja sort time: {end-start} s.")
    for i in range(5):
        print(AJA.structure[i], sum(aja == i))
