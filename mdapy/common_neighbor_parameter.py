# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import taichi as ti
import numpy as np


@ti.data_oriented
class CommonNeighborParameter:
    """This class use Common Neighbor Parameter (CNP) method to recgonize the lattice structure.

    .. note:: If one use this module in publication, one should also cite the original paper.
      `Tsuzuki H, Branicio P S, Rino J P. Structural characterization of deformed crystals by analysis of common atomic neighborhood[J].
      Computer physics communications, 2007, 177(6): 518-523. <https://doi.org/10.1016/j.cpc.2007.05.018>`_.

    CNP method is sensitive to the given cutoff distance. The suggesting cutoff can be obtained from the
    following formulas:

    .. math::

        r_{c}^{\mathrm{fcc}} = \\frac{1}{2} \\left(\\frac{\\sqrt{2}}{2} + 1\\right) a
        \\approx 0.8536 a,

    .. math::

        r_{c}^{\mathrm{bcc}} = \\frac{1}{2}(\\sqrt{2} + 1) a
        \\approx 1.207 a,

    .. math::

        r_{c}^{\mathrm{hcp}} = \\frac{1}{2}\\left(1+\\sqrt{\\frac{4+2x^{2}}{3}}\\right) a,

    where :math:`a` is the lattice constant and :math:`x=(c/a)/1.633` and 1.633 is the ideal ratio of :math:`c/a`
    in HCP structure.

    Some typical CNP values:

    - FCC : 0.0
    - BCC : 0.0
    - HCP : 4.4
    - FCC (111) surface : 13.0
    - FCC (100) surface : 26.5
    - FCC dislocation core : 11.
    - Isolated atom : 1000. (manually assigned by mdapy)

    Args:
        pos (np.ndarray): (:math:`N_p, 3`) particles positions.
        box (np.ndarray): (:math:`3, 2`) system box.
        boundary (list): boundary conditions, 1 is periodic and 0 is free boundary.
        rc (float): cutoff distance.
        verlet_list (np.ndarray): (:math:`N_p, max\_neigh`) verlet_list[i, j] means j atom is a neighbor of i atom if j > -1.
        distance_list (np.ndarray): (:math:`N_p, max\_neigh`) distance_list[i, j] means distance between i and j atom.
        neighbor_number (np.ndarray): (:math:`N_p`) neighbor atoms number.

    Outputs:
        - **cnp** (np.ndarray) - (:math:`N_p`) CNP results.

    Examples:

        >>> import mdapy as mp

        >>> mp.init()

        >>> FCC = mp.LatticeMaker(3.615, 'FCC', 10, 10, 10) # Create a FCC structure

        >>> FCC.compute() # Get atom positions

        >>> neigh = mp.Neighbor(FCC.pos, FCC.box,
                                3.615, max_neigh=20) # Initialize Neighbor class.

        >>> neigh.compute() # Calculate particle neighbor information.

        >>> CNP = mp.CommonNeighborParameter(FCC.pos, FCC.box, [1, 1, 1], 3.615*0.8536, neigh.verlet_list, neigh.distance_list,
                    neigh.neighbor_number) # Initialize CNP class

        >>> CNP.compute() # Calculate the CNP per atoms

        >>> CNP.cnp[0] # Check results, should be 0.0 here.
    """

    def __init__(
        self, pos, box, boundary, rc, verlet_list, distance_list, neighbor_number
    ) -> None:
        self.pos = pos
        self.box = box
        self.boundary = np.array(boundary)
        self.rc = rc
        self.verlet_list = verlet_list
        self.distance_list = distance_list
        self.neighbor_number = neighbor_number

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
        neighbor_number: ti.types.ndarray(),
        cnp: ti.types.ndarray(),
    ):

        for i in range(pos.shape[0]):
            N = 0
            for m in range(neighbor_number[i]):
                r = ti.Vector([ti.f64(0.0), ti.f64(0.0), ti.f64(0.0)])
                j = verlet_list[i, m]
                if distance_list[i, m] <= self.rc:
                    N += 1
                    for s in range(neighbor_number[j]):
                        for h in range(neighbor_number[i]):
                            if verlet_list[j, s] == verlet_list[i, h]:
                                if (
                                    distance_list[j, s] <= self.rc
                                    and distance_list[i, h] <= self.rc
                                ):
                                    k = verlet_list[j, s]
                                    rik = pos[i] - pos[k]
                                    rjk = pos[j] - pos[k]
                                    rik = self._pbc(rik, box, boundary)
                                    rjk = self._pbc(rjk, box, boundary)
                                    r += rik + rjk
                cnp[i] += r.norm_sqr()
            if N > 0:
                cnp[i] /= N
            else:
                cnp[i] = ti.f64(1000.0)

    def compute(self):
        """Do the real CNP calculation."""
        self.cnp = np.zeros(self.pos.shape[0])
        self._compute(
            self.pos,
            self.box,
            self.boundary,
            self.verlet_list,
            self.distance_list,
            self.neighbor_number,
            self.cnp,
        )


if __name__ == "__main__":
    from lattice_maker import LatticeMaker
    from neighbor import Neighbor
    from time import time

    ti.init(ti.cpu)
    start = time()
    lattice_constant = 4.05
    x, y, z = 50, 50, 50
    FCC = LatticeMaker(lattice_constant, "FCC", x, y, z)
    FCC.compute()
    end = time()
    print(f"Build {FCC.pos.shape[0]} atoms FCC time: {end-start} s.")
    neigh = Neighbor(FCC.pos, FCC.box, 4.05, max_neigh=20)
    neigh.compute()
    print(neigh.neighbor_number.max())
    end = time()
    print(f"Build neighbor time: {end-start} s.")
    start = time()
    CNP = CommonNeighborParameter(
        FCC.pos,
        FCC.box,
        [1, 1, 1],
        4.05 * 0.8536,
        neigh.verlet_list,
        neigh.distance_list,
        neigh.neighbor_number,
    )
    CNP.compute()
    cnp = CNP.cnp
    end = time()
    print(f"Cal cnp time: {end-start} s.")
    print(cnp[:10])
    print(cnp.min(), cnp.max(), cnp.mean())
