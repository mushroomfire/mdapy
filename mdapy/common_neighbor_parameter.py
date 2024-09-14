# Copyright (c) 2022-2024, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import taichi as ti
import numpy as np

try:
    from tool_function import _check_repeat_cutoff
    from replicate import Replicate
    from neighbor import Neighbor
    from box import init_box, _pbc, _pbc_rec
except Exception:
    from .tool_function import _check_repeat_cutoff
    from .replicate import Replicate
    from .neighbor import Neighbor
    from .box import init_box, _pbc, _pbc_rec


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
        box (np.ndarray): (:math:`3, 2`) or (:math:`4, 3`) system box.
        boundary (list): boundary conditions, 1 is periodic and 0 is free boundary.
        rc (float): cutoff distance.
        verlet_list (np.ndarray, optional): (:math:`N_p, max\_neigh`) verlet_list[i, j] means j atom is a neighbor of i atom if j > -1.
        distance_list (np.ndarray, optional): (:math:`N_p, max\_neigh`) distance_list[i, j] means distance between i and j atom.
        neighbor_number (np.ndarray, optional): (:math:`N_p`) neighbor atoms number.

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
        self,
        pos,
        box,
        boundary,
        rc,
        verlet_list=None,
        distance_list=None,
        neighbor_number=None,
    ) -> None:
        self.rc = rc
        box, inverse_box, rec = init_box(box)
        repeat = _check_repeat_cutoff(box, boundary, self.rc)
        if pos.dtype != np.float64:
            pos = pos.astype(np.float64)

        self.old_N = None
        if sum(repeat) == 3:
            self.pos = pos
            self.box, self.inverse_box, self.rec = box, inverse_box, rec
        else:
            self.old_N = pos.shape[0]
            repli = Replicate(pos, box, *repeat)
            repli.compute()
            self.pos = repli.pos
            self.box, self.inverse_box, self.rec = init_box(repli.box)

        self.box_length = ti.Vector([np.linalg.norm(self.box[i]) for i in range(3)])
        self.boundary = ti.Vector([int(boundary[i]) for i in range(3)])
        self.verlet_list = verlet_list
        self.distance_list = distance_list
        self.neighbor_number = neighbor_number

    @ti.kernel
    def _compute(
        self,
        pos: ti.types.ndarray(dtype=ti.math.vec3),
        box: ti.types.ndarray(element_dim=1),
        verlet_list: ti.types.ndarray(),
        distance_list: ti.types.ndarray(),
        neighbor_number: ti.types.ndarray(),
        cnp: ti.types.ndarray(),
        inverse_box: ti.types.ndarray(element_dim=1),
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
                                    if ti.static(self.rec):
                                        rik = _pbc_rec(
                                            rik, self.boundary, self.box_length
                                        )
                                        rjk = _pbc_rec(
                                            rjk, self.boundary, self.box_length
                                        )
                                    else:
                                        rik = _pbc(rik, self.boundary, box, inverse_box)
                                        rjk = _pbc(rjk, self.boundary, box, inverse_box)
                                    r += rik + rjk
                cnp[i] += r.norm_sqr()
            if N > 0:
                cnp[i] /= N
            else:
                cnp[i] = ti.f64(1000.0)

    def compute(self):
        """Do the real CNP calculation."""
        if (
            self.verlet_list is None
            or self.neighbor_number is None
            or self.distance_list is None
        ):
            neigh = Neighbor(self.pos, self.box, self.rc, self.boundary)
            neigh.compute()
            self.verlet_list, self.distance_list, self.neighbor_number = (
                neigh.verlet_list,
                neigh.distance_list,
                neigh.neighbor_number,
            )
        self.cnp = np.zeros(self.pos.shape[0])

        self._compute(
            self.pos,
            self.box,
            self.verlet_list,
            self.distance_list,
            self.neighbor_number,
            self.cnp,
            self.inverse_box,
        )
        if self.old_N is not None:
            self.cnp = self.cnp[: self.old_N]


if __name__ == "__main__":
    from lattice_maker import LatticeMaker
    from neighbor import Neighbor
    from time import time

    ti.init(ti.cpu)
    start = time()
    lattice_constant = 4.05
    x, y, z = 1, 1, 1
    FCC = LatticeMaker(lattice_constant, "HCP", x, y, z)
    FCC.compute()
    end = time()
    print(f"Build {FCC.pos.shape[0]} atoms FCC time: {end-start} s.")
    # neigh = Neighbor(FCC.pos, FCC.box, 4.05 * 1.207, max_neigh=20)
    # neigh.compute()
    # end = time()
    # print(f"Build neighbor time: {end-start} s.")
    start = time()
    # 0.8536, 1.207
    CNP = CommonNeighborParameter(
        FCC.pos,
        FCC.box,
        [1, 1, 1],
        4.05 * 1.207,
    )
    CNP.compute()
    cnp = CNP.cnp
    end = time()
    print(f"Cal cnp time: {end-start} s.")
    print(cnp[:10])
    print(cnp.min(), cnp.max(), cnp.mean())
