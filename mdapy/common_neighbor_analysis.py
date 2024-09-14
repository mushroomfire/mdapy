# Copyright (c) 2022-2024, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import taichi as ti
import numpy as np

try:
    from tool_function import _check_repeat_cutoff, _check_repeat_nearest
    from replicate import Replicate
    from neighbor import Neighbor
    from nearest_neighbor import NearestNeighbor
    from box import init_box
    from cna import _cna
except Exception:
    from .tool_function import _check_repeat_cutoff, _check_repeat_nearest
    from .replicate import Replicate
    from .neighbor import Neighbor
    from .nearest_neighbor import NearestNeighbor
    from .box import init_box
    import _cna


@ti.data_oriented
class CommonNeighborAnalysis:
    """This class use Common Neighbor Analysis (CNA) method to recgonize the lattice structure, based
    on which atoms can be divided into FCC, BCC, HCP and Other structure.

    .. note:: If one use this module in publication, one should also cite the original paper.
      `Stukowski, A. (2012). Structure identification methods for atomistic simulations of crystalline materials.
      Modelling and Simulation in Materials Science and Engineering, 20(4), 045021. <https://doi.org/10.1088/0965-0393/20/4/045021>`_.

    .. hint:: We use the `same algorithm as in OVITO <https://www.ovito.org/docs/current/reference/pipelines/modifiers/common_neighbor_analysis.html#particles-modifiers-common-neighbor-analysis>`_.

    The original CNA method is sensitive to the given cutoff distance. The suggesting cutoff can be obtained from the
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

    Prof. Alexander Stukowski has improved this method using adaptive cutoff distances based on the atomic neighbor environment, which is the default method
    in mdapy from version 0.11.1.

    The CNA method can recgonize the following structure:

    1. Other
    2. FCC
    3. HCP
    4. BCC
    5. ICO

    Args:

        pos (np.ndarray): (:math:`N_p, 3`) particles positions.
        box (np.ndarray): (:math:`4, 3`) system box.
        boundary (list, optional): boundary conditions, 1 is periodic and 0 is free boundary. Defaults to [1, 1, 1].
        rc (float, optional): cutoff distance, if not given, will using adaptive cutoff.
        verlet_list (np.ndarray, optional): (:math:`N_p, max\_neigh`) verlet_list[i, j] means j atom is a neighbor of i atom if j > -1. Defaults to None.
        neighbor_number (np.ndarray, optional): (:math:`N_p`) neighbor atoms number. Defaults to None.

    Outputs:
        - **pattern** (np.ndarray) - (:math:`N_p`) CNA results.
        - **structure** (list) - the corresponding structure to each pattern.

    Examples:
        >>> import mdapy as mp

        >>> mp.init()

        >>> FCC = mp.LatticeMaker(3.615, 'FCC', 10, 10, 10) # Create a FCC structure

        >>> FCC.compute() # Get atom positions

        >>> neigh = mp.Neighbor(FCC.pos, FCC.box,
                                3.615*0.8536, max_neigh=20) # Initialize Neighbor class.

        >>> neigh.compute() # Calculate particle neighbor information.

        >>> CNA = mp.CommonNeighborAnalysis(FCC.pos, FCC.box, [1, 1, 1], 3.615*0.8536,
                    neigh.verlet_list, neigh.neighbor_number) # Initialize CNA class

        >>> CNA.compute() # Calculate the CNA per atoms

        >>> CNA.pattern # Check results, should be 1 here.

        >>> CNA.structure[CNA.pattern[0]] # Structure of atom 0, should be fcc here.

    """

    def __init__(
        self,
        pos,
        box,
        boundary=[1, 1, 1],
        rc=None,
        verlet_list=None,
        neighbor_number=None,
    ):
        self.rc = rc
        box, inverse_box, rec = init_box(box)
        if rc is None:
            repeat = _check_repeat_nearest(pos, box, boundary)
        else:
            # Make sure the box_length is four times larger than the rc.
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

        self.boundary = boundary
        self.N = self.pos.shape[0]
        self.verlet_list = verlet_list
        self.neighbor_number = neighbor_number
        self.structure = ["other", "fcc", "hcp", "bcc", "ico"]

    def compute(self):
        """Do the real CNA calculation."""
        if self.verlet_list is None or self.neighbor_number is None:
            if self.rc is not None:
                neigh = Neighbor(self.pos, self.box, self.rc, self.boundary)
                neigh.compute()
                self.verlet_list, self.neighbor_number = (
                    neigh.verlet_list,
                    neigh.neighbor_number,
                )
            else:
                if self.N < 14:
                    self.pattern = np.zeros(self.N, dtype=np.int32)
                    return
                else:
                    neigh = NearestNeighbor(self.pos, self.box, self.boundary)
                    _, self.verlet_list = neigh.query_nearest_neighbors(14)

        self.pattern = np.zeros(self.N, dtype=np.int32)
        if self.rc is None:
            _cna._acna(
                self.pos,
                self.box,
                self.inverse_box,
                np.bool_(self.boundary),
                self.verlet_list,
                self.pattern,
            )
        else:
            _cna._fcna(
                self.pos,
                self.box,
                self.inverse_box,
                np.bool_(self.boundary),
                self.verlet_list,
                self.neighbor_number,
                self.pattern,
                self.rc,
            )
        if self.old_N is not None:
            self.pattern = np.ascontiguousarray(self.pattern[: self.old_N])


if __name__ == "__main__":
    from lattice_maker import LatticeMaker
    from neighbor import Neighbor
    from time import time

    # ti.init(ti.gpu, device_memory_GB=5.0)
    ti.init(ti.cpu)
    start = time()
    lattice_constant = 4.05
    x, y, z = 1, 1, 1
    FCC = LatticeMaker(lattice_constant, "HCP", x, y, z)
    FCC.compute()
    end = time()
    FCC.write_data()
    print(f"Build {FCC.pos.shape[0]} atoms HCP time: {end-start} s.")
    rc = 4.05 * 0.86  # 1.207
    # start = time()

    # neigh = Neighbor(FCC.pos, FCC.box, rc, max_neigh=20)
    # neigh.compute()
    # print(neigh.neighbor_number.max())
    # end = time()
    # print(f"Build neighbor time: {end-start} s.")
    start = time()
    CNA = CommonNeighborAnalysis(
        FCC.pos, FCC.box, [1, 1, 1]
    )  # , rc=lattice_constant*1.207)
    CNA.compute()
    end = time()

    print(f"Cal CNA time: {end-start} s.")
    for i in range(5):
        print(CNA.structure[i], ":", len(CNA.pattern[CNA.pattern == i]))
