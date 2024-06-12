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
class CommonNeighborAnalysis:
    """This class use Common Neighbor Analysis (CNA) method to recgonize the lattice structure, based
    on which atoms can be divided into FCC, BCC, HCP and Other structure.

    .. note:: If one use this module in publication, one should also cite the original paper.
      `Faken D, Jónsson H. Systematic analysis of local atomic structure combined with 3D computer graphics[J].
      Computational Materials Science, 1994, 2(2): 279-286. <https://doi.org/10.1016/0927-0256(94)90109-0>`_.

    .. hint:: We use the `same algorithm as in LAMMPS <https://docs.lammps.org/compute_cna_atom.html>`_.

    CNA method is sensitive to the given cutoff distance. The suggesting cutoff can be obtained from the
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

    The CNA method can recgonize the following structure:

    1. Other
    2. FCC
    3. HCP
    4. BCC
    5. ICO

    Args:
        rc (float): cutoff distance.
        pos (np.ndarray): (:math:`N_p, 3`) particles positions.
        box (np.ndarray): (:math:`4, 3`) system box.
        boundary (list, optional): boundary conditions, 1 is periodic and 0 is free boundary. Defaults to [1, 1, 1].
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

        >>> CNA = mp.CommonNeighborAnalysis(3.615*0.8536, FCC.pos, FCC.box, [1, 1, 1],
                    neigh.verlet_list, neigh.neighbor_number) # Initialize CNA class

        >>> CNA.compute() # Calculate the CNA per atoms

        >>> CNA.pattern # Check results, should be 1 here.

        >>> CNA.structure[CNA.pattern[0]] # Structure of atom 0, should be fcc here.

    """

    def __init__(
        self, rc, pos, box, boundary=[1, 1, 1], verlet_list=None, neighbor_number=None
    ):
        self.rc = rc
        box, inverse_box, rec = init_box(box)
        # Make sure the boxl_length is four times larger than the rc.
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

        self.N = self.pos.shape[0]
        self.verlet_list = verlet_list
        self.neighbor_number = neighbor_number
        self.MAXNEAR = 14
        self.MAXCOMMON = 7

        self.structure = ["other", "fcc", "hcp", "bcc", "ico"]

    @ti.kernel
    def _compute(
        self,
        pos: ti.types.ndarray(dtype=ti.math.vec3),
        box: ti.types.ndarray(element_dim=1),
        verlet_list: ti.types.ndarray(),
        neighbor_number: ti.types.ndarray(),
        cna: ti.types.ndarray(),
        common: ti.types.ndarray(),
        bonds: ti.types.ndarray(),
        pattern: ti.types.ndarray(),
        inverse_box: ti.types.ndarray(element_dim=1)
    ):
        rcsq = self.rc * self.rc
        for i in range(self.N):
            if neighbor_number[i] == 12 or neighbor_number[i] == 14:
                for m in range(neighbor_number[i]):
                    j = verlet_list[i, m]
                    ncommon = 0
                    for inear in range(neighbor_number[i]):
                        for jnear in range(neighbor_number[j]):
                            if verlet_list[i, inear] == verlet_list[j, jnear]:
                                if ncommon < self.MAXCOMMON:
                                    common[i, ncommon] = verlet_list[i, inear]
                                    ncommon += 1

                    cna[i, m, 0] = ncommon

                    for n in range(ncommon):
                        bonds[i, n] = 0

                    nbonds = 0
                    for jj in range(ncommon - 1):
                        j = common[i, jj]
                        for kk in range(jj + 1, ncommon):
                            k = common[i, kk]
                            rjk = pos[j] - pos[k]
                            if ti.static(self.rec):
                                rjk = _pbc_rec(rjk, self.boundary, self.box_length)
                            else:
                                rjk = _pbc(rjk, self.boundary, box, inverse_box)
                            if rjk.norm_sqr() <= rcsq:
                                nbonds += 1
                                bonds[i, jj] += 1
                                bonds[i, kk] += 1

                    cna[i, m, 1] = nbonds
                    maxbonds = 0
                    minbonds = self.MAXCOMMON

                    for n in range(ncommon):
                        maxbonds = ti.max(bonds[i, n], maxbonds)
                        minbonds = ti.min(bonds[i, n], minbonds)

                    cna[i, m, 2] = maxbonds
                    cna[i, m, 3] = minbonds

                nfcc = nhcp = nbcc4 = nbcc6 = nico = 0

                if neighbor_number[i] == 12:
                    for inear in range(12):
                        if (
                            cna[i, inear, 0] == 4
                            and cna[i, inear, 1] == 2
                            and cna[i, inear, 2] == 1
                            and cna[i, inear, 3] == 1
                        ):
                            nfcc += 1
                        if (
                            cna[i, inear, 0] == 4
                            and cna[i, inear, 1] == 2
                            and cna[i, inear, 2] == 2
                            and cna[i, inear, 3] == 0
                        ):
                            nhcp += 1
                        if (
                            cna[i, inear, 0] == 5
                            and cna[i, inear, 1] == 5
                            and cna[i, inear, 2] == 2
                            and cna[i, inear, 3] == 2
                        ):
                            nico += 1
                    if nfcc == 12:
                        pattern[i] = 1
                    elif nfcc == 6 and nhcp == 6:
                        pattern[i] = 2
                    elif nico == 12:
                        pattern[i] = 4
                elif neighbor_number[i] == 14:
                    for inear in range(14):
                        if (
                            cna[i, inear, 0] == 4
                            and cna[i, inear, 1] == 4
                            and cna[i, inear, 2] == 2
                            and cna[i, inear, 3] == 2
                        ):
                            nbcc4 += 1
                        if (
                            cna[i, inear, 0] == 6
                            and cna[i, inear, 1] == 6
                            and cna[i, inear, 2] == 2
                            and cna[i, inear, 3] == 2
                        ):
                            nbcc6 += 1
                    if nbcc4 == 6 and nbcc6 == 8:
                        pattern[i] = 3

    def compute(self):
        """Do the real CNA calculation."""
        if self.verlet_list is None or self.neighbor_number is None:
            neigh = Neighbor(self.pos, self.box, self.rc, self.boundary)
            neigh.compute()
            self.verlet_list, self.neighbor_number = (
                neigh.verlet_list,
                neigh.neighbor_number,
            )

        cna = np.zeros((self.N, self.MAXNEAR, 4), dtype=np.int32)
        common = np.zeros((self.N, self.MAXCOMMON), dtype=np.int32)
        bonds = np.zeros((self.N, self.MAXCOMMON), dtype=np.int32)
        self.pattern = np.zeros(self.N, dtype=np.int32)
        self._compute(
            self.pos,
            self.box,
            self.verlet_list,
            self.neighbor_number,
            cna,
            common,
            bonds,
            self.pattern,
            self.inverse_box
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
    x, y, z = 100, 100, 125
    FCC = LatticeMaker(lattice_constant, "FCC", x, y, z)
    FCC.compute()
    end = time()
    # FCC.write_data()
    print(f"Build {FCC.pos.shape[0]} atoms HCP time: {end-start} s.")
    rc = 4.05 * 0.86  # 1.207
    # start = time()

    # neigh = Neighbor(FCC.pos, FCC.box, rc, max_neigh=20)
    # neigh.compute()
    # # print(neigh.neighbor_number.max())
    # end = time()
    print(f"Build neighbor time: {end-start} s.")
    start = time()
    CNA = CommonNeighborAnalysis(rc, FCC.pos, FCC.box, [1, 1, 1])
    CNA.compute()
    end = time()

    print(f"Cal CNA time: {end-start} s.")
    for i in range(5):
        print(CNA.structure[i], ":", len(CNA.pattern[CNA.pattern == i]))
