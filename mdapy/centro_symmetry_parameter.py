# Copyright (c) 2022-2024, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import taichi as ti
import numpy as np

try:
    from nearest_neighbor import NearestNeighbor
    from replicate import Replicate
    from tool_function import _check_repeat_nearest
    from box import init_box, _pbc, _pbc_rec
except Exception:
    from .nearest_neighbor import NearestNeighbor
    from .replicate import Replicate
    from .tool_function import _check_repeat_nearest
    from .box import init_box, _pbc, _pbc_rec


@ti.data_oriented
class CentroSymmetryParameter:
    """This class is used to compute the CentroSymmetry Parameter (CSP),
    which is heluful to recgonize the structure in lattice, such as FCC and BCC.
    The  CSP is given by:

    .. math::

        p_{\mathrm{CSP}} = \sum_{i=1}^{N/2}{|\mathbf{r}_i + \mathbf{r}_{i+N/2}|^2},

    where :math:`r_i` and :math:`r_{i+N/2}` are two neighbor vectors from the central atom to a pair of opposite neighbor atoms.
    For ideal centrosymmetric crystal, the contributions of all neighbor pairs will be zero. Atomic sites within a defective
    crystal region, in contrast, typically have a positive CSP value.

    This parameter :math:`N` indicates the number of nearest neighbors that should be taken into account when computing
    the centrosymmetry value for an atom. Generally, it should be a positive, even integer. Note that larger number decreases the
    calculation speed. For FCC is 12 and BCC is 8.

    .. note:: If you use this module in publication, you should also cite the original paper.
      `Kelchner C L, Plimpton S J, Hamilton J C. Dislocation nucleation and defect
      structure during surface indentation[J]. Physical review B, 1998, 58(17): 11085. <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.58.11085>`_.

    .. hint:: The CSP is calculated by the `same algorithm as LAMMPS <https://docs.lammps.org/compute_centro_atom.html>`_.
      First calculate all :math:`N (N - 1) / 2` pairs of neighbor atoms, and the summation of the :math:`N/2` lowest weights
      is CSP values.

    Args:
        N (int): Neighbor number.
        pos (np.ndarray): (:math:`N_p, 3`) particles positions.
        box (np.ndarray): (:math:`4, 3`) system box.
        boundary (list, optional): boundary conditions, 1 is periodic and 0 is free boundary. Defaults to [1, 1, 1].
        verlet_list (np.ndarray, optional): (:math:`N_p`, >=N), first N neighbors is sorted, if not given, use kdtree to obtain it. Defaults to None.

    Outputs:
        - **csp** (np.ndarray) - (:math:`N_p`) CSP value per atoms.

    Examples:

        >>> import mdapy as mp

        >>> mp.init()

        >>> FCC = mp.LatticeMaker(3.615, 'FCC', 10, 10, 10) # Create a FCC structure

        >>> FCC.compute() # Get atom positions

        >>> CSP = mp.CentroSymmetryParameter(12, FCC.pos, FCC.box, [1, 1, 1]) # Initialize CSP class

        >>> CSP.compute() # Calculate the csp per atoms

        >>> CSP.csp # Check the csp value
    """

    def __init__(self, N, pos, box, boundary=[1, 1, 1], verlet_list=None):
        self.N = N
        assert N > 0 and N % 2 == 0, "N must be a positive even number."
        box, inverse_box, rec = init_box(box)
        repeat = _check_repeat_nearest(pos, box, boundary)
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

    @ti.kernel
    def _get_csp(
        self,
        pair: ti.types.ndarray(),
        pos: ti.types.ndarray(dtype=ti.math.vec3),
        box: ti.types.ndarray(element_dim=1),
        verlet_list: ti.types.ndarray(),
        loop_index: ti.types.ndarray(),
        csp: ti.types.ndarray(),
        inverse_box: ti.types.ndarray(element_dim=1),
    ):
        # Get loop index
        num = 0
        ti.loop_config(serialize=True)
        for i in range(self.N):
            for j in range(i + 1, self.N):
                loop_index[num, 0] = i
                loop_index[num, 1] = j
                num += 1

        for i, index in ti.ndrange(pair.shape[0], pair.shape[1]):
            j = loop_index[index, 0]
            k = loop_index[index, 1]
            rij = pos[verlet_list[i, j]] - pos[i]
            rik = pos[verlet_list[i, k]] - pos[i]
            if ti.static(self.rec):
                rij = _pbc_rec(rij, self.boundary, self.box_length)
                rik = _pbc_rec(rik, self.boundary, self.box_length)
            else:
                rij = _pbc(rij, self.boundary, box, inverse_box)
                rik = _pbc(rik, self.boundary, box, inverse_box)
            pair[i, index] = (rij + rik).norm_sqr()

        # Select sort
        for i in range(pair.shape[0]):
            res = ti.f64(0.0)
            for j in range(int(self.N / 2)):
                minIndex = j
                for k in range(j + 1, pair.shape[1]):
                    if pair[i, k] < pair[i, minIndex]:
                        minIndex = k
                if minIndex != j:
                    pair[i, minIndex], pair[i, j] = pair[i, j], pair[i, minIndex]
                res += pair[i, j]
            csp[i] = res

    def compute(self):
        """Do the real CSP calculation."""
        self.csp = np.zeros(self.pos.shape[0])
        if self.pos.shape[0] < self.N and sum(self.boundary) == 0:
            self.csp += 10000
        else:
            verlet_list = self.verlet_list
            if verlet_list is None:
                kdt = NearestNeighbor(self.pos, self.box, self.boundary)
                _, verlet_list = kdt.query_nearest_neighbors(self.N)
            loop_index = np.zeros((int(self.N * (self.N - 1) / 2), 2), dtype=int)
            pair = np.zeros((self.pos.shape[0], int(self.N * (self.N - 1) / 2)))
            self._get_csp(
                pair,
                self.pos,
                self.box,
                verlet_list,
                loop_index,
                self.csp,
                self.inverse_box,
            )
        if self.old_N is not None:
            self.csp = np.ascontiguousarray(self.csp[: self.old_N])


if __name__ == "__main__":
    from lattice_maker import LatticeMaker

    # from neighbor import Neighbor
    from time import time

    # ti.init(ti.gpu, device_memory_GB=5.0)
    ti.init(ti.cpu)
    start = time()
    lattice_constant = 4.05
    x, y, z = 100, 100, 100
    FCC = LatticeMaker(lattice_constant, "BCC", x, y, z)
    FCC.compute()
    end = time()
    print(f"Build {FCC.pos.shape[0]} atoms BCC time: {end-start} s.")

    # Neigh = Neighbor(FCC.pos, FCC.box, 4.05, max_neigh=30)
    # Neigh.compute()
    # print(Neigh.neighbor_number.min())

    # start = time()
    # verlet_list_sort = np.ascontiguousarray(np.take_along_axis(Neigh.verlet_list, np.argpartition(Neigh.distance_list, 12, axis=-1), axis=-1)[:, :12])
    # end = time()
    # print(f'numpy sort time: {end-start} s.')
    # print(verlet_list_sort[0])

    # start = time()
    # Neigh.sort_verlet_by_distance(12)
    # end = time()
    # print(f"taichi sort time: {end-start} s.")
    # print(Neigh.verlet_list[0, :12])
    # print(Neigh.distance_list[0, :12])

    # start = time()
    # kdt = kdtree(FCC.pos, FCC.box, [1, 1, 1])
    # _, verlet_list_kdt = kdt.query_nearest_neighbors(12)
    # end = time()
    # print(f'kdt time: {end-start} s.')
    # print(verlet_list_kdt[0])

    start = time()
    CSP = CentroSymmetryParameter(8, FCC.pos, FCC.box, [1, 1, 1])
    CSP.compute()
    csp = CSP.csp
    end = time()
    print(f"Cal csp kdt time: {end-start} s.")
    print(csp)
    print(csp.min(), csp.max(), csp.mean())

    # start = time()
    # CSP = CentroSymmetryParameter(
    #     12, FCC.pos, FCC.box, [1, 1, 1], verlet_list=Neigh.verlet_list
    # )
    # CSP.compute()
    # csp = CSP.csp
    # end = time()
    # print(f"Cal csp verlet time: {end-start} s.")
    # print(csp[:10])
    # print(csp.min(), csp.max(), csp.mean())
