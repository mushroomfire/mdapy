# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import taichi as ti
import numpy as np

try:
    from nearest_neighbor import NearestNeighbor
    from replicate import Replicate
    from tool_function import _check_repeat_nearest
except Exception:
    from .nearest_neighbor import NearestNeighbor
    from .replicate import Replicate
    from .tool_function import _check_repeat_nearest


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
        repeat = _check_repeat_nearest(pos, box, boundary)

        if pos.dtype != np.float64:
            pos = pos.astype(np.float64)
        if box.dtype != np.float64:
            box = box.astype(np.float64)
        self.old_N = None
        if sum(repeat) == 3:
            self.pos = pos
            if box.shape == (3, 2):
                self.box = np.zeros((4, 3), dtype=box.dtype)
                self.box[0, 0], self.box[1, 1], self.box[2, 2] = box[:, 1] - box[:, 0]
                self.box[-1] = box[:, 0]
            elif box.shape == (4, 3):
                self.box = box
        else:
            self.old_N = pos.shape[0]
            repli = Replicate(pos, box, *repeat)
            repli.compute()
            self.pos = repli.pos
            self.box = repli.box

        assert self.box[0, 1] == 0
        assert self.box[0, 2] == 0
        assert self.box[1, 2] == 0
        self.box_length = ti.Vector([np.linalg.norm(self.box[i]) for i in range(3)])
        self.rec = True
        if self.box[1, 0] != 0 or self.box[2, 0] != 0 or self.box[2, 1] != 0:
            self.rec = False
        self.boundary = ti.Vector([int(boundary[i]) for i in range(3)])
        self.verlet_list = verlet_list
        self.distance_list = distance_list
        self.structure = ["other", "fcc", "hcp", "bcc", "ico"]

    @ti.func
    def _pbc_rec(self, rij):
        for m in ti.static(range(3)):
            if self.boundary[m]:
                dx = rij[m]
                x_size = self.box_length[m]
                h_x_size = x_size * 0.5
                if dx > h_x_size:
                    dx = dx - x_size
                if dx <= -h_x_size:
                    dx = dx + x_size
                rij[m] = dx
        return rij

    @ti.func
    def _pbc(self, rij, box: ti.types.ndarray(element_dim=1)) -> ti.math.vec3:
        nz = rij[2] / box[2][2]
        ny = (rij[1] - nz * box[2][1]) / box[1][1]
        nx = (rij[0] - ny * box[1][0] - nz * box[2][0]) / box[0][0]
        n = ti.Vector([nx, ny, nz])
        for i in ti.static(range(3)):
            if self.boundary[i] == 1:
                if n[i] > 0.5:
                    n[i] -= 1
                elif n[i] < -0.5:
                    n[i] += 1
        return n[0] * box[0] + n[1] * box[1] + n[2] * box[2]

    @ti.kernel
    def _compute(
        self,
        pos: ti.types.ndarray(dtype=ti.math.vec3),
        box: ti.types.ndarray(element_dim=1),
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
                    if ti.static(self.rec):
                        rij = self._pbc_rec(rij)
                        rik = self._pbc_rec(rik)
                    else:
                        rij = self._pbc(rij, box)
                        rik = self._pbc(rik, box)
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

        self.aja = np.zeros(self.pos.shape[0], dtype=int)
        if self.pos.shape[0] < 14 and sum(self.boundary) == 0:
            pass
        elif self.verlet_list is not None and self.distance_list is not None:
            self._compute(
                self.pos, self.box, self.verlet_list, self.distance_list, self.aja
            )
        else:
            kdt = NearestNeighbor(self.pos, self.box, self.boundary)
            distance_list, verlet_list = kdt.query_nearest_neighbors(14)
            self._compute(self.pos, self.box, verlet_list, distance_list, self.aja)
        if self.old_N is not None:
            self.aja = self.aja[: self.old_N]


if __name__ == "__main__":
    from lattice_maker import LatticeMaker
    from time import time
    from neighbor import Neighbor

    # ti.init(ti.gpu, device_memory_GB=5.0)
    ti.init(ti.cpu, offline_cache=True)
    start = time()
    lattice_constant = 4.05
    x, y, z = 10, 10, 10
    FCC = LatticeMaker(lattice_constant, "BCC", x, y, z)
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

    # Neigh = Neighbor(FCC.pos, FCC.box, 4.05, max_neigh=30)
    # Neigh.compute()
    # print(Neigh.neighbor_number.min())

    # start = time()
    # Neigh.sort_verlet_by_distance(14)
    # end = time()
    # print(f"taichi sort time: {end-start} s.")

    # start = time()
    # AJA = AcklandJonesAnalysis(
    #     FCC.pos, FCC.box, [1, 1, 1], Neigh.verlet_list, Neigh.distance_list
    # )
    # AJA.compute()
    # aja = AJA.aja
    # end = time()
    # print(f"Cal aja sort time: {end-start} s.")
    # for i in range(5):
    #     print(AJA.structure[i], sum(aja == i))
