# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import taichi as ti
import numpy as np

try:
    from tool_function import _check_repeat_cutoff
    from replicate import Replicate
    from neighbor import Neighbor
except Exception:
    from .tool_function import _check_repeat_cutoff
    from .replicate import Replicate
    from .neighbor import Neighbor


@ti.data_oriented
class Calculator:
    """This class is used to calculate the atomic energy and force based on the given embedded atom method
    EAM potential. Multi-elements alloy is also supported.

    Args:
        potential (mp.EAM): A EAM class.
        pos (np.ndarray): (:math:`N_p, 3`) particles positions.
        boundary (list): boundary conditions, 1 is periodic and 0 is free boundary. Such as [1, 1, 1].
        box (np.ndarray): (:math:`3, 2`) or (:math:`4, 3`) system box.
        elements_list (list): elements need to be calculated. Such as ['Al', 'Fe'].
        init_type_list (np.ndarray): (:math:`N_p`) per atom type.
        verlet_list (np.ndarray, optional): (:math:`N_p, max\_neigh`) verlet_list[i, j] means j atom is a neighbor of i atom if j > -1. Defaults to None.
        distance_list (np.ndarray, optional): (:math:`N_p, max\_neigh`) distance_list[i, j] means distance between i and j atom. Defaults to None.
        neighbor_number (np.ndarray, optional): (:math:`N_p`) neighbor atoms number. Defaults to None.

    Outputs:
        - **energy** (np.ndarray) - (:math:`N_p`) atomic energy (eV).
        - **force** (np.ndarray) - (:math:`N_p, 3`) atomic force (eV/A).

    Examples:

        >>> import mdapy as mp

        >>> mp.init()

        >>> potential = mp.EAM("./example/CoNiFeAlCu.eam.alloy") # Read a eam.alloy file.

        >>> FCC = mp.LatticeMaker(3.615, 'FCC', 10, 10, 10) # Create a FCC structure

        >>> FCC.compute() # Get atom positions

        >>> neigh = mp.Neighbor(FCC.pos, FCC.box,
                                potential.rc, max_neigh=100) # Initialize Neighbor class.

        >>> neigh.compute() # Calculate particle neighbor information.

        >>> Cal = mp.Calculator(
                potential,
                FCC.pos,
                [1, 1, 1],
                FCC.box,
                ["Al"],
                np.ones(FCC.pos.shape[0], dtype=np.int32),
                neigh.verlet_list,
                neigh.distance_list,
                neigh.neighbor_number,
            ) # Initialize Calculator class.

        >>> Cal.compute() # Calculate the atomic energy and force.

        >>> Cal.energy # Check the energy.

        >>> Cal.force # Check the force.
    """

    def __init__(
        self,
        potential,
        pos,
        boundary,
        box,
        elements_list,
        init_type_list,
        verlet_list=None,
        distance_list=None,
        neighbor_number=None,
    ):
        self.potential = potential
        self.rc = self.potential.rc
        repeat = _check_repeat_cutoff(box, boundary, self.rc, 5)

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
            self.init_type_list = init_type_list
        else:
            self.old_N = pos.shape[0]
            repli = Replicate(pos, box, *repeat, type_list=init_type_list)
            repli.compute()
            self.pos = repli.pos
            self.box = repli.box
            self.init_type_list = repli.type_list

        assert self.box[0, 1] == 0
        assert self.box[0, 2] == 0
        assert self.box[1, 2] == 0
        self.box_length = ti.Vector([np.linalg.norm(self.box[i]) for i in range(3)])
        self.rec = True
        if self.box[1, 0] != 0 or self.box[2, 0] != 0 or self.box[2, 1] != 0:
            self.rec = False
        self.boundary = ti.Vector([int(boundary[i]) for i in range(3)])
        self.elements_list = elements_list
        self.verlet_list = verlet_list
        self.distance_list = distance_list
        self.neighbor_number = neighbor_number

    def _get_type_list(self):
        assert len(self.elements_list) == len(
            np.unique(self.init_type_list)
        ), "All type must be assigned."
        for i in self.elements_list:
            assert (
                i in self.potential.elements_list
            ), f"Input element {i} not included in potential."

        init_to_now = np.array(
            [self.potential.elements_list.index(i) for i in self.elements_list]
        )
        N = self.pos.shape[0]
        type_list = np.zeros(N, dtype=int)

        @ti.kernel
        def _get_type_list_real(
            type_list: ti.types.ndarray(),
            N: int,
            init_to_now: ti.types.ndarray(),
            init_type_list: ti.types.ndarray(),
        ):
            for i in range(N):
                type_list[i] = init_to_now[init_type_list[i] - 1] + 1

        _get_type_list_real(type_list, N, init_to_now, self.init_type_list)
        return type_list

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
    def _compute_energy_force(
        self,
        box: ti.types.ndarray(element_dim=1),
        verlet_list: ti.types.ndarray(),
        distance_list: ti.types.ndarray(),
        neighbor_number: ti.types.ndarray(),
        atype_list: ti.types.ndarray(),
        dr: float,
        drho: float,
        nr: int,
        nrho: int,
        embedded_data: ti.types.ndarray(),
        phi_data: ti.types.ndarray(),
        elec_density_data: ti.types.ndarray(),
        d_embedded_data: ti.types.ndarray(),
        d_phi_data: ti.types.ndarray(),
        d_elec_density_data: ti.types.ndarray(),
        pos: ti.types.ndarray(dtype=ti.math.vec3),
        energy: ti.types.ndarray(),
        force: ti.types.ndarray(dtype=ti.math.vec3),
        elec_density: ti.types.ndarray(),
        d_embedded_rho: ti.types.ndarray(),
    ):
        N = verlet_list.shape[0]

        for i in range(N):
            i_type = atype_list[i] - 1
            for jj in range(neighbor_number[i]):
                j = verlet_list[i, jj]
                if j > i:
                    j_type = atype_list[j] - 1
                    r = distance_list[i, jj]
                    if r <= self.rc:
                        rk = r / dr
                        k = int(rk)
                        quanzhong = rk - k
                        if k > nr - 2:
                            k = nr - 2

                        pair_enegy = (
                            quanzhong * phi_data[i_type, j_type, k + 1]
                            + (1 - quanzhong) * phi_data[i_type, j_type, k]
                        )
                        energy[i] += pair_enegy / 2.0
                        energy[j] += pair_enegy / 2.0
                        elec_density[i] += (
                            quanzhong * elec_density_data[j_type, k + 1]
                            + (1 - quanzhong) * elec_density_data[j_type, k]
                        )
                        elec_density[j] += (
                            quanzhong * elec_density_data[i_type, k + 1]
                            + (1 - quanzhong) * elec_density_data[i_type, k]
                        )

        for i in range(N):
            i_type = atype_list[i] - 1
            rk = elec_density[i] / drho
            k = int(rk)
            quanzhong = rk - k
            if k > nrho - 2:
                k = nrho - 2
            energy[i] += (
                quanzhong * embedded_data[i_type, k + 1]
                + (1 - quanzhong) * embedded_data[i_type, k]
            )
            d_embedded_rho[i] = (
                quanzhong * d_embedded_data[i_type, k + 1]
                + (1 - quanzhong) * d_embedded_data[i_type, k]
            )

        for i in range(N):
            i_type = atype_list[i] - 1
            for jj in range(neighbor_number[i]):
                j = verlet_list[i, jj]
                if j > i:
                    rij = pos[i] - pos[j]
                    if ti.static(self.rec):
                        rij = self._pbc_rec(rij)
                    else:
                        rij = self._pbc(rij, box)
                    j_type = atype_list[j] - 1
                    r = distance_list[i, jj]
                    if r <= self.rc:
                        rk = r / dr
                        k = int(rk)
                        quanzhong = rk - k
                        if k > nr - 2:
                            k = nr - 2

                        d_pair = (
                            quanzhong * d_phi_data[i_type, j_type, k + 1]
                            + (1 - quanzhong) * d_phi_data[i_type, j_type, k]
                        )
                        d_elec_density_i = (
                            quanzhong * d_elec_density_data[j_type, k + 1]
                            + (1 - quanzhong) * d_elec_density_data[j_type, k]
                        )
                        d_elec_density_j = (
                            quanzhong * d_elec_density_data[i_type, k + 1]
                            + (1 - quanzhong) * d_elec_density_data[i_type, k]
                        )
                        d_pair += (
                            d_embedded_rho[i] * d_elec_density_i
                            + d_embedded_rho[j] * d_elec_density_j
                        )

                        force[i] -= d_pair * rij / r
                        force[j] += d_pair * rij / r

    def compute(self):
        """Do the real energy and force calculation."""
        N = self.pos.shape[0]
        self.energy = np.zeros(N)
        self.force = np.zeros((N, 3))
        elec_density = np.zeros(N)
        d_embedded_rho = np.zeros(N)

        type_list = self._get_type_list()
        if (
            self.distance_list is None
            or self.verlet_list is None
            or self.neighbor_number is None
        ):
            neigh = Neighbor(self.pos, self.box, self.rc, self.boundary)
            neigh.compute()
            self.distance_list, self.verlet_list, self.neighbor_number = (
                neigh.distance_list,
                neigh.verlet_list,
                neigh.neighbor_number,
            )

        self._compute_energy_force(
            self.box,
            self.verlet_list,
            self.distance_list,
            self.neighbor_number,
            type_list,
            self.potential.dr,
            self.potential.drho,
            self.potential.nr,
            self.potential.nrho,
            self.potential.embedded_data,
            self.potential.phi_data,
            self.potential.elec_density_data,
            self.potential.d_embedded_data,
            self.potential.d_phi_data,
            self.potential.d_elec_density_data,
            self.pos,
            self.energy,
            self.force,
            elec_density,
            d_embedded_rho,
        )
        if self.old_N is not None:
            self.energy = np.ascontiguousarray(self.energy[: self.old_N])
            self.force = np.ascontiguousarray(self.force[: self.old_N])


if __name__ == "__main__":
    from lattice_maker import LatticeMaker
    from neighbor import Neighbor
    from potential import EAM
    from time import time

    # ti.init(ti.gpu, device_memory_GB=5.0)
    ti.init(ti.cpu)
    start = time()
    lattice_constant = 4.05
    x, y, z = 1, 1, 1
    FCC = LatticeMaker(lattice_constant, "FCC", x, y, z)
    FCC.compute()
    end = time()
    print(f"Build {FCC.pos.shape[0]} atoms FCC time: {end-start} s.")

    potential = EAM("./example/Al_DFT.eam.alloy")
    print(4.05 * 8 / potential.rc)
    # start = time()
    # neigh = Neighbor(FCC.pos, FCC.box, potential.rc, max_neigh=100)
    # neigh.compute()
    # end = time()
    # print(f"Build neighbor time: {end-start} s.")

    start = time()
    Cal = Calculator(
        potential,
        FCC.pos * 0.9,
        [1, 1, 1],
        FCC.box * 0.9,
        ["Al"],
        np.ones(FCC.pos.shape[0], dtype=np.int32),
    )
    # neigh.verlet_list,
    # neigh.distance_list,
    # neigh.neighbor_number,
    Cal.compute()
    end = time()
    print(f"Calculate energy and force time: {end-start} s.")
    print("energy:")
    print(Cal.energy[:4])
    print("force:")
    print(Cal.force[:4, :])
