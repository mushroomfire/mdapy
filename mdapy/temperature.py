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
class AtomicTemperature:

    """This class is used to calculated an average thermal temperature per atom, wchich is useful at shock
    simulations. The temperature of atom :math:`i` is given by:

    .. math:: T_i=\\sum^{N^i_{neigh}}_0 m^i_j(v_j^i -v_{COM}^i)^2/(3N^i_{neigh}k_B),

    where :math:`N^i_{neigh}` is neighbor atoms number of atom :math:`i`,
    :math:`m^i_j` and :math:`v^i_j` are the atomic mass and velocity of neighbor atom :math:`j` of atom :math:`i`,
    :math:`k_B` is the Boltzmann constant, :math:`v^i_{COM}` is
    the center of mass COM velocity of neighbor of atom :math:`i` and is given by:

    .. math:: v^i_{COM}=\\frac{\\sum _0^{N^i_{neigh}}m^i_jv_j^i}{\\sum_0^{N^i_{neigh}} m_j^i}.

    Here the neighbor of atom :math:`i` includes itself.

    Args:
        amass (np.ndarray): (:math:`N_{type}`) atomic mass.
        vel (np.ndarray): (:math:`N_p, 3`), atomic velocity.
        atype_list (np.ndarray): (:math:`N_p`), atomic type.
        rc (float): cutoff distance to average.
        verlet_list (np.ndarray, optional): (:math:`N_p, max\_neigh`) verlet_list[i, j] means j atom is a neighbor of i atom if j > -1.
        distance_list (np.ndarray, optional): (:math:`N_p, max\_neigh`) distance_list[i, j] means distance between i and j atom.
        neighbor_number (np.ndarray, optional): (:math:`N_p`) neighbor atoms number.
        pos (np.ndarray, optional): (:math:`N_p, 3`) particles positions. Defaults to None.
        box (np.ndarray, optional): (:math:`3, 2`) or (:math:`4, 3`) system box. Defaults to None.
        boundary (list, optional): boundary conditions, 1 is periodic and 0 is free boundary. Such as [1, 1, 1]. Defaults to None.
        units (str, optional): `units <https://docs.lammps.org/units.html>`_ defined in LAMMPS, supports *metal* and *charge*. Defaults to "metal".

    Outputs:
        - **T** (np.ndarray) - (:math:`N_p`), atomic temperature.

    Examples:
        >>> import mdapy as mp

        >>> mp.init()

        >>> FCC = mp.LatticeMaker(3.615, 'FCC', 10, 10, 10) # Create a FCC structure.

        >>> FCC.compute() # Get atom positions.

        >>> neigh = mp.Neighbor(FCC.pos, FCC.box,
                                5., max_neigh=50) # Initialize Neighbor class.

        >>> neigh.compute() # Calculate particle neighbor information.

        >>> def init_vel(N, T, Mass=1.0):
                # Generate random velocity at T K.
                Boltzmann_Constant = 8.617385e-5
                np.random.seed(10086)
                x1 = np.random.random(N * 3)
                x2 = np.random.random(N * 3)
                vel = (
                    np.sqrt(T * Boltzmann_Constant / Mass)
                    * np.sqrt(-2 * np.log(x1))
                    * np.cos(2 * np.pi * x2)
                ).reshape(N, 3)
                vel -= vel.mean(axis=0)
                return vel * 100  # A/ps

        >>> vel = init_vel(FCC.N, 300.0, 1.0) # Generate random velocity at 300 K.

        >>> Temp = AtomicTemperature(
                np.array([1.0]),
                vel,
                np.ones(FCC.N, dtype=int),
                5.0,
                neigh.verlet_list,
                neigh.distance_list,
            ) # Initilize the temperature class.

        >>> Temp.compute() # Do the temperature calculation.

        >>> Temp.T.mean() # Average temperature should be close to 300 K.
    """

    def __init__(
        self,
        amass,
        vel,
        atype_list,
        rc,
        verlet_list=None,
        distance_list=None,
        pos=None,
        box=None,
        boundary=None,
        units="metal",
    ):
        self.amass = amass
        self.atype_list = atype_list
        self.units = units
        if self.units == "metal":
            self.vel = vel * 100.0
        elif self.units == "real":
            self.vel = vel * 100000.0
        self.verlet_list = verlet_list
        self.distance_list = distance_list
        self.rc = rc
        self.old_N = None
        if verlet_list is None or distance_list is None:
            assert pos is not None
            assert box is not None
            assert boundary is not None
            repeat = _check_repeat_cutoff(box, boundary, self.rc)

            if pos.dtype != np.float64:
                pos = pos.astype(np.float64)
            if box.dtype != np.float64:
                box = box.astype(np.float64)
            if sum(repeat) == 3:
                self.pos = pos
                if box.shape == (3, 2):
                    self.box = np.zeros((4, 3), dtype=box.dtype)
                    self.box[0, 0], self.box[1, 1], self.box[2, 2] = (
                        box[:, 1] - box[:, 0]
                    )
                    self.box[-1] = box[:, 0]
                elif box.shape == (4, 3):
                    self.box = box
            else:
                self.old_N = pos.shape[0]
                repli = Replicate(pos, box, *repeat, self.atype_list)
                repli.compute()
                self.pos = repli.pos
                self.box = repli.box
                self.atype_list = repli.type_list
                self.vel = np.vstack([self.vel for _ in range(np.product(repeat))])

            assert self.box[0, 1] == 0
            assert self.box[0, 2] == 0
            assert self.box[1, 2] == 0
            self.boundary = [int(boundary[i]) for i in range(3)]

        self.N = self.vel.shape[0]
        self.T = np.zeros(self.N)

    @ti.kernel
    def _compute(
        self,
        verlet_list: ti.types.ndarray(),
        distance_list: ti.types.ndarray(),
        vel: ti.types.ndarray(),
        amass: ti.types.ndarray(),
        atype_list: ti.types.ndarray(),
        T: ti.types.ndarray(),
    ):
        """
        kb = 8.617333262145e-5 eV / K
        kb = 1.380649e−23 J/K
        dim = 3.
        afu = 6.022140857e23 1/mol
        j2e = 6.24150913e18
        """
        kb = 1.380649e-23
        dim = 3.0
        afu = 6.022140857e23
        max_neigh = verlet_list.shape[1]
        for i in range(self.N):
            # obtain v_COM of neighbor of atom_i
            v_neigh = ti.Vector([ti.float64(0.0)] * 3)
            n_neigh = 0
            mass_neigh = ti.float64(0.0)
            for j_index in range(max_neigh):
                j = verlet_list[i, j_index]
                disj = distance_list[i, j_index]
                if j > -1:
                    if j != i and disj <= self.rc:
                        j_mass = amass[atype_list[j] - 1]
                        v_neigh += ti.Vector([vel[j, 0], vel[j, 1], vel[j, 2]]) * j_mass
                        n_neigh += 1
                        mass_neigh += j_mass
                else:
                    break
            v_neigh += ti.Vector([vel[i, 0], vel[i, 1], vel[i, 2]])
            n_neigh += 1
            mass_neigh += amass[atype_list[i] - 1]
            v_mean = v_neigh / mass_neigh

            # subtract v_COM
            ke_neigh = ti.float64(0.0)
            for j_index in range(max_neigh):
                j = verlet_list[i, j_index]
                disj = distance_list[i, j_index]
                if j > -1:
                    if j != i and disj <= self.rc:
                        v_j = (
                            ti.Vector([vel[j, 0], vel[j, 1], vel[j, 2]]) - v_mean
                        ).norm_sqr()
                        ke_neigh += 0.5 * amass[atype_list[j] - 1] / afu / 1000.0 * v_j
                else:
                    break
            ke_neigh += (
                0.5
                * amass[atype_list[i] - 1]
                / afu
                / 1000.0
                * (ti.Vector([vel[i, 0], vel[i, 1], vel[i, 2]]) - v_mean).norm_sqr()
            )

            # obtain temperature
            T[i] = ke_neigh * 2.0 / dim / n_neigh / kb

    def compute(self):
        """Do the real temperature calculation."""
        if self.verlet_list is None or self.distance_list is None:
            neigh = Neighbor(self.pos, self.box, self.rc, self.boundary)
            neigh.compute()
            self.verlet_list, self.distance_list = (
                neigh.verlet_list,
                neigh.distance_list,
            )
        self._compute(
            self.verlet_list,
            self.distance_list,
            self.vel,
            self.amass,
            self.atype_list,
            self.T,
        )
        if self.old_N is not None:
            self.T = np.ascontiguousarray(self.T[: self.old_N])


if __name__ == "__main__":
    from tool_function import _init_vel
    from lattice_maker import LatticeMaker
    from neighbor import Neighbor
    from time import time

    # ti.init(ti.gpu, device_memory_GB=5.0)
    ti.init(ti.cpu)
    start = time()
    lattice_constant = 3.615
    x, y, z = (10, 10, 10)
    FCC = LatticeMaker(lattice_constant, "FCC", x, y, z)
    FCC.compute()
    end = time()
    print(f"Build {FCC.pos.shape[0]} atoms FCC time: {end-start} s.")
    # start = time()
    # neigh = Neighbor(FCC.pos, FCC.box, 5.0)
    # neigh.compute()
    # end = time()
    # print(f"Build neighbor time: {end-start} s.")

    vel = _init_vel(FCC.N, 300.0, 1.0)
    start = time()
    T = AtomicTemperature(
        np.array([1.0]),
        vel,
        np.ones(FCC.pos.shape[0], dtype=int),
        5.0,
        None,
        None,
        FCC.pos,
        FCC.box,
        [1, 1, 1],
    )

    T.compute()
    end = time()
    print(f"Calculating T time: {end-start} s.")
    print(T.T)
    print("Average temperature is", T.T.mean(), "K.")
