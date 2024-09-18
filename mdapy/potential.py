# Copyright (c) 2022-2024, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import taichi as ti
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

try:
    from plotset import set_figure
    from tool_function import _check_repeat_cutoff, atomic_masses, atomic_numbers
    from replicate import Replicate
    from neighbor import Neighbor
    from nep._nep import NEPCalculator
    from load_save_data import BuildSystem
    from box import init_box, _pbc, _pbc_rec
except Exception:
    from .plotset import set_figure
    from .tool_function import _check_repeat_cutoff, atomic_masses, atomic_numbers
    from .replicate import Replicate
    from .neighbor import Neighbor
    from _nep import NEPCalculator
    from .load_save_data import BuildSystem
    from .box import init_box, _pbc, _pbc_rec

from abc import ABC, abstractmethod


class BasePotential(ABC):
    @abstractmethod
    def compute(self, pos, box, elements_list, type_list, boundary=[1, 1, 1]):
        """Interface function.

        Args:
            pos (np.ndarray): (:math:`N_p, 3`) particles positions.
            box (np.ndarray): (:math:`4, 3`) system box.
            elements_list (list[str]): elements to be calculated, such as ['Al', 'Ni'].
            type_list (np.ndarray): (:math:`N_p`) atom type list.
            boundary (list, optional): boundary conditions, 1 is periodic and 0 is free boundary. Defaults to [1, 1, 1].

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: energy, force, virial.
        """
        pass


@ti.data_oriented
class EAMCalculator:
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
        - **virial** (np.ndarray) - (:math:`N_p, 9`) atomic force (eV*A^3).

    Examples:

        >>> import mdapy as mp

        >>> mp.init()

        >>> potential = mp.EAM("./example/CoNiFeAlCu.eam.alloy") # Read a eam.alloy file.

        >>> FCC = mp.LatticeMaker(3.615, 'FCC', 10, 10, 10) # Create a FCC structure

        >>> FCC.compute() # Get atom positions

        >>> neigh = mp.Neighbor(FCC.pos, FCC.box,
                                potential.rc, max_neigh=100) # Initialize Neighbor class.

        >>> neigh.compute() # Calculate particle neighbor information.

        >>> Cal = EAMCalculator(
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

        >>> Cal.virial # Check the virial.
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
        box, inverse_box, rec = init_box(box)
        repeat = _check_repeat_cutoff(box, boundary, self.rc, 5)

        if pos.dtype != np.float64:
            pos = pos.astype(np.float64)
        self.old_N = None
        if sum(repeat) == 3:
            self.pos = pos
            self.box, self.inverse_box, self.rec = box, inverse_box, rec
            self.init_type_list = init_type_list
        else:
            self.old_N = pos.shape[0]
            repli = Replicate(pos, box, *repeat, type_list=init_type_list)
            repli.compute()
            self.pos = repli.pos
            self.box, self.inverse_box, self.rec = init_box(repli.box)
            self.init_type_list = repli.type_list

        self.box_length = ti.Vector([np.linalg.norm(self.box[i]) for i in range(3)])
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
        virial: ti.types.ndarray(),
        elec_density: ti.types.ndarray(),
        d_embedded_rho: ti.types.ndarray(),
        inverse_box: ti.types.ndarray(element_dim=1),
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
                        rij = _pbc_rec(rij, self.boundary, self.box_length)
                    else:
                        rij = _pbc(rij, self.boundary, box, inverse_box)
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

                        virial[i, 0] += rij[0] * d_pair * rij[0] / r  # xx
                        virial[j, 0] += rij[0] * d_pair * rij[0] / r  # xx

                        virial[i, 1] += rij[1] * d_pair * rij[1] / r  # yy
                        virial[j, 1] += rij[1] * d_pair * rij[1] / r  # yy

                        virial[i, 2] += rij[2] * d_pair * rij[2] / r  # zz
                        virial[j, 2] += rij[2] * d_pair * rij[2] / r  # zz

                        virial[i, 3] += rij[0] * d_pair * rij[1] / r  # xy
                        virial[j, 3] += rij[0] * d_pair * rij[1] / r  # xy

                        virial[i, 4] += rij[0] * d_pair * rij[2] / r  # xz
                        virial[j, 4] += rij[0] * d_pair * rij[2] / r  # xz

                        virial[i, 5] += rij[1] * d_pair * rij[2] / r  # yz
                        virial[j, 5] += rij[1] * d_pair * rij[2] / r  # yz

                        virial[i, 6] += rij[1] * d_pair * rij[0] / r  # yx
                        virial[j, 6] += rij[1] * d_pair * rij[0] / r  # yx

                        virial[i, 7] += rij[2] * d_pair * rij[0] / r  # zx
                        virial[j, 7] += rij[2] * d_pair * rij[0] / r  # zx

                        virial[i, 8] += rij[2] * d_pair * rij[1] / r  # zy
                        virial[j, 8] += rij[2] * d_pair * rij[1] / r  # zy

    def compute(self):
        """Do the real energy, force and virial calculation."""
        N = self.pos.shape[0]
        self.energy = np.zeros(N)
        self.force = np.zeros((N, 3))
        self.virial = np.zeros((N, 9))
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
            self.virial,
            elec_density,
            d_embedded_rho,
            self.inverse_box,
        )
        self.virial /= -2.0
        if self.old_N is not None:
            self.energy = np.ascontiguousarray(self.energy[: self.old_N])
            self.force = np.ascontiguousarray(self.force[: self.old_N])
            self.virial = np.ascontiguousarray(self.virial[: self.old_N])


class EAM(BasePotential):
    """This class is used to read/write a embedded-atom method (EAM) potentials.
    The energy of atom :math:`i` is given by:

    .. math:: E_i = F_\\alpha \\left(\\sum_{j \\neq i}\\rho_\\beta (r_{ij})\\right) + \\frac{1}{2} \\sum_{j \\neq i} \\phi_{\\alpha\\beta} (r_{ij}),

    where :math:`F` is the embedding energy, :math:`\\rho` is the electron density, :math:`\\phi` is the pair interaction, and :math:`\\alpha` and
    :math:`\\beta` are the elements species of atom :math:`i` and :math:`j`. Both summation go over all neighbor atom :math:`j` of atom
    :math:`i` withing the cutoff distance.

    .. note:: Only support `eam.alloy <https://docs.lammps.org/pair_eam.html>`_ definded in LAMMPS format now.

    Args:
        filename (str): filename of eam.alloy file.

    Outputs:
        - **Nelements** (int) - number of elements.
        - **elements_list** (list) - elements list.
        - **nrho** (int) - number of :math:`\\rho` array.
        - **nr** (int) - number of :math:`r` array.
        - **drho** (float) - spacing of electron density :math:`\\rho` space.
        - **dr** (float) - spacing of real distance :math:`r` space. Unit is Angstroms.
        - **rc** (float) - cutoff distance. Unit is Angstroms.
        - **r** (np.ndarray) - (nr), distance space.
        - **rho** (np.ndarray) - (nrho), electron density space.
        - **aindex** (np.ndarray) - (Nelements), element serial number.
        - **amass** (np.ndarray) - (Nelements), element mass.
        - **lattice_constant** (np.ndarray) - (Nelements), lattice constant. Unit is Angstroms.
        - **lattice_type** (list) - (Nelements), lattice type, such as [fcc, bcc].
        - **embedded_data** (np.ndarray) - (Nelements, nrho), embedded energy :math:`F`.
        - **elec_density_data** (np.ndarray) - (Nelements, nr), electron density :math:`\\rho`.
        - **rphi_data** (np.ndarray) = (Nelements, Nelements, nr), :math:`r*\\phi`.
        - **d_embedded_data** (np.ndarray) - (Nelements, nrho), derivatives of embedded energy :math:`dF/d\\rho`.
        - **d_elec_density_data** (np.ndarray) - (Nelements, nr), derivatives of electron density :math:`d\\rho/dr`.
        - **phi_data** (np.ndarray) = (Nelements, Nelements, nr), pair interaction :math:`\\phi`.
        - **d_phi_data** (np.ndarray) = (Nelements, Nelements, nr), derivatives of pair interaction :math:`d\\phi/dr`.


    Examples:
        >>> import mdapy as mp

        >>> mp.init()

        >>> potential = mp.EAM("./example/CoNiFeAlCu.eam.alloy") # Read a eam.alloy file.

        >>> potential.embedded_data # Check embedded energy.

        >>> potential.phi_data # Check pair interaction.

        >>> potential.plot() # Plot information of potential.

        >>> FCC = LatticeMaker(4.05, "FCC", x, y, z)

        >>> FCC.compute()

        >>> Compute energy, force and virial.

        >>> energy, force, virial = potential.compute(FCC.pos, FCC.box, ["Al"], np.ones(FCC.pos.shape[0], dtype=np.int32))

    """

    def __init__(self, filename):
        self.filename = filename
        self._read_eam_alloy()

    def _read_eam_alloy(self):
        file = open(self.filename).readlines()
        self.header = file[:3]
        self.data = []
        for i in file[3:]:
            self.data.extend(i.split())

        self.Nelements = int(self.data[0])
        self.elements_list = self.data[1 : 1 + self.Nelements]

        self.nrho = int(self.data[1 + self.Nelements])
        self.drho = float(self.data[1 + self.Nelements + 1])
        self.nr = int(self.data[1 + self.Nelements + 2])
        self.dr = float(self.data[1 + self.Nelements + 3])

        self.rc = float(self.data[1 + self.Nelements + 4])

        self.embedded_data = np.zeros((self.Nelements, self.nrho))
        self.elec_density_data = np.zeros((self.Nelements, self.nr))
        self.aindex = np.zeros(self.Nelements, dtype=int)
        self.amass = np.zeros(self.Nelements)
        self.lattice_constant = np.zeros(self.Nelements)
        self.lattice_type = []
        self.r = np.arange(0, self.nr) * self.dr
        self.rho = np.arange(0, self.nrho) * self.drho

        start = 1 + self.Nelements + 4 + 1

        for element in range(self.Nelements):
            self.aindex[element] = int(self.data[start])
            self.amass[element] = float(self.data[start + 1])
            self.lattice_constant[element] = float(self.data[start + 2])
            self.lattice_type.append(self.data[start + 3])
            start += 4

            self.embedded_data[element] = np.array(
                self.data[start : start + self.nrho], dtype=float
            )
            start += self.nrho

            self.elec_density_data[element] = np.array(
                self.data[start : start + self.nr], dtype=float
            )
            start += self.nr

        self.rphi_data = np.zeros((self.Nelements, self.Nelements, self.nr))

        for element_i in range(self.Nelements):
            for element_j in range(self.Nelements):
                if element_i >= element_j:
                    self.rphi_data[element_i, element_j] = np.array(
                        self.data[start : start + self.nr], dtype=float
                    )
                    start += self.nr
                    if element_i != element_j:
                        self.rphi_data[element_j, element_i] = self.rphi_data[
                            element_i, element_j
                        ]

        self.d_embedded_data = np.zeros((self.Nelements, self.nrho))
        self.d_elec_density_data = np.zeros((self.Nelements, self.nr))

        for i in range(self.Nelements):
            self.d_embedded_data[i] = spline(
                self.rho, self.embedded_data[i]
            ).derivative(n=1)(self.rho)

            self.d_elec_density_data[i] = spline(
                self.r, self.elec_density_data[i]
            ).derivative(n=1)(self.r)

        self.phi_data = np.zeros((self.Nelements, self.Nelements, self.nr))
        self.d_phi_data = np.zeros((self.Nelements, self.Nelements, self.nr))

        for i in range(self.Nelements):
            for j in range(self.Nelements):
                if i >= j:
                    self.phi_data[i, j, 1:] = self.rphi_data[i, j][1:] / self.r[1:]
                    self.d_phi_data[i, j, 1:] = spline(
                        self.r[1:], self.phi_data[i, j][1:]
                    ).derivative(n=1)(self.r[1:])
                    if i != j:
                        self.phi_data[j, i] = self.phi_data[i, j]
                        self.d_phi_data[j, i] = self.d_phi_data[i, j]
        self.phi_data[:, :, 0] = self.phi_data[:, :, 1]
        self.d_phi_data[:, :, 0] = self.d_phi_data[:, :, 1]

    def write_eam_alloy(self, output_name=None):
        """Save to a eam.alloy file.

        Args:
            output_name (str, optional): filename of generated eam file.
        """
        if output_name is None:
            output_name = ""
            for i in self.elements_list:
                output_name += i
            output_name += ".new.eam.alloy"
        with open(output_name, "w") as op:
            op.write("".join(self.header))

            op.write(f"    {self.Nelements} ")
            for i in self.elements_list:
                op.write(f"{i} ")
            op.write("\n")
            op.write(f" {self.nrho} {self.drho} {self.nr} {self.dr} {self.rc}\n")
            num = 1
            colnum = 5
            for i in range(self.Nelements):
                op.write(
                    f" {int(self.aindex[i])} {self.amass[i]} {self.lattice_constant[i]:.4f} {self.lattice_type[i]}\n "
                )
                for j in range(self.nrho):
                    op.write(f"{self.embedded_data[i, j]:.16E} ")
                    if num > colnum - 1:
                        op.write("\n ")
                        num = 0
                    num += 1
                for j in range(self.nr):
                    op.write(f"{self.elec_density_data[i, j]:.16E} ")
                    if num > colnum - 1:
                        op.write("\n ")
                        num = 0
                    num += 1
            for i1 in range(self.Nelements):
                for i2 in range(i1 + 1):
                    for j in range(self.nr):
                        op.write(f"{self.rphi_data[i1, i2, j]:.16E} ")
                        if num > colnum - 1:
                            op.write("\n ")
                            num = 0
                        num += 1

    def plot_rho_r(self):
        """Plot the :math:`\\rho` with :math:`r`.

        Returns:
            tuple: (fig, ax) matplotlib figure and axis class.
        """
        fig, ax = set_figure(
            figsize=(10, 7),
            bottom=0.18,
            top=0.98,
            left=0.2,
            right=0.98,
            use_pltset=True,
        )
        for i in range(self.Nelements):
            plt.plot(self.r, self.elec_density_data[i], label=self.elements_list[i])
        plt.legend()
        plt.xlim(0, self.rc)
        plt.xlabel("r ($\mathregular{\AA}$)")
        plt.ylabel(r"$\mathregular{\rho}$ (r)")
        plt.show()
        return fig, ax

    def plot_embded_rho(self):
        """Plot the :math:`F` with :math:`\\rho`.

        Returns:
            tuple: (fig, ax) matplotlib figure and axis class.
        """
        fig, ax = set_figure(
            figsize=(10, 7),
            bottom=0.18,
            top=0.98,
            left=0.2,
            right=0.98,
            use_pltset=True,
        )

        for i in range(self.Nelements):
            plt.plot(self.rho, self.embedded_data[i], label=self.elements_list[i])

        plt.legend()
        plt.xlim(0, self.rho[-1])
        plt.xlabel(r"$\mathregular{\rho}$")
        plt.ylabel(r"F($\mathregular{\rho}$) (eV)")

        plt.show()
        return fig, ax

    def plot_phi_r(self):
        """Plot the :math:`\\phi` with :math:`r`.

        Returns:
            tuple: (fig, ax) matplotlib figure and axis class.
        """
        fig, ax = set_figure(
            figsize=(10, 7),
            bottom=0.18,
            top=0.97,
            left=0.2,
            right=0.98,
            use_pltset=True,
        )

        for i in range(self.Nelements):
            for j in range(self.Nelements):
                if i == j:
                    plt.plot(
                        self.r,
                        self.phi_data[i, j],
                        label=f"{self.elements_list[i]}-{self.elements_list[j]}",
                    )

        plt.legend()
        plt.xlim(0, self.rc)
        plt.ylim(-50, 400)
        plt.xlabel("r ($\mathregular{\AA}$)")
        plt.ylabel(r"$\mathregular{\phi}$(r) (eV)")

        plt.show()
        return fig, ax

    def plot(self):
        """Plot :math:`F`, :math:`\\rho`, :math:`\\phi`."""
        self.plot_rho_r()
        self.plot_embded_rho()
        self.plot_phi_r()

    def compute(
        self,
        pos,
        box,
        elements_list,
        type_list,
        boundary=[1, 1, 1],
        verlet_list=None,
        distance_list=None,
        neighbor_number=None,
    ):
        """This function is used to calculate the energy, force and virial.

        Args:
            pos (np.ndarray): (:math:`N_p, 3`) particles positions.
            box (np.ndarray): (:math:`4, 3`) system box.
            elements_list (list[str]): elements to be calculated, such as ['Al', 'Ni'].
            type_list (np.ndarray): (:math:`N_p`) atom type list.
            boundary (list, optional): boundary conditions, 1 is periodic and 0 is free boundary. Defaults to [1, 1, 1].
            verlet_list (np.ndarray, optional): (:math:`N_p, max\_neigh`) verlet_list[i, j] means j atom is a neighbor of i atom if j > -1. Defaults to None.
            distance_list (np.ndarray, optional): (:math:`N_p, max\_neigh`) distance_list[i, j] means distance between i and j atom. Defaults to None.
            neighbor_number (np.ndarray, optional): (:math:`N_p`) neighbor atoms number. Defaults to None.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: energy, force, virial.

        Units & Shape:
            energy : eV (:math:`N_p`)
            force : eV/A (:math:`N_p, 3`). The order is fx, fy and fz.
            virial : eV*A^3 (:math:`N_p, 9`). The order is xx, yy, zz, xy, xz, yz, yx, zx, zy.
        """
        Cal = EAMCalculator(
            self,
            pos,
            boundary,
            box,
            elements_list,
            type_list,
            verlet_list,
            distance_list,
            neighbor_number,
        )
        Cal.compute()
        return Cal.energy, Cal.force, Cal.virial


class NEP(BasePotential):
    """This class is a python interface for `NEP_CPU <https://github.com/brucefan1983/NEP_CPU>`_ version,
    which can be used to evaluate the energy, force and virial of a given system.

    Args:
        filename (str): filename of a NEP potential file, such as nep.txt.

    Outputs:
        - **rc** (float) - cutoff distance. Unit is Angstroms.
        - **info** (dict) - information for NEP potential.

    Example:

        >>> import mdapy as mp

        >>> mp.init()

        >>> FCC = LatticeMaker(4.05, "FCC", x, y, z)

        >>> FCC.compute()

        >>> nep = NEP("nep.txt")

        >>> energy, force, virial = nep.compute(
                FCC.pos, FCC.box, ["Al"], np.ones(FCC.pos.shape[0], dtype=np.int32)
                )

        >>> des = nep.get_descriptors(
                    FCC.pos, FCC.box, ["Al"], np.ones(FCC.pos.shape[0], dtype=np.int32)
                    ) # obtain the descriptor.
    """

    def __init__(self, filename) -> None:
        self.filename = filename
        self._nep = NEPCalculator(filename)
        self.info = self._nep.info
        self.rc = max(self.info["radial_cutoff"], self.info["angular_cutoff"])

    def compute(self, pos, box, elements_list, type_list, boundary=[1, 1, 1]):
        """This function is used to calculate the energy, force and virial.

        Args:
            pos (np.ndarray): (:math:`N_p, 3`) particles positions.
            box (np.ndarray): (:math:`4, 3`) system box.
            elements_list (list[str]): elements to be calculated, such as ['Al', 'Ni'].
            type_list (np.ndarray): (:math:`N_p`) atom type list.
            boundary (list, optional): boundary conditions, 1 is periodic and 0 is free boundary. Defaults to [1, 1, 1].

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: energy, force, virial.

        Units & Shape:
            energy : eV (:math:`N_p`)
            force : eV/A (:math:`N_p, 3`). The order is fx, fy and fz.
            virial : eV*A^3 (:math:`N_p, 9`). The order is xx, yy, zz, xy, xz, yz, yx, zx, zy.
        """
        box, _, _ = init_box(box)
        for i in elements_list:
            assert (
                i in self.info["element_list"]
            ), f"{i} not contained in {self.info['element_list']}."

        type_list = np.array(
            [self.info["element_list"].index(elements_list[i - 1]) for i in type_list],
            int,
        )
        box = np.array(box[:-1])
        for i, j in enumerate(boundary):
            if j == 0:
                box[i, i] += 1.5 * self.rc
        box = box.T.flatten()
        pos = pos.T.flatten()  # make sure right order.
        # print(type_list, box, pos)
        e, f, v = self._nep.calculate(type_list, box, pos)

        e = np.array(e)
        f = np.array(f).reshape(3, -1).T
        v = np.array(v).reshape(9, -1).T
        v = v[:, [0, 4, 8, 1, 2, 5, 3, 6, 7]]

        return e, f, v

    def get_descriptors(self, pos, box, elements_list, type_list, boundary=[1, 1, 1]):
        """This function is used to calculate the descriptor.

        Args:
            pos (np.ndarray): (:math:`N_p, 3`) particles positions.
            box (np.ndarray): (:math:`4, 3`) system box.
            elements_list (list[str]): elements to be calculated, such as ['Al', 'Ni'].
            type_list (np.ndarray): (:math:`N_p`) atom type list.
            boundary (list, optional): boundary conditions, 1 is periodic and 0 is free boundary. Defaults to [1, 1, 1].

        Returns:
            np.ndarray: descriptor.
        """

        box, _, _ = init_box(box)

        for i in elements_list:
            assert (
                i in self.info["element_list"]
            ), f"{i} not contained in {self.info['element_list']}."

        type_list = np.array(
            [self.info["element_list"].index(elements_list[i - 1]) for i in type_list],
            int,
        )
        box = np.array(box[:-1])
        for i, j in enumerate(boundary):
            if j == 0:
                box[i, i] += 1.5 * self.rc
        box = box.T.flatten()
        pos = pos.T.flatten()

        return (
            np.array(self._nep.get_descriptors(type_list, box, pos))
            .reshape(-1, len(type_list))
            .T
        )

    def fps_sample(
        self,
        n_sample,
        des_total=None,
        filename_list=None,
        elements_list=None,
        start_idx=None,
        fmt=None,
    ):
        """This function is used to sample the configurations using farthest point sampling method, based
        on the NEP descriptors. It is helpful to select the structures during active learning process.

        Args:
            n_sample (int): number of structures one wants to select.
            des_total (np.ndarray): two dimensional ndarray, it actually can be any descriptors. If this parameter is given, the filename_list, elements_list and fmt will be ignored. Defaults to None.
            filename_list (list): filename list, such as ['0.xyz', '1.xyz']. Defaults to None.
            elements_list (list): elements list, such as ['Al', 'C']. Defaults to None.
            start_idx (int, optional): for deterministic results, fix the first sampled point index. Defaults to None.
            fmt (str, optional): selected in ['data', 'lmp', 'dump', 'dump.gz', 'poscar', 'xyz', 'cif'], One can explicitly assign the file format or mdapy will handle it with the postsuffix of filename. Defaults to None.

        Returns:
            np.ndarray: selected index.
        """
        if des_total is None:
            assert filename_list is not None
            assert elements_list is not None
            des_total = []
            bar = tqdm(filename_list)
            for filename in bar:
                bar.set_description(f"Reading {filename}")
                fmt = BuildSystem.getformat(filename, fmt)
                if fmt in ["xyz", "dump", "dump.gz"]:
                    data, box, boundary, _ = BuildSystem.fromfile(filename, fmt)
                elif fmt in ["data", "lmp", "poscar", "cif"]:
                    data, box, boundary = BuildSystem.fromfile(filename, fmt)
                else:
                    raise "Something is wrong here."

                des = self.get_descriptors(
                    data.select(["x", "y", "z"]).to_numpy(),
                    box,
                    elements_list,
                    data["type"].to_numpy(),
                    boundary,
                ).sum(axis=0)
                des_total.append(des)
            des_total = np.array(des_total)
        assert des_total.ndim == 2, "Only support 2-D ndarray."
        n_points = des_total.shape[0]
        assert n_sample <= n_points, f"n_sample must <= {n_points}."
        assert n_sample > 0, "n_sample must be a positive number."
        if start_idx is None:
            start_idx = np.random.randint(0, n_points)
        else:
            assert (
                start_idx >= 0 and start_idx < n_points
            ), f"start_idx must belong [0, {n_points-1}]."

        sampled_indices = [start_idx]
        min_distances = np.full(n_points, np.inf)
        farthest_point_idx = start_idx
        bar = tqdm(range(n_sample - 1))
        for i in bar:
            bar.set_description(f"Fps sampling {i+2} frames")
            current_point = des_total[farthest_point_idx]
            dist_to_current_point = np.linalg.norm(des_total - current_point, axis=1)
            min_distances = np.minimum(min_distances, dist_to_current_point)
            farthest_point_idx = np.argmax(min_distances)
            sampled_indices.append(farthest_point_idx)

        return np.array(sampled_indices)


try:
    from lammps import lammps
except Exception:
    pass


class LammpsPotential(BasePotential):
    """This class provide a interface to use potential supported in lammps to calculate the energy, force and virial.

    Args:
        pair_parameter (str): including pair_style and pair_coeff.
        units (str, optional): lammps units, such as metal, real etc. Defaults to "metal".
        atomic_style (str, optional): atomic_style, such as atomic, charge etc. Defaults to "atomic".
        extra_args (str, optional): any lammps commond. Defaults to None.
        conversion_factor (dict, optional): units conversion. It must be {'energy':float, 'force':float, 'virial':float}. The float can be any number, while the key is fixed. Defaults to None.
    """

    def __init__(
        self,
        pair_parameter,
        units="metal",
        atomic_style="atomic",
        extra_args=None,
        conversion_factor=None,
    ):
        self.pair_parameter = pair_parameter
        self.units = units
        self.atomic_style = atomic_style
        self.extra_args = extra_args
        self.conversion_factor = conversion_factor

    def to_lammps_box(self, box):
        xlo, ylo, zlo = box[-1]
        xhi, yhi, zhi = (
            xlo + box[0, 0],
            ylo + box[1, 1],
            zlo + box[2, 2],
        )
        xy, xz, yz = box[1, 0], box[2, 0], box[2, 1]

        return (
            [xlo, ylo, zlo],
            [xhi, yhi, zhi],
            xy,
            xz,
            yz,
        )

    def compute(
        self,
        pos,
        box,
        elements_list,
        type_list,
        boundary=[1, 1, 1],
        centroid_stress=False,
    ):
        """This function is used to calculate the energy, force and virial.

        # If one use NEP potential, set centroid_stress to True to obtain right 9 indices virial.

        Args:
            pos (np.ndarray): (:math:`N_p, 3`) particles positions.
            box (np.ndarray): (:math:`4, 3`) system box.
            elements_list (list[str]): elements to be calculated, such as ['Al', 'Ni'].
            type_list (np.ndarray): (:math:`N_p`) atom type list.
            boundary (list, optional): boundary conditions, 1 is periodic and 0 is free boundary. Defaults to [1, 1, 1].
            centroid_stress (bool, optional): if Ture, use compute stress/atomm. If False, use compute centroid/stress/atom. Defaults to False.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: energy, force, virial.

        Shape:
            energy : (:math:`N_p`)
            force : (:math:`N_p, 3`). The order is fx, fy and fz.
            virial : (:math:`N_p, 9`). If centroid_stress is True, the order is xx, yy, zz, xy, xz, yz, yx, zx, zy. Otherwise the order is xx, yy, zz, xy, xz, yz.
        """
        boundary = " ".join(["p" if i == 1 else "s" for i in boundary])
        energy, force, virial = None, None, None
        lmp = lammps()
        try:
            lmp.commands_string(f"units {self.units}")
            lmp.commands_string(f"boundary {boundary}")
            lmp.commands_string(f"atom_style {self.atomic_style}")
            num_type = len(elements_list)
            create_box = f"""
            region 1 prism 0 1 0 1 0 1 0 0 0
            create_box {num_type} 1
            """
            lmp.commands_string(create_box)
            if box[0, 1] != 0 or box[0, 2] != 0 or box[1, 2] != 0:
                old_box = box.copy()
                ax = np.linalg.norm(box[0])
                bx = box[1] @ (box[0] / ax)
                by = np.sqrt(np.linalg.norm(box[1]) ** 2 - bx**2)
                cx = box[2] @ (box[0] / ax)
                cy = (box[1] @ box[2] - bx * cx) / by
                cz = np.sqrt(np.linalg.norm(box[2]) ** 2 - cx**2 - cy**2)
                box = np.array([[ax, bx, cx], [0, by, cy], [0, 0, cz]]).T
                rotation = np.linalg.solve(old_box[:-1], box)
                pos = pos @ rotation
                box = np.r_[box, box[-1].reshape(1, -1)]
            # lmp.reset_box(*self.to_lammps_box(box))
            lo, hi, xy, xz, yz = self.to_lammps_box(box)
            lmp.commands_string(
                f"change_box all x final {lo[0]} {hi[0]} y final {lo[1]} {hi[1]} z final {lo[2]} {hi[2]} xy final {xy} xz final {xz} yz final {yz}"
            )
            # print(box, lmp.extract_box())
            N = pos.shape[0]
            N_lmp = lmp.create_atoms(N, np.arange(1, N + 1), type_list, pos.flatten())
            assert N == N_lmp, "Wrong atom numbers."

            for i, m in enumerate(elements_list, start=1):
                mass = atomic_masses[atomic_numbers[m]]
                lmp.commands_string(f"mass {i} {mass}")
            lmp.commands_string(self.pair_parameter)
            if self.extra_args is not None:
                lmp.commands_string(self.extra_args)
            if centroid_stress:
                lmp.commands_string("compute 1 all centroid/stress/atom NULL")
            else:
                lmp.commands_string("compute 1 all stress/atom NULL")
            lmp.commands_string("compute 2 all pe/atom")
            # lmp.commands_string("variable vol equal vol")
            lmp.commands_string("run 0")
            sort_index = np.argsort(lmp.numpy.extract_atom("id"))
            energy = np.array(lmp.numpy.extract_compute("2", 1, 1))[sort_index]
            force = np.array(lmp.numpy.extract_atom("f"))[sort_index]
            virial = -np.array(lmp.numpy.extract_compute("1", 1, 2))[sort_index]
            # print(lmp.extract_variable("vol"))

        except Exception as e:
            lmp.close()
            os.remove("log.lammps")
            raise e
        lmp.close()
        os.remove("log.lammps")
        if self.conversion_factor is not None:
            energy *= self.conversion_factor["energy"]
            force *= self.conversion_factor["force"]
            virial *= self.conversion_factor["virial"]
        return energy, force, virial


if __name__ == "__main__":
    from lattice_maker import LatticeMaker
    from time import time
    from system import System

    # eam = EAM("example/Al_DFT.eam.alloy")
    # print(isinstance(eam, EAM))
    # print(isinstance(eam, NEP))

    ti.init(ti.cpu)

    # file_list = []
    # for step in range(10):
    #     file_list.append(
    #         rf"D:\Study\Gra-Al\init_data\active\gra_al\interface\300K\split\dump.{step}.xyz"
    #     )
    # file_list = np.array(file_list)
    # nep = NEP(r"D:\Study\Gra-Al\init_data\active\gra_al\interface\300K\nep.txt")

    # sele = nep.fps_sample(
    #     10, filename_list=file_list, elements_list=["Al", "C"], start_idx=2
    # )

    # print(sele)

    # des_total = []
    # for filename in file_list:
    #     system = System(filename)
    #     des = nep.get_descriptors(
    #         system.pos, system.box, ["Al", "C"], system.data["type"].to_numpy()
    #     ).sum(axis=0)
    #     des_total.append(des)
    # des_total = np.array(des_total)

    # sele = nep.fps_sample(10, des_total, start_idx=2)
    # print(sele)
    # start = time()
    # lattice_constant = 4.05
    # x, y, z = 10, 10, 10
    # FCC = LatticeMaker(lattice_constant, "FCC", x, y, z)
    # FCC.compute()
    # end = time()
    # print(f"Build {FCC.pos.shape[0]} atoms FCC time: {end-start} s.")

    # start = time()
    # nep = NEP(r"D:\Study\Gra-Al\potential_test\elastic\aluminum\elastic\nep.txt")

    # e, f, v = nep.compute(
    #     FCC.pos, FCC.box, ["Al"], np.ones(FCC.pos.shape[0], dtype=np.int32)
    # )
    # potential = LammpsPotential(
    #     "pair_style eam/alloy\npair_coeff * * example/Al_DFT.eam.alloy Al"
    # )
    # e, f, v = potential.compute(
    #     FCC.pos, FCC.box, ["Al"], np.ones(FCC.pos.shape[0], dtype=np.int32)
    # )
    # end = time()
    # print(f"Calculate energy and force time: {end-start} s.")
    system = System(r"D:\Study\Gra-Al\potential_test\phonon\alc\POSCAR")
    # nep = NEP(r"D:\Study\Gra-Al\potential_test\total\nep.txt")
    # lnep = LammpsPotential(
    #     r"""
    # pair_style nep
    # pair_coeff * * D:\Study\Gra-Al\potential_test\total\nep.txt Al C
    # """
    # )
    lnep = LammpsPotential(
        r"""
    pair_style nep 
    pair_coeff * * D:\Study\Gra-Al\potential_test\total\nep.txt Al C
    """
    )

    e, f, v = lnep.compute(
        system.pos,
        system.box,
        ["Al", "C"],
        system.data["type"].to_numpy(),
        [1, 1, 1],
        True,
    )
    print(e.sum())
    print(f[:5])
    print(v.sum(axis=0) / system.vol * 6.241509125883258e-07)
    print(v[:2] * 6.241509125883258e-07)  # to eV
    # start = time()
    # des = nep.get_descriptors(
    #     system.pos, system.box, ["Al", "C"], system.data["type"].to_numpy()
    # )
    # end = time()
    # print(f"Calculate descriptors time: {end-start} s.")
    # print(des[:5])
    # system = System(
    #     r"D:\Study\Gra-Al\potential_test\phonon\graphene\phonon_interface\stress\test.0.dump"
    # )
    # potential = EAM("./example/Al_DFT.eam.alloy")
    # start = time()
    # energy, force, virial = potential.compute(
    #     system.pos, system.box, ["Al"], np.ones(system.N, dtype=np.int32)
    # )
    # end = time()
    # print(f"Calculate energy and force time: {end-start} s.")
    # print("energy:")
    # print(energy[:4])
    # print("force:")
    # print(force[:4, :])
    # print("virial:")
    # print(virial[:4, :] * 160.2176621 * 1e4)

    # print(potential.d_elec_density_data[0, :10])
    # print(potential.d_elec_density_data[0, :10])
    # print(potential.d_phi_data[0, :10])
    # potential.plot()
    # plt.plot(potential.r, potential.d_phi_data[0][0])
    # plt.plot(potential.r, potential.d_embedded_data[0])
    # plt.plot(potential.rho, potential.d_elec_density_data[0])
    # plt.plot(potential.rho, potential.elec_density_data[0])
    # plt.show()

    # potential.write_eam_alloy()
    # potential = EAM("CoNiFeAlCu.new.eam.alloy")
    # potential = EAM("./example/Al_DFT.eam.alloy")
