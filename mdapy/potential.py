# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import taichi as ti
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import matplotlib.pyplot as plt

try:
    from plotset import set_figure
    from tool_function import _check_repeat_cutoff
    from replicate import Replicate
    from neighbor import Neighbor
    from nep._nep import NepCalculator
except Exception:
    from .plotset import set_figure
    from .tool_function import _check_repeat_cutoff
    from .replicate import Replicate
    from .neighbor import Neighbor
    from .nep._nep import NepCalculator


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


class EAM:
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

        >>> potential = EAM("./example/CoNiFeAlCu.eam.alloy") # Read a eam.alloy file.

        >>> potential.embedded_data # Check embedded energy.

        >>> potential.phi_data # Check pair interaction.

        >>> potential.plot() # Plot information of potential.
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
        return Cal.energy, Cal.force


class NEP:

    def __init__(self, filename) -> None:
        self.filename = filename
        self._nep = NepCalculator(filename)
        self.info = self._nep.info
        self.rc = max(self.info["radial_cutoff"], self.info["angular_cutoff"])

    def compute(self, pos, box, elements_list, type_list, boundary=[1, 1, 1]):
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
        box = box.flatten()
        pos = pos.T.flatten()

        e, f, v = self._nep.calculate(type_list, box, pos)

        e = np.array(e)
        f = np.array(f).reshape(3, -1).T
        v = np.array(v).reshape(9, -1)

        return e, f, v

    def get_descriptors(self, pos, box, elements_list, type_list, boundary=[1, 1, 1]):
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
        box = box.flatten()
        pos = pos.T.flatten()

        return (
            np.array(self._nep.get_descriptors(type_list, box, pos))
            .reshape(-1, len(type_list))
            .T
        )


if __name__ == "__main__":

    from lattice_maker import LatticeMaker
    from time import time

    ti.init(ti.cpu)
    start = time()
    lattice_constant = 4.05
    x, y, z = 10, 10, 10
    FCC = LatticeMaker(lattice_constant, "FCC", x, y, z)
    FCC.compute()
    end = time()
    print(f"Build {FCC.pos.shape[0]} atoms FCC time: {end-start} s.")

    start = time()
    nep = NEP(r"D:\Study\Gra-Al\init_data\nep_interface\nep.txt")

    e, f, v = nep.compute(
        FCC.pos, FCC.box, ["Al"], np.ones(FCC.pos.shape[0], dtype=np.int32)
    )
    end = time()
    print(f"Calculate energy and force time: {end-start} s.")
    print(e[:5])
    print(f[:5])
    print(v[:5])
    start = time()
    des = nep.get_descriptors(
        FCC.pos, FCC.box, ["Al"], np.ones(FCC.pos.shape[0], dtype=np.int32)
    )
    end = time()
    print(f"Calculate descriptors time: {end-start} s.")
    print(des[:5])

    # potential = EAM("./example/CoNiFeAlCu.eam.alloy")
    # start = time()
    # energy, force = potential.compute(
    #     FCC.pos * 0.9, FCC.box * 0.9, ["Al"], np.ones(FCC.pos.shape[0], dtype=np.int32)
    # )
    # end = time()
    # print(f"Calculate energy and force time: {end-start} s.")
    # print("energy:")
    # print(energy[:4])
    # print("force:")
    # print(force[:4, :])

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
