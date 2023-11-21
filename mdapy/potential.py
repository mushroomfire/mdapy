# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import matplotlib.pyplot as plt

try:
    from .plotset import set_figure
except Exception:
    from plotset import set_figure


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


if __name__ == "__main__":
    potential = EAM("./example/CoNiFeAlCu.eam.alloy")
    # print(potential.d_elec_density_data[0, :10])
    # print(potential.d_elec_density_data[0, :10])
    # print(potential.d_phi_data[0, :10])
    potential.plot()
    # plt.plot(potential.r, potential.d_phi_data[0][0])
    # plt.plot(potential.r, potential.d_embedded_data[0])
    # plt.plot(potential.rho, potential.d_elec_density_data[0])
    # plt.plot(potential.rho, potential.elec_density_data[0])
    # plt.show()

    # potential.write_eam_alloy()
    # potential = EAM("CoNiFeAlCu.new.eam.alloy")
    # potential = EAM("./example/Al_DFT.eam.alloy")
