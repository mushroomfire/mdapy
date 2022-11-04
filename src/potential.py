import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.getcwd())
from plot.pltset import pltset, cm2inch


class EAM:
    def __init__(self, filename):
        self.filename = filename
        self.read_eam_alloy()
        pltset()

    def read_eam_alloy(self):

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
                self.rho, self.embedded_data[i], k=3
            ).derivative(n=1)(self.rho)
            self.d_elec_density_data[i] = spline(
                self.r, self.elec_density_data[i], k=3
            ).derivative(n=1)(self.r)

        self.phi_data = np.zeros((self.Nelements, self.Nelements, self.nr))
        self.d_phi_data = np.zeros((self.Nelements, self.Nelements, self.nr))

        for i in range(self.Nelements):
            for j in range(self.Nelements):
                if i >= j:
                    self.phi_data[i, j, 1:] = self.rphi_data[i, j][1:] / self.r[1:]
                    self.d_phi_data[i, j, 1:] = spline(
                        self.r[1:], self.phi_data[i, j][1:], k=3
                    ).derivative(n=1)(self.r[1:])
                    if i != j:
                        self.phi_data[j, i] = self.phi_data[i, j]
                        self.d_phi_data[j, i] = self.d_phi_data[i, j]
        self.phi_data[:, :, 0] = self.phi_data[:, :, 1]
        self.d_phi_data[:, :, 0] = self.d_phi_data[:, :, 1]

    def plot_rho_r(self):
        fig = plt.figure(figsize=(cm2inch(10), cm2inch(7)), dpi=150)
        plt.subplots_adjust(bottom=0.18, top=0.98, left=0.2, right=0.98)
        for i in range(self.Nelements):
            plt.plot(self.r, self.elec_density_data[i], label=self.elements_list[i])
        plt.legend()
        plt.xlim(0, self.rc)
        plt.xlabel("r ($\mathregular{\AA}$)")
        plt.ylabel(r"$\mathregular{\rho}$ (r)")
        ax = plt.gca()
        plt.show()
        return fig, ax

    def plot_embded_rho(self):
        fig = plt.figure(figsize=(cm2inch(10), cm2inch(7)), dpi=150)
        plt.subplots_adjust(bottom=0.18, top=0.98, left=0.2, right=0.98)
        for i in range(self.Nelements):
            plt.plot(self.rho, self.embedded_data[i], label=self.elements_list[i])

        plt.legend()
        plt.xlim(0, self.rho[-1])
        plt.xlabel(r"$\mathregular{\rho}$")
        plt.ylabel(r"F($\mathregular{\rho}$) (eV)")
        ax = plt.gca()
        plt.show()
        return fig, ax

    def plot_phi_r(self):
        fig = plt.figure(figsize=(cm2inch(10), cm2inch(7)), dpi=150)
        plt.subplots_adjust(bottom=0.18, top=0.97, left=0.2, right=0.98)
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
        ax = plt.gca()
        plt.show()
        return fig, ax

    def plot(self):
        self.plot_rho_r()
        self.plot_embded_rho()
        self.plot_phi_r()


if __name__ == "__main__":

    potential = EAM("./example/CoNiFeAlCu.eam.alloy")
    # potential = EAM("./example/Al_DFT.eam.alloy")
    potential.plot()
