# Copyright (c) 2022-2024, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import matplotlib.pyplot as plt
import numpy as np
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections

try:
    from .plotset import set_figure
    from .potential import BasePotential
except Exception:
    from plotset import set_figure
    from potential import BasePotential


class Phonon:
    """This class can be used to calculate the phono dispersion based on Phonopy. We support NEP and
    eam/alloy potential now.

    Args:
        path (str | np.ndarray| nested list): band path, such as '0. 0. 0. 0.5 0.5 0.5', indicating two points.
        labels (str | list): high symmetry points label, such as ["$\Gamma$", "K", "M", "$\Gamma$"].
        potential (BasePotential): base potential class defined in mdapy, which must including a compute method to calculate the energy, force, virial.
        pos (np.ndarray): (:math:`N_p, 3`) particles positions.
        box (np.ndarray): (:math:`4, 3`) system box.
        elements_list (list[str]): element list, such as ['Al']
        type_list (np.ndarray): (:math:`N_p`) atom type list.
        replicate (list, optional): replication to pos, such as [3, 3, 3]. If not given, we will replicate it exceeding 15 A per directions. Defaults to None.

    Outputs:
        **band_dict** : band information, which can be used to plot phono dispersion.
    """

    def __init__(
        self,
        path,
        labels,
        potential,
        pos,
        box,
        elements_list,
        type_list,
        replicate=None,
    ):

        if isinstance(path, str):
            self.path = np.array(path.split(), float).reshape(1, -1, 3)
        else:
            assert len(path[0]) == 3
            self.path = np.array(path).reshape(1, -1, 3)
        if isinstance(labels, str):
            self.labels = labels.split()
        else:
            self.labels = labels
        assert (
            len(self.labels) == self.path.shape[1]
        ), "The length of path should be equal to labels."

        assert isinstance(
            potential, BasePotential
        ), "potential must be a mdapy BasePotential."

        self.potential = potential
        self.pos = pos
        assert box.shape == (4, 3)
        self.box = box[:-1]
        self.scaled_positions = self.pos @ np.linalg.inv(self.box)
        self.elements_list = elements_list
        self.type_list = type_list
        self.type_name = [elements_list[i - 1] for i in self.type_list]

        if replicate is None:
            lengths = np.linalg.norm(self.box, axis=1)
            self.replicate = np.round(15.0 / lengths).astype(int)
        else:
            self.replicate = replicate

        self.bands_dict = None
        self.dos_dict = None
        self.pdos_dict = None
        self.thermal_dict = None
        self.get_force_constants()

    def build_phononAtoms(self):

        return PhonopyAtoms(
            symbols=self.type_name,
            cell=self.box,
            scaled_positions=self.scaled_positions,
        )

    def get_force_constants(self):

        unitcell = self.build_phononAtoms()

        self.phonon = Phonopy(
            unitcell=unitcell,
            supercell_matrix=self.replicate,
            primitive_matrix="auto",
        )
        self.phonon.generate_displacements(distance=0.01)
        supercells = self.phonon.get_supercells_with_displacements()
        set_of_forces = []
        type_list = np.array(self.type_list.tolist() * np.prod(self.replicate), int)

        for cell in supercells:
            _, forces, _ = self.potential.compute(
                cell.get_positions(),
                np.r_[cell.get_cell(), np.zeros((1, 3))],
                self.elements_list,
                type_list,
                [1, 1, 1],
            )

            forces -= np.mean(forces, axis=0)
            set_of_forces.append(forces)
        set_of_forces = np.array(set_of_forces)
        self.phonon.produce_force_constants(forces=set_of_forces)

    def compute(self):

        qpoints, connections = get_band_qpoints_and_path_connections(
            self.path, npoints=101
        )
        self.phonon.run_band_structure(
            qpoints, path_connections=connections, labels=self.labels
        )
        self.bands_dict = self.phonon.get_band_structure_dict()

    def compute_dos(self, mesh=(10, 10, 10)):
        self.phonon.run_mesh(mesh)
        self.phonon.run_total_dos(use_tetrahedron_method=True)
        self.dos_dict = self.phonon.get_total_dos_dict()

    def compute_pdos(self, mesh=(10, 10, 10)):
        self.phonon.run_mesh(mesh, with_eigenvectors=True, is_mesh_symmetry=False)
        self.phonon.run_projected_dos()
        self.pdos_dict = self.phonon.get_projected_dos_dict()

    def compute_thermal(self, t_min, t_step, t_max, mesh=(10, 10, 10)):
        self.phonon.run_mesh(mesh)
        self.phonon.run_thermal_properties(t_min=t_min, t_step=t_step, t_max=t_max)
        self.thermal_dict = self.phonon.get_thermal_properties_dict()

    def plot_dos(self):
        if self.dos_dict is None:
            raise "call compute_dos before plot_dos."
        fig, ax = set_figure(
            figsize=(10, 7.5), bottom=0.15, left=0.16, use_pltset=True, figdpi=150
        )
        x, y = self.dos_dict["frequency_points"], self.dos_dict["total_dos"]
        ax.plot(x, y)
        ax.set_xlabel("Frequency (THz)")
        ax.set_ylabel("Density of states")
        ax.set_ylim(y.min(), y.max() * 1.1)
        plt.show()
        return fig, ax

    def plot_pdos(self):

        if self.pdos_dict is None:
            raise "call compute_pdos before plot_pdos."

        fig, ax = set_figure(
            figsize=(10, 7.5), bottom=0.15, left=0.16, use_pltset=True, figdpi=150
        )
        x, y1 = self.pdos_dict["frequency_points"], self.pdos_dict["projected_dos"]
        for i, y in enumerate(y1, start=1):
            ax.plot(x, y, label=f"[{i}]")

        ax.legend()
        ax.set_xlabel("Frequency (THz)")
        ax.set_ylabel("Partial density of states")
        ax.set_ylim(y1.min(), y1.max() * 1.1)
        plt.show()
        return fig, ax

    def plot_thermal(self):
        if self.thermal_dict is None:
            raise "call compute_thermal before plot_thermal."

        temperatures = self.thermal_dict["temperatures"]
        free_energy = self.thermal_dict["free_energy"]
        entropy = self.thermal_dict["entropy"]
        heat_capacity = self.thermal_dict["heat_capacity"]

        fig, ax = set_figure(
            figsize=(10, 7.5), bottom=0.15, left=0.16, use_pltset=True, figdpi=150
        )
        ax.plot(temperatures, free_energy, label="Free energy (kJ/mol)")
        ax.plot(temperatures, entropy, label="Entropy (J/K/mol)")
        ax.plot(temperatures, heat_capacity, label="$C_v$ (J/K/mol)")

        ax.legend()
        ax.set_xlabel("Temperature (K)")

        ax.set_xlim(temperatures[0], temperatures[-1])
        plt.show()
        return fig, ax

    def _rgb2hex(self, rgb):
        color = "#"
        for i in rgb:
            color += str(hex(i))[-2:].replace("x", "0").upper()
        return color

    def plot_dispersion(
        self,
        units="THz",
        yticks=None,
        ylim=None,
        color=None,
        merge_kpoints=None,
    ):
        """This function can plot the phonon dispersion.

        Args:
            units (str, optional): units of frequency, selected in ['THz', '1/cm']. Defaults to "THz".
            yticks (list[float], optional): y axis ticks, such as [0, 10, 20]. Defaults to None.
            ylim (list[float], optional): y axis limitation, such as [0, 100]. Defaults to None.
            color (str | rgb turple, optional): line color, can be a str, such as 'r', '#729CBD', or a rgb turple, such as [125, 125, 125]. Defaults to None.
            merge_kpoints (list, optional): sometimes you want to merge two equalvalue points, such as [2, 3]. Defaults to None.

        Returns:
            tuple: (fig, ax) matplotlib figure and axis class.
        """
        if self.bands_dict is None:
            self.compute()

        fig, ax = set_figure(
            figsize=(10, 7.5), bottom=0.08, left=0.16, use_pltset=True, figdpi=150
        )
        if color is None:
            color = "deepskyblue"
        elif isinstance(color, str):
            color = color
        else:
            assert (
                len(color) == 3
            ), "Only support str or a three-elements rgb turple, such as [125, 125, 125]."
            color = self._rgb2hex(color)
        frequencies = self.bands_dict["frequencies"]
        distances = self.bands_dict["distances"]
        kpoints = [distances[0][0]] + [i[-1] for i in distances]
        if merge_kpoints is None:
            for i in range(len(distances)):
                x, y = distances[i], frequencies[i]
                if units == "1/cm":
                    y *= 33.4
                ax.plot(x, y, lw=1.2, c=color)

            ax.plot(
                [kpoints[0], kpoints[-1]],
                [0, 0],
                "--",
                c="grey",
                lw=1.0,
                alpha=0.5,
            )
            ax.set_xlim(kpoints[0], kpoints[-1])
            ax.set_xticks(kpoints)
        else:
            assert len(merge_kpoints) == 2
            assert min(merge_kpoints) >= 0
            assert max(merge_kpoints) <= len(kpoints) - 1
            L, R = merge_kpoints
            assert L < R
            move = kpoints[R] - kpoints[L]
            for i in range(len(distances)):
                x, y = distances[i], frequencies[i]
                if units == "1/cm":
                    y *= 33.4
                if x.min() >= kpoints[L] - 0.01 and x.max() <= kpoints[R] + 0.01:
                    pass
                else:
                    if x.min() >= kpoints[R] - 0.01:
                        ax.plot(x - move, y, lw=1.2, c=color)
                    else:
                        ax.plot(x, y, lw=1.2, c=color)
            ax.plot(
                [kpoints[0], kpoints[-1] - move],
                [0, 0],
                "--",
                c="grey",
                lw=1.0,
                alpha=0.5,
            )
            ax.set_xlim(kpoints[0], kpoints[-1] - move)

            if R == len(kpoints) - 1:
                ax.set_xticks(kpoints[: L + 1])
            else:
                ax.set_xticks(
                    np.hstack(
                        [
                            kpoints[: L + 1],
                            (kpoints[min(R + 1, len(kpoints) - 1) :] - move),
                        ]
                    )
                )

        ax.set_xticks([], minor=True)
        ax.set_yticks([], minor=True)

        if merge_kpoints is None:
            ax.set_xticklabels(self.labels)
        else:
            if R == len(kpoints) - 1:
                ax.set_xticklabels(
                    np.hstack(
                        [
                            self.labels[:L],
                            [self.labels[L] + "$|$" + self.labels[R]],
                        ]
                    )
                )
            else:
                ax.set_xticklabels(
                    np.hstack(
                        [
                            self.labels[:L],
                            [self.labels[L] + "$|$" + self.labels[R]],
                            self.labels[min(R + 1, len(self.labels) - 1) :],
                        ]
                    )
                )
        if yticks is not None:
            ax.set_yticks(yticks)

        if units == "1/cm":
            ax.set_ylabel("Frequency ($cm^{-1}$)")
        else:
            ax.set_ylabel("Frequency (THz)")

        if ylim is not None:
            ylo, yhi = ylim
        else:
            ylo, yhi = ax.get_ylim()
        xticks = ax.get_xticks()
        for i in xticks:
            ax.plot(
                [i, i],
                [ylo, yhi],
                "--",
                lw=0.8,
                c="grey",
                alpha=0.5,
            )

        ax.set_ylim(ylo, yhi)
        plt.show()
        return fig, ax

    @classmethod
    def read_band_data(cls, filename):
        with open(filename) as op:
            band = op.readlines()
        kpoints = np.array(band[1].split()[1:], float)
        sepa = [1]
        for i, j in enumerate(band):
            if j == "\n":
                sepa.append(i)
        data = {}
        num = 0
        for i in range(len(sepa) - 1):
            pot = np.array([k.split() for k in band[sepa[i] + 1 : sepa[i + 1]]], float)
            if len(pot) > 0:
                data[f"{num}"] = pot
                num += 1
        return kpoints, data

    @classmethod
    def plot_dispersion_from_band_data(
        cls,
        filename,
        labels,
        units="THz",
        yticks=None,
        ylim=None,
        color=None,
        merge_kpoints=None,
    ):
        """This function can plot the phonon dispersion based on band.dat.

        Args:
            filename (str) : filename of band data generated from phonopy, such as band.dat.
            labels (list[str], optional): kpoints label, such as ["$\Gamma$", "K", "M", "$\Gamma$"] for graphene.
            units (str, optional): units of frequency, selected in ['THz', '1/cm']. Defaults to "THz".
            yticks (list[float], optional): y axis ticks, such as [0, 10, 20]. Defaults to None.
            ylim (list[float], optional): y axis limitation, such as [0, 100]. Defaults to None.
            color (str | rgb turple, optional): line color, can be a str, such as 'r', '#729CBD', or a rgb turple, such as [125, 125, 125]. Defaults to None.
            merge_kpoints (list, optional): sometimes you want to merge two equalvalue points, such as [2, 3]. Defaults to None.

        Returns:
            tuple: (fig, ax) matplotlib figure and axis class.
        """

        kpoints, data = cls.read_band_data(filename)
        if isinstance(labels, str):
            labels = labels.split()
        assert len(labels) == len(
            kpoints
        ), f"length of labels should be {len(kpoints)}."
        fig, ax = set_figure(
            figsize=(10, 7.5), bottom=0.08, left=0.16, use_pltset=True, figdpi=150
        )
        if color is None:
            color = "deepskyblue"
        elif isinstance(color, str):
            color = color
        else:
            assert (
                len(color) == 3
            ), "Only support str or a three-elements rgb turple, such as [125, 125, 125]."
            color = cls._rgb2hex(color)
        if merge_kpoints is None:
            for i in data.keys():
                x, y = data[i][:, 0], data[i][:, 1]
                if units == "1/cm":
                    y *= 33.4
                ax.plot(x, y, lw=1.2, c=color)

            ax.plot(
                [kpoints[0], kpoints[-1]],
                [0, 0],
                "--",
                c="grey",
                lw=1.0,
                alpha=0.5,
            )
            ax.set_xlim(kpoints[0], kpoints[-1])
            ax.set_xticks(kpoints)
        else:
            assert len(merge_kpoints) == 2
            assert min(merge_kpoints) >= 0
            assert max(merge_kpoints) <= len(kpoints) - 1
            L, R = merge_kpoints
            assert L < R
            move = kpoints[R] - kpoints[L]
            for i in data.keys():
                x, y = data[i][:, 0], data[i][:, 1]
                if units == "1/cm":
                    y *= 33.4
                if x.min() >= kpoints[L] - 0.01 and x.max() <= kpoints[R] + 0.01:
                    pass
                else:
                    if x.min() >= kpoints[R] - 0.01:
                        ax.plot(x - move, y, lw=1.2, c=color)
                    else:
                        ax.plot(x, y, lw=1.2, c=color)
            ax.plot(
                [kpoints[0], kpoints[-1] - move],
                [0, 0],
                "--",
                c="grey",
                lw=1.0,
                alpha=0.5,
            )
            ax.set_xlim(kpoints[0], kpoints[-1] - move)

            if R == len(kpoints) - 1:
                ax.set_xticks(kpoints[: L + 1])
            else:
                ax.set_xticks(
                    np.hstack(
                        [
                            kpoints[: L + 1],
                            (kpoints[min(R + 1, len(kpoints) - 1) :] - move),
                        ]
                    )
                )

        ax.set_xticks([], minor=True)
        ax.set_yticks([], minor=True)
        if yticks is not None:
            ax.set_yticks(yticks)

        if merge_kpoints is None:
            ax.set_xticklabels(labels)
        else:
            if R == len(kpoints) - 1:
                ax.set_xticklabels(
                    np.hstack(
                        [
                            labels[:L],
                            [labels[L] + "$|$" + labels[R]],
                        ]
                    )
                )
            else:
                ax.set_xticklabels(
                    np.hstack(
                        [
                            labels[:L],
                            [labels[L] + "$|$" + labels[R]],
                            labels[min(R + 1, len(labels) - 1) :],
                        ]
                    )
                )

        if units == "1/cm":
            ax.set_ylabel("Frequency ($cm^{-1}$)")
        else:
            ax.set_ylabel("Frequency (THz)")

        if ylim is not None:
            ylo, yhi = ylim
        else:
            ylo, yhi = ax.get_ylim()
        xticks = ax.get_xticks()
        for i in xticks:
            ax.plot(
                [i, i],
                [ylo, yhi],
                "--",
                lw=0.8,
                c="grey",
                alpha=0.5,
            )

        ax.set_ylim(ylo, yhi)
        plt.show()

        return fig, ax


if __name__ == "__main__":

    # Phonon.plot_dispersion_from_band_data(
    #     r"D:\Study\Gra-Al\init_data\cp2k_test\band_data\aluminum\band.dat",
    #     "$\Gamma$ X U K $\Gamma$ L",
    #     merge_kpoints=[2, 3],
    #     ylim=[0, 10],
    # )

    import taichi as ti

    ti.init()
    from lattice_maker import LatticeMaker
    from potential import LammpsPotential, EAM, NEP

    lat = LatticeMaker(1.42, "GRA", 1, 1, 1)
    lat.compute()
    lat.box[2, 2] += 20

    # potential = LammpsPotential(
    #     """pair_style airebo 3.0
    #    pair_coeff * * D:\Study\Gra-Al\potential_test\phonon\graphene\phonon_interface\CH.airebo C"""
    # )
    potential = NEP("example/C_2022_NEP3.txt")
    # potential = NEP(r"D:\Study\Gra-Al\potential_test\phonon\aluminum\nep.txt")
    pho = Phonon(
        "0.0 0.0 0.0 0.3333333333 0.3333333333 0.0 0.5 0.0 0.0 0.0 0.0 0.0",
        "$\Gamma$ K M $\Gamma$",
        potential,
        lat.pos,
        lat.box,
        elements_list=["C"],
        type_list=lat.type_list,
    )
    # pho = Phonon(
    #     "0.0 0.0 0.0 0.5 0.0 0.5 0.625 0.25 0.625 0.375 0.375 0.75 0.0 0.0 0.0 0.5 0.5 0.5",
    #     "$\Gamma$ X U K $\Gamma$ L",
    #     potential,
    #     lat.pos,
    #     lat.box,
    #     elements_list=["Al"],
    #     type_list=lat.type_list,
    # )
    pho.compute()
    # pho.compute_thermal(0, 50, 1000, (30, 30, 30))
    # pho.plot_thermal()
    pho.plot_dispersion(
        units="1/cm",
    )
    # pho = Phonon(r"D:\Study\Gra-Al\init_data\cp2k_test\band_data\aluminum\band.dat")
    # # ["$\Gamma$", "X", "U", "K", "$\Gamma$", "L"] Al
    # # ["$\Gamma$", "K", "M", "$\Gamma$"] graphene
    # # [
    # #     "$\Gamma$",
    # #     "T",
    # #     "$H_2$",
    # #     "L",
    # #     "$\Gamma$",
    # #     "$S_0$",
    # #     "F",
    # #     "$\Gamma$",
    # # ] alc
    # # (24, 170, 201)
    # fig, ax = pho.plot_dispersion(
    #     kpoints_label=["$\Gamma$", "X", "U", "K", "$\Gamma$", "L"],
    #     color="#729CBD",
    #     units="1/cm",
    #     merge_kpoints=[2, 3],
    #     # yticks=range(0, 2000, 400),
    # )
    # # fig.savefig(
    # #     r"D:\Study\Gra-Al\init_data\cp2k_test\band_data\aluminum\band.png",
    # #     dpi=300,
    # #     bbox_inches="tight",
    # #     transparent=True,
    # # )

    # # print(Phonon.get_supercell(1, 1, 1, "cp2k.inp"))
