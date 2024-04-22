# Copyright (c) 2022-2024, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import matplotlib.pyplot as plt
import numpy as np
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections

try:
    from .plotset import set_figure
    from .potential import EAM, NEP
except Exception:
    from plotset import set_figure
    from potential import EAM, NEP


class Phonon:
    def __init__(
        self,
        path,
        labels,
        pos,
        box,
        filename,
        elements_list,
        type_list,
        pair_style="eam/alloy",
        replicate=None,
    ):
        """This class can be used to calculate the phono dispersion based on Phonopy. We support NEP and
        eam/alloy potential now.

        Args:
            path (str | np.ndarray| nested list): band path, such as '0. 0. 0. 0.5 0.5 0.5', indicating two points.
            labels (str | list): high symmetry points label, such as ["$\Gamma$", "K", "M", "$\Gamma$"].
            pos (np.ndarray): (:math:`N_p, 3`) particles positions.
            box (np.ndarray): (:math:`4, 3`) system box.
            filename (str): potential filename, such as 'nep.txt'.
            elements_list (list[str]): element list, such as ['Al']
            type_list (np.ndarray): (:math:`N_p`) atom type list.
            pair_style (str, optional): pair style, selected in ['nep', 'eam/alloy']. Defaults to "eam/alloy".
            replicate (list, optional): replication to pos, such as [3, 3, 3]. If not given, we will replicate it exceeding 15 A per directions. Defaults to None.

        Outputs:
            **band_dict** : band information, which can be used to plot phono dispersion.
        """

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

        self.pos = pos
        assert box.shape == (4, 3)
        self.box = box[:-1]
        self.scaled_positions = self.pos @ np.linalg.inv(self.box)
        self.filename = filename
        self.elements_list = elements_list
        self.type_list = type_list
        self.type_name = [elements_list[i - 1] for i in self.type_list]
        self.pair_style = pair_style

        if replicate is None:
            lengths = np.linalg.norm(self.box, axis=1)
            self.replicate = np.round(15.0 / lengths).astype(int)
        else:
            self.replicate = replicate

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
        type_list = self.type_list.tolist() * np.prod(self.replicate)
        if self.pair_style == "eam/alloy":
            potential = EAM(self.filename)
        elif self.pair_style == "nep":
            potential = NEP(self.filename)
        for cell in supercells:
            _, forces, _ = potential.compute(
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
        self.get_force_constants()
        qpoints, connections = get_band_qpoints_and_path_connections(
            self.path, npoints=101
        )
        self.phonon.run_band_structure(
            qpoints, path_connections=connections, labels=self.labels
        )
        self.bands_dict = self.phonon.get_band_structure_dict()

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
        fig, ax = set_figure(
            figsize=(10, 7.5), bottom=0.08, left=0.16, use_pltset=True, figdpi=150
        )
        if color is None:
            color = "b"
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
            color = "b"
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

    lat = LatticeMaker(4.05, "FCC", 1, 1, 1)
    lat.compute()
    # gra.box[2, 2] += 20

    pho = Phonon(
        "0.0 0.0 0.0 0.5 0.0 0.5 0.625 0.25 0.625 0.375 0.375 0.75 0.0 0.0 0.0 0.5 0.5 0.5",
        "$\Gamma$ X U K $\Gamma$ L",
        lat.pos,
        lat.box,
        filename=r"example\Al_DFT.eam.alloy",
        elements_list=["Al"],
        type_list=lat.type_list,
        pair_style="eam/alloy",
    )
    pho.compute()
    pho.plot_dispersion(
        units="1/cm", merge_kpoints=[2, 3], color=(123, 204, 33), ylim=[0, 350]
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
