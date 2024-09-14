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
        symprec (float): this is used to set geometric tolerance to find symmetry of crystal structure. Defaults to 1e-5.
        replicate (list, optional): replication to pos, such as [3, 3, 3]. If not given, we will replicate it exceeding 15 A per directions. Defaults to None.
        displacement (float, optional): displacement distance. Defaults to 0.01.
        cutoff_radius (float, optional): Set zero to force constants outside of cutoff distance. If not given, the force constant will consider the whole supercell. This parameter will not reduce the computation cost. Defaults to None.

    Outputs:
        **bands_dict** : band information, which can be used to plot phono dispersion.
        **dos_dict** : dos information.
        **pdos_dict** : pdos information.
        **thermal_dict** : thermal information.
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
        symprec=1e-5,
        replicate=None,
        displacement=0.01,
        cutoff_radius=None,
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
        self.elements_list = elements_list
        self.type_list = type_list
        self.type_name = [elements_list[i - 1] for i in self.type_list]

        if replicate is None:
            lengths = np.linalg.norm(self.box, axis=1)
            self.replicate = np.ceil(15.0 / lengths).astype(int)
        else:
            self.replicate = replicate
        self.symprec = symprec
        self.displacement = float(displacement)
        self.cutoff_radius = cutoff_radius
        self.bands_dict = None
        self.dos_dict = None
        self.pdos_dict = None
        self.thermal_dict = None
        self.get_force_constants()

    def build_phononAtoms(self):
        return PhonopyAtoms(
            symbols=self.type_name,
            cell=self.box,
            positions=self.pos,
        )

    def get_force_constants(self):
        unitcell = self.build_phononAtoms()

        self.phonon = Phonopy(
            unitcell=unitcell,
            supercell_matrix=self.replicate,
            primitive_matrix="auto",
            symprec=self.symprec,
        )

        self.phonon.generate_displacements(distance=self.displacement)

        supercells = self.phonon.get_supercells_with_displacements()
        set_of_forces = []
        # type_list = np.array(self.type_list.tolist() * np.prod(self.replicate), int)
        type_dict = {j: i + 1 for i, j in enumerate(self.elements_list)}
        type_list = np.array(
            [type_dict[symbols] for symbols in supercells[0].get_chemical_symbols()]
        )

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
        if self.cutoff_radius is not None:
            self.phonon.set_force_constants_zero_with_radius(float(self.cutoff_radius))

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

    def _check_color(self, color):
        if color is None:
            color = "deepskyblue"
        elif isinstance(color, str):
            color = color
        else:
            assert (
                len(color) == 3
            ), "Only support str or a three-elements rgb turple, such as [125, 125, 125]."
            color = self._rgb2hex(color)
        return color

    def _plot_lines(self, ax, units, filename, **kwargs):
        if "c" not in kwargs.keys() and "color" not in kwargs.keys():
            kwargs["c"] = "deepskyblue"

        if filename is None:
            frequencies = self.bands_dict["frequencies"]
            distances = self.bands_dict["distances"]
            kpoints = [distances[0][0]] + [i[-1] for i in distances]
            for i in range(len(distances)):
                x, y = distances[i], frequencies[i]
                if units == "1/cm":
                    y *= 33.4
                h = ax.plot(x, y, **kwargs)
            ax.set_xlim(kpoints[0], kpoints[-1])
            ax.set_xticks(kpoints)
            ax.set_xticklabels(self.labels)
        else:
            kpoints, data = self.read_band_data(filename)
            for i in data.keys():
                x, y = data[i][:, 0], data[i][:, 1]
                if units == "1/cm":
                    y *= 33.4
                h = ax.plot(x, y, **kwargs)
            ax.set_xlim(kpoints[0], kpoints[-1])
            ax.set_xticks(kpoints)
            ax.set_xticklabels(self.labels)
        return h[0]

    def plot_dispersion(
        self,
        fig=None,
        ax=None,
        units="THz",
        filename=None,
        **kwargs,
    ):
        """This function is used to plot the phonon dispersion.

        Args:
            fig (matplotlib.figure.Figure, optional): figure object. Defaults to None.
            ax (matplotlib.axes.Axes, optional): axes object. Defaults to None.
            units (str, optional): selected in ['THz', '1/cm']. Defaults to 'THz'.
            filename (str, optional): One can obtain the band data from band.dat saved by Phonopy. Defaults to None.

        Returns:
            tuple: (fig, ax, line)
        """
        if self.bands_dict is None:
            self.compute()
        if fig is None or ax is None:
            fig, ax = set_figure(
                figsize=(10, 7.5), bottom=0.08, left=0.16, use_pltset=True, figdpi=200
            )
        line = self._plot_lines(ax, units, filename, **kwargs)
        if units == "1/cm":
            ax.set_ylabel("Frequency ($cm^{-1}$)")
        else:
            ax.set_ylabel("Frequency (THz)")

        return fig, ax, line

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
    import mdapy as mp

    # lat = LatticeMaker(1.42, "GRA", 1, 1, 1)
    # lat.compute()
    # lat.box[2, 2] += 20

    # potential = LammpsPotential(
    #     """pair_style airebo 3.0
    #    pair_coeff * * D:\Study\Gra-Al\potential_test\phonon\graphene\phonon_interface\CH.airebo C"""
    # )
    # "0.0 0.0 0.0 0.3333333333 0.3333333333 0.0 0.5 0.0 0.0 0.0 0.0 0.0",
    # "$\Gamma$ K M $\Gamma$",
    system = mp.System(r"D:\Study\Gra-Al\potential_test\phonon\alc\min.data")
    print(system)
    potential = NEP(r"D:\Study\Gra-Al\potential_test\phonon\graphene\nep.txt")
    # system = System(r"/mnt/d/Study/Gra-Al/potential_test/phonon/alc/POSCAR")
    # potential = LammpsPotential(
    #     """pair_style nep /mnt/d/Study/Gra-Al/potential_test/total/nep.txt
    #        pair_coeff * *"""
    # )
    # potential = NEP(r"D:\Study\Gra-Al\potential_test\phonon\aluminum\nep.txt")
    # "0.0 0.0 0.0 0.5 0.5 0.5 0.8244 0.1755 0.5 0.5 0.0 0.0 0.0 0.0 0.0 0.3377 -0.337 0.0 0.5 0.0 0.5 0.0 0.0 0.0",
    # "$\Gamma$ T $H_2$ L $\Gamma$ $S_0$ F $\Gamma$",
    pho = Phonon(
        "0.0 0.0 0.0 0.5 0.5 0.5 0.8244 0.1755 0.5 0.5 0.0 0.0 0.0 0.0 0.0 0.3377 -0.337 0.0 0.5 0.0 0.5 0.0 0.0 0.0",
        "$\Gamma$ T $H_2$ L $\Gamma$ $S_0$ F $\Gamma$",
        potential,
        system.pos,
        system.box,
        elements_list=["Al", "C"],
        type_list=system.data["type"].to_numpy(),
        symprec=1e-3,
        displacement=0.01,
        cutoff_radius=15.0,
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
    pho.plot_dispersion()
    # pho.compute_thermal(0, 50, 1000, (30, 30, 30))
    # pho.plot_thermal()
    # mp.pltset(**{"xtick.major.width":1., "ytick.major.width":1., "axes.linewidth":1., 'xtick.minor.visible':False, 'ytick.minor.visible':False})
    # fig, ax = mp.set_figure(figsize=(10, 8), figdpi=300)
    # fig, ax, line1 = pho.plot_dispersion(fig, ax, )
    # fig, ax, line2 = pho.plot_dispersion(fig, ax, linestyle='--', c='green', linewidth=1.6, alpha=0.6, filename=r"D:\Study\Gra-Al\init_data\cp2k_test\band_data\graphene\band.dat")
    # # fig, ax, line1 = pho.plot_dispersion(fig, ax, color='k', show=False)
    # # fig, ax, line2 = pho.plot_dispersion(fig, ax, color='r', linestyle='-.', show=False)
    # ax.legend([line1, line2], ['NEP', 'DFT'])
    # ax.set_ylim(0, 50)
    plt.show()
    # fig, ax = pho.plot_dispersion_from_band_data(r"D:\Study\Gra-Al\init_data\cp2k_test\band_data\graphene\band.dat", labels=pho.labels,
    #                                    fig=fig, ax=ax, color='b')
    # plt.show()
    # fig.show()
    # fig.savefig("test.png")
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
