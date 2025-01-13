# Copyright (c) 2022-2024, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
from math import pi

try:
    from tool_function import _check_repeat_cutoff
    from box import init_box, _pbc, _pbc_rec
    from replicate import Replicate
    from neighbor import Neighbor
    from plotset import set_figure
except Exception:
    from .tool_function import _check_repeat_cutoff
    from .box import init_box, _pbc, _pbc_rec
    from .replicate import Replicate
    from .neighbor import Neighbor
    from .plotset import set_figure


@ti.data_oriented
class BondAnalysis:
    """This class calculates the distribution of bond length and angle based on a given cutoff distance.

    Args:
        pos (np.ndarray): (:math:`N_p, 3`) particles positions.
        box (np.ndarray): (:math:`3, 2`) or (:math:`4, 3`) system box.
        boundary (list): boundary conditions, 1 is periodic and 0 is free boundary.
        rc (float): cutoff distance.
        nbins (int): number of bins.
        verlet_list (np.ndarray, optional): (:math:`N_p, max\_neigh`) verlet_list[i, j] means j atom is a neighbor of i atom if j > -1.
        distance_list (np.ndarray, optional): (:math:`N_p, max\_neigh`) distance_list[i, j] means distance between i and j atom.
        neighbor_number (np.ndarray, optional): (:math:`N_p`) neighbor atoms number.

    Outputs:
        - **bond_length_distribution** (np.ndarray) - (nbins) bond length distribution.
        - **bond_angle_distribution** (np.ndarray) - (nbins) bond angle distribution.
        - **r_length** (np.ndarray) - (nbins) bond length (x axis).
        - **r_angle** (np.ndarray) - (nbins) bond angle (x axis).
    """

    def __init__(
        self,
        pos,
        box,
        boundary,
        rc,
        nbins,
        verlet_list=None,
        distance_list=None,
        neighbor_number=None,
    ) -> None:
        self.rc = rc
        self.nbins = nbins
        box, inverse_box, rec = init_box(box)
        repeat = _check_repeat_cutoff(box, boundary, self.rc)
        if pos.dtype != np.float64:
            pos = pos.astype(np.float64)

        self.old_N = None
        if sum(repeat) == 3:
            self.pos = pos
            self.box, self.inverse_box, self.rec = box, inverse_box, rec
        else:
            self.old_N = pos.shape[0]
            repli = Replicate(pos, box, *repeat)
            repli.compute()
            self.pos = repli.pos
            self.box, self.inverse_box, self.rec = init_box(repli.box)

        self.box_length = ti.Vector([np.linalg.norm(self.box[i]) for i in range(3)])
        self.boundary = ti.Vector([int(boundary[i]) for i in range(3)])
        self.verlet_list = verlet_list
        self.distance_list = distance_list
        self.neighbor_number = neighbor_number

    @ti.kernel
    def _compute(
        self,
        pos: ti.types.ndarray(element_dim=1),
        verlet_list: ti.types.ndarray(),
        distance_list: ti.types.ndarray(),
        neighbor_number: ti.types.ndarray(),
        bond_length_distribution: ti.types.ndarray(),
        bond_angle_distribution: ti.types.ndarray(),
        box: ti.types.ndarray(element_dim=1),
        inverse_box: ti.types.ndarray(element_dim=1),
        delta_r: float,
        delta_theta: float,
        rc: float,
        nbins: int,
    ):
        N = verlet_list.shape[0]
        delta_r_inv = 1.0 / delta_r
        delta_theta_inv = 1.0 / delta_theta
        for i in range(N):
            i_neigh = neighbor_number[i]
            for jj in range(i_neigh):
                j = verlet_list[i, jj]
                if j > i:
                    r = distance_list[i, jj]
                    if r <= rc:
                        index = ti.floor(r * delta_r_inv, ti.i32)
                        if index > nbins - 1:
                            index = nbins - 1
                        bond_length_distribution[index] += 1

            for jj in range(i_neigh):
                j = verlet_list[i, jj]
                r_ij = distance_list[i, jj]
                if r_ij <= rc:
                    for kk in range(jj + 1, i_neigh):
                        k = verlet_list[i, kk]
                        r_ik = distance_list[i, kk]
                        if r_ik <= rc:
                            rij = pos[j] - pos[i]
                            rik = pos[k] - pos[i]
                            if ti.static(self.rec):
                                rij = _pbc_rec(rij, self.boundary, self.box_length)
                                rik = _pbc_rec(rik, self.boundary, self.box_length)
                            else:
                                rij = _pbc(rij, self.boundary, box, inverse_box)
                                rik = _pbc(rik, self.boundary, box, inverse_box)
                            theta = ti.acos((rij @ rik) / (r_ij * r_ik)) * 180 / pi
                            index = ti.floor(theta * delta_theta_inv, ti.i32)
                            if index > nbins - 1:
                                index = nbins - 1
                            bond_angle_distribution[index] += 1

    def compute(
        self,
    ):
        """Do the real computation."""
        if (
            self.verlet_list is None
            or self.neighbor_number is None
            or self.distance_list is None
        ):
            neigh = Neighbor(self.pos, self.box, self.rc, self.boundary)
            neigh.compute()
            self.verlet_list, self.distance_list, self.neighbor_number = (
                neigh.verlet_list,
                neigh.distance_list,
                neigh.neighbor_number,
            )

        delta_r = self.rc / self.nbins
        delta_theta = 180.0 / self.nbins
        self.bond_length_distribution = np.zeros(self.nbins, int)
        self.bond_angle_distribution = np.zeros(self.nbins, int)

        self._compute(
            self.pos,
            self.verlet_list,
            self.distance_list,
            self.neighbor_number,
            self.bond_length_distribution,
            self.bond_angle_distribution,
            self.box,
            self.inverse_box,
            delta_r,
            delta_theta,
            self.rc,
            self.nbins,
        )

        r = np.linspace(0, self.rc, self.nbins + 1)
        self.r_length = (r[1:] + r[:-1]) / 2
        r = np.linspace(0, 180.0, self.nbins + 1)
        self.r_angle = (r[1:] + r[:-1]) / 2

    def plot_bond_length_distribution(
        self,
    ):
        assert hasattr(self, "bond_length_distribution"), "call compute first."
        fig, ax = set_figure(figsize=(10, 8), figdpi=150, use_pltset=True)

        ax.plot(self.r_length, self.bond_length_distribution)
        ax.fill_between(self.r_length, self.bond_length_distribution, alpha=0.3)
        ax.set_xlabel("Bond length ($\mathregular{\AA}$)")
        ax.set_ylabel("Count")
        ax.set_xlim(0, self.rc)
        ax.set_ylim(0, self.bond_length_distribution.max() * 1.05)
        fig.tight_layout()
        plt.show()
        return fig, ax

    def plot_bond_angle_distribution(
        self,
    ):
        assert hasattr(self, "bond_angle_distribution"), "call compute first."
        fig, ax = set_figure(figsize=(10, 8), figdpi=150, use_pltset=True)

        ax.plot(self.r_angle, self.bond_angle_distribution)
        ax.fill_between(self.r_angle, self.bond_angle_distribution, alpha=0.3)
        ax.set_xlabel(r"Bond angle ($\mathregular{\theta}$)")
        ax.set_ylabel("Count")
        ax.set_xlim(0, 180)
        ax.set_ylim(0, self.bond_angle_distribution.max() * 1.05)
        ax.set_xticks([0, 60, 120, 180])
        fig.tight_layout()
        plt.show()
        return fig, ax


@ti.data_oriented
class AngularDistributionFunctions:
    def __init__(
        self,
        pos,
        box,
        boundary,
        rc,
        nbins,
        type_list,
        verlet_list=None,
        distance_list=None,
        neighbor_number=None,
    ):
        self.pair_list = np.array([i.split("-") for i in rc.keys()], int)
        self.rc_list = np.array(list(rc.values()))
        assert self.rc_list.shape[1] == 4, "rc should be a list of 4 floats."
        assert self.pair_list.shape[1] == 3, "pair_list should be a list of 3 integers."
        for i in np.unique(self.pair_list):
            assert i in type_list, f"{i}-type is not in type_list."
        assert all(self.rc_list[:, 1] >= self.rc_list[:, 0]), (
            "rc[0] should be smaller than rc[1]."
        )
        assert all(self.rc_list[:, 3] >= self.rc_list[:, 2]), (
            "rc[2] should be smaller than rc[3]."
        )
        assert all(self.rc_list[:, 0] >= 0.0), "rc[0] should be larger than 0."
        assert all(self.rc_list[:, 2] >= 0.0), "rc[2] should be larger than 0."
        self.Npair = self.pair_list.shape[0]
        self.rc_max = self.rc_list.max()
        box, inverse_box, rec = init_box(box)
        repeat = _check_repeat_cutoff(box, boundary, self.rc_max)
        if pos.dtype != np.float64:
            pos = pos.astype(np.float64)

        self.old_N = None
        if sum(repeat) == 3:
            self.pos = pos
            self.box, self.inverse_box, self.rec = box, inverse_box, rec
        else:
            self.old_N = pos.shape[0]
            repli = Replicate(pos, box, *repeat)
            repli.compute()
            self.pos = repli.pos
            self.box, self.inverse_box, self.rec = init_box(repli.box)

        self.box_length = ti.Vector([np.linalg.norm(self.box[i]) for i in range(3)])
        self.boundary = ti.Vector([int(boundary[i]) for i in range(3)])

        self.nbins = nbins
        self.type_list = type_list
        self.verlet_list = verlet_list
        self.distance_list = distance_list
        self.neighbor_number = neighbor_number

    @ti.kernel
    def _compute(
        self,
        pos: ti.types.ndarray(element_dim=1),
        verlet_list: ti.types.ndarray(),
        distance_list: ti.types.ndarray(),
        neighbor_number: ti.types.ndarray(),
        bond_angle_distribution: ti.types.ndarray(),
        box: ti.types.ndarray(element_dim=1),
        inverse_box: ti.types.ndarray(element_dim=1),
        delta_theta: float,
        rc_list: ti.types.ndarray(),
        pair_list: ti.types.ndarray(),
        type_list: ti.types.ndarray(),
        nbins: int,
    ):
        N = verlet_list.shape[0]
        delta_theta_inv = 1.0 / delta_theta
        # ti.loop_config(serialize=True)
        for i in range(N):
            itype = type_list[i]
            for m in range(self.Npair):
                if itype == pair_list[m, 0]:
                    i_neigh = neighbor_number[i]
                    for jj in range(i_neigh):
                        j = verlet_list[i, jj]
                        jtype = type_list[j]
                        if jtype == pair_list[m, 1]:
                            
                            r_ij = distance_list[i, jj]
                            if r_ij <= rc_list[m, 1] and r_ij >= rc_list[m, 0]:

                                for kk in range(jj+1, i_neigh):
                                    k = verlet_list[i, kk]
                                    ktype = type_list[k]
                                    if ktype == pair_list[m, 2]:
                                        r_ik = distance_list[i, kk]
                                        if (
                                            r_ik <= rc_list[m, 3]
                                            and r_ik >= rc_list[m, 2]
                                        ):
                                            rij = pos[j] - pos[i]
                                            rik = pos[k] - pos[i]
                                            if ti.static(self.rec):
                                                rij = _pbc_rec(
                                                    rij, self.boundary, self.box_length
                                                )
                                                rik = _pbc_rec(
                                                    rik, self.boundary, self.box_length
                                                )
                                            else:
                                                rij = _pbc(
                                                    rij, self.boundary, box, inverse_box
                                                )
                                                rik = _pbc(
                                                    rik, self.boundary, box, inverse_box
                                                )
                                            theta = (
                                                ti.acos((rij @ rik) / (r_ij * r_ik))
                                                * 180
                                                / pi
                                            )
                                            index = ti.floor(
                                                theta * delta_theta_inv, ti.i32
                                            )

                                            if index > nbins - 1:
                                                index = nbins - 1
                                            bond_angle_distribution[m, index] += 1

    def compute(
        self,
    ):
        if (
            self.verlet_list is None
            or self.distance_list is None
            or self.neighbor_number is None
        ):
            neigh = Neighbor(self.pos, self.box, self.rc_max, self.boundary)
            neigh.compute()
            self.verlet_list, self.distance_list, self.neighbor_number = (
                neigh.verlet_list,
                neigh.distance_list,
                neigh.neighbor_number,
            )

        delta_theta = 180.0 / self.nbins
        self.bond_angle_distribution = np.zeros((self.Npair, self.nbins), int)
        from time import time
        start = time()
        self._compute(
            self.pos,
            self.verlet_list,
            self.distance_list,
            self.neighbor_number,
            self.bond_angle_distribution,
            self.box,
            self.inverse_box,
            delta_theta,
            self.rc_list,
            self.pair_list,
            self.type_list,
            self.nbins,
        )
        end = time()
        print(f"Time: {end - start} s.")
        r = np.linspace(0, 180.0, self.nbins + 1)
        self.r_angle = (r[1:] + r[:-1]) / 2
    
    def plot_bond_angle_distribution(
        self, type_name=None, fig=None, ax=None
    ):
        assert hasattr(self, "bond_angle_distribution"), "call compute first."
        if fig is None and ax is None:
            fig, ax = set_figure(figsize=(10, 8), figdpi=150, use_pltset=True)
        if type_name is not None:
            assert np.unique(self.pair_list).max() <= len(type_name), "type_name is not enough."
            for m in range(self.Npair):
                itype, jtype, ktype = self.pair_list[m]
                i, j, k = type_name[itype-1], type_name[jtype-1], type_name[ktype-1]
                total = self.bond_angle_distribution[m].sum()
                ax.plot(self.r_angle, self.bond_angle_distribution[m]/total, 'o', label=f"{j}-{i}-{k}")
        else:
            for i in range(self.Npair):
                ax.plot(self.r_angle, self.bond_angle_distribution[i], label=f"{self.pair_list[i, 1]}-{self.pair_list[i, 0]}-{self.pair_list[i, 2]}")
        
        #ax.fill_between(self.r_angle, self.bond_angle_distribution, alpha=0.3)
        ax.set_xlabel(r"Bond angle ($\mathregular{\theta}$)")
        ax.set_ylabel("Count")
        ax.set_xlim(0, 180)
        # ax.set_ylim(0, self.bond_angle_distribution.max() * 1.05)
        ax.set_xticks([0, 60, 120, 180])
        ax.legend()
        fig.tight_layout()
        plt.show()
        return fig, ax


if __name__ == "__main__":
    ti.init(offline_cache=True)
    from mdapy import System
    import polars as pl

    system = System(r"C:\Users\HerrWu\Desktop\adf\model.xyz")
    system.update_data(system.data.with_columns(
        pl.col('type_name').replace_strict({'H':1, 'O':2}).alias('type')
    ))

    # system = System(r"C:\Users\HerrWu\Desktop\adf\nvt_heat_Mg+O2\nvt_heat_Mg+O2\dump\dump.00500.dump")
    # system = System(r"C:\Users\HerrWu\Desktop\adf\gcmc_Mg\gcmc_Mg\dump\dump.00500.dump")
    # system.replicate(10, 10, 10)

    adf = AngularDistributionFunctions(
        system.pos,
        system.box,
        system.boundary,
        {
         "2-1-1": [0.0, 1.2, 0.0, 1.2],
         },
        30,
        system.data['type'].to_numpy(),
    )
    adf.compute()
    # print(adf.r_angle)

    # fig, ax = set_figure(figsize=(10, 8), figdpi=150, use_pltset=True)
    # total = adf.bond_angle_distribution[0].sum()
    # # print(total)
    # adf_l = np.loadtxt(r"C:\Users\HerrWu\Desktop\adf\adf.dat", skiprows=4)
    # ax.plot(adf_l[:, 1], adf_l[:, 2], 'o', label="LAMMPS")
    # x, y = adf_l[:, 1], adf_l[:, 2]
    # print(np.trapz(y, x))
    # delta = adf.r_angle[1] - adf.r_angle[0]
    # x1, y1 = adf.r_angle, adf.bond_angle_distribution[0]/total/delta
    # print(np.trapz(y1, x1))
    # ax.plot(x1, y1, '-*', label="O-Mg-O")
    # ax.legend()
    # plt.show()
    #adf.plot_bond_angle_distribution(['H', 'O'], fig, ax)
    # print(np.cumsum(adf.bond_angle_distribution[1]/system.data.filter(pl.col('type') == 2).shape[0]))
    # print(adf.verlet_list[0])
    # print(adf.distance_list)
    # print(adf.type_list)
    # adf.plot_bond_angle_distribution()

    # BA = BondAnalysis(system.pos, system.box, system.boundary, 1.3, 100)
    # BA.compute()

    # BA.plot_bond_length_distribution()
    # BA.plot_bond_angle_distribution()

    # rc = {
    #     "1-1-1": [0.0, 1.2, 0.0, 1.4],
    #     "1-1-2": [0.0, 1.2, 0.0, 1.4],
    #     "1-2-1": [0.0, 1.2, 0.0, 1.4],
    #     "1-2-2": [0.0, 1.2, 0.0, 1.4],
    #     "2-1-2": [0.0, 1.2, 0.0, 1.4],
    # }
    # rc = {
    #     "1-1-1": [0.0, 1.2, 0.0, 1.4],
    # }
    # pair_list = np.array([i.split("-") for i in rc.keys()], int)
    # rc_list = np.array(list(rc.values()))
    # print(pair_list)
    # print(rc_list)
    # print(np.unique(pair_list))
