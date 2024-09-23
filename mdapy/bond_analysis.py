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


if __name__ == "__main__":
    ti.init()
    from mdapy import System

    system = System(r"D:\Study\water\0.1MPa\relax\dump.99.xyz")

    BA = BondAnalysis(system.pos, system.box, system.boundary, 1.3, 100)
    BA.compute()

    BA.plot_bond_length_distribution()
    BA.plot_bond_angle_distribution()
