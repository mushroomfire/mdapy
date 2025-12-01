# Copyright (c) 2022-2025, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

from __future__ import annotations
from mdapy import _bond_analysis
from mdapy.box import Box
import polars as pl
import numpy as np
from typing import TYPE_CHECKING, Optional, Tuple, Dict, List

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes


class AngularDistributionFunction:
    """
    Calculate Angular Distribution Function (ADF) for bond angle analysis.

    The angular distribution function quantifies the distribution of bond angles
    between atomic triplets (i-j-k), where i is the central atom. This is useful
    for characterizing local atomic structure and coordination environments.

    Parameters
    ----------
    data : pl.DataFrame
        Atomic data containing positions and element information.
        Must have columns: 'x', 'y', 'z', 'element'.
    box : Box
        Simulation box object with lattice parameters and boundaries.
    rc_dict : dict of str to list of float
        Dictionary mapping triplet patterns to cutoff radii.
        Format: {'A-B-C': [rc_AB_min, rc_AB_max, rc_AC_min, rc_AC_max]}
        where A, B, C are element symbols and A is the central atom.
        Example: {'O-H-H': [0, 2.0, 0, 2.0]} for water molecules.
    nbin : int
        Number of bins for angle discretization (0-180 degrees).
    verlet_list : np.ndarray
        Verlet neighbor list, shape (N, max_neigh).
    distance_list : np.ndarray
        Neighbor distances, shape (N, max_neigh).
    neighbor_number : np.ndarray
        Number of neighbors for each atom, shape (N,).

    Attributes
    ----------
    bond_angle_distribution : np.ndarray
        Bond angle distribution for each triplet type, shape (Npair, nbin).
        Available after calling :meth:`compute`.
    r_angle : np.ndarray
        Angle bin centers in degrees, shape (nbin,).
        Available after calling :meth:`compute`.
    ele_unique : list of str
        Unique element symbols in the system, sorted alphabetically.
    pair_list : np.ndarray
        Type indices for each triplet pattern, shape (Npair, 3).
    rc_list : np.ndarray
        Cutoff radii for each triplet pattern, shape (Npair, 4).

    Notes
    -----
    The angular distribution function is calculated by:

    1. For each central atom i of type A
    2. Find neighbors j of type B within [rc_AB_min, rc_AB_max]
    3. Find neighbors k of type C within [rc_AC_min, rc_AC_max]
    4. Calculate angle θ between j-i-k bonds
    5. Histogram the angles into bins

    Examples
    --------
    Calculate ADF for water molecules (H-O-H angle):

    >>> from mdapy import System
    >>> system = System("water.xyz")
    >>> system.build_neighbor(rc=3.0, max_neigh=50)
    >>> from mdapy.angular_distribution_function import AngularDistributionFunction
    >>> adf = AngularDistributionFunction(
    ...     system.data,
    ...     system.box,
    ...     rc_dict={"O-H-H": [0, 2.0, 0, 2.0]},  # O is central atom
    ...     nbin=40,
    ...     verlet_list=system.verlet_list,
    ...     distance_list=system.distance_list,
    ...     neighbor_number=system.neighbor_number,
    ... )
    >>> adf.compute()
    >>> fig, ax = adf.plot_bond_angle_distribution()
    """

    def __init__(
        self,
        data: pl.DataFrame,
        box: Box,
        rc_dict: Dict[str, List[float]],
        nbin: int,
        verlet_list: np.ndarray,
        distance_list: np.ndarray,
        neighbor_number: np.ndarray,
    ):
        self.data = data
        assert "element" in data.columns
        self.box = box
        self.ele_unique = self.data["element"].unique().sort().to_list()
        pair_list = []
        for i in rc_dict.keys():
            a, b, c = i.split("-")
            assert a in self.ele_unique
            assert b in self.ele_unique
            assert c in self.ele_unique
            pair_list.append(
                [
                    self.ele_unique.index(a),
                    self.ele_unique.index(b),
                    self.ele_unique.index(c),
                ]
            )

        self.pair_list = np.array(pair_list, np.int32)
        self.rc_list = np.array(list(rc_dict.values()), float)

        assert self.rc_list.shape[1] == 4, "rc should be a list of 4 floats."
        self.nbin = nbin
        self.verlet_list = verlet_list
        self.distance_list = distance_list
        self.neighbor_number = neighbor_number

    def compute(self):
        """
        Compute the angular distribution function.

        This method calculates the bond angle distribution for all specified
        triplet patterns. Results are stored in :attr:`bond_angle_distribution`
        and :attr:`r_angle`.
        """
        Npair = self.pair_list.shape[0]
        delta_theta = 180.0 / self.nbin
        self.bond_angle_distribution = np.zeros((Npair, self.nbin), np.int32)
        ele2type = {j: i for i, j in enumerate(self.ele_unique)}

        type_list = self.data.with_columns(
            pl.col("element")
            .replace_strict(ele2type, return_dtype=pl.Int32)
            .rechunk()
            .alias("type")
        )["type"].to_numpy(allow_copy=False)

        _bond_analysis.compute_adf(
            self.data["x"].to_numpy(allow_copy=False),
            self.data["y"].to_numpy(allow_copy=False),
            self.data["z"].to_numpy(allow_copy=False),
            self.box.box,
            self.box.origin,
            self.box.boundary,
            self.verlet_list,
            self.distance_list,
            self.neighbor_number,
            delta_theta,
            self.rc_list,
            self.pair_list,
            type_list,
            self.nbin,
            self.bond_angle_distribution,
        )
        r = np.linspace(0, 180.0, self.nbin + 1)
        self.r_angle = (r[1:] + r[:-1]) / 2

    def plot_bond_angle_distribution(
        self, fig: Optional[Figure] = None, ax: Optional[Axes] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot bond angle distribution.

        Parameters
        ----------
        fig : Figure, optional
            Existing matplotlib figure. If None, a new figure is created.
        ax : Axes, optional
            Existing matplotlib axes. If None, new axes are created.

        Returns
        -------
        fig : Figure
            Matplotlib figure object.
        ax : Axes
            Matplotlib axes object.

        Notes
        -----
        The plot shows bond angle distribution as a function of angle (0-180°)
        for all triplet patterns specified in rc_dict. Each pattern is shown
        as a separate line with markers.

        """
        assert hasattr(self, "bond_angle_distribution"), "call compute first."
        if fig is None and ax is None:
            from mdapy import set_figure

            fig, ax = set_figure()

        for m in range(self.pair_list.shape[0]):
            itype, jtype, ktype = self.pair_list[m]
            i, j, k = (
                self.ele_unique[itype],
                self.ele_unique[jtype],
                self.ele_unique[ktype],
            )
            # total = self.bond_angle_distribution[m].sum()
            ax.plot(
                self.r_angle,
                self.bond_angle_distribution[m],
                "-o",
                label=f"{j}-{i}-{k}",
            )
        ax.legend()
        ax.set_xlabel(r"Bond angle ($\mathregular{\theta}$)")
        ax.set_ylabel("Count")
        ax.set_xlim(0, 180)
        ax.set_ylim(0, self.bond_angle_distribution.max() * 1.05)
        ax.set_xticks([0, 60, 120, 180])

        return fig, ax


if __name__ == "__main__":
    from mdapy import System
    # import matplotlib.pyplot as plt

    system = System(
        r"C:\Users\HerrW\Desktop\03-Demo-MD\03-Demo-MD\CMD\Density\model.xyz"
    )

    adf = system.cal_angular_distribution_function(
        {
            "O-H-H": [0, 2.0, 0, 2.0],
            "H-H-H": [0, 2.0, 0, 2.0],
            "O-O-O": [0, 2.0, 0, 2.0],
        },
        40,
    )
    print(adf.bond_angle_distribution)

    # adf.plot_bond_angle_distribution()
    # plt.show()
