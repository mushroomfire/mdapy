# Copyright (c) 2022-2025, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

from __future__ import annotations
from mdapy import _bond_analysis
from mdapy.box import Box
import polars as pl
import numpy as np
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes


class BondAnalysis:
    """
    Analyze bond length and bond angle distributions in atomic systems.

    This class computes histograms of bond lengths (distances between bonded
    atoms) and bond angles (angles formed by atomic triplets i-j-k where i is
    the central atom). These distributions are useful for characterizing local
    structure and identifying coordination patterns.

    Parameters
    ----------
    data : pl.DataFrame
        Atomic data containing positions. Must have columns: 'x', 'y', 'z'.
    box : Box
        Simulation box object with lattice parameters and boundaries.
    rc : float
        Cutoff radius for bond analysis in Angstroms. Pairs within this distance
        are considered bonded.
    nbin : int
        Number of histogram bins for both bond length (0 to rc) and bond angle
        (0 to 180 degrees) distributions.
    verlet_list : np.ndarray
        Verlet neighbor list, shape (N, max_neigh).
    distance_list : np.ndarray
        Neighbor distances, shape (N, max_neigh).
    neighbor_number : np.ndarray
        Number of neighbors for each atom, shape (N,).

    Attributes
    ----------
    bond_length_distribution : np.ndarray
        Histogram of bond lengths, shape (nbin,). Available after :meth:`compute`.
    bond_angle_distribution : np.ndarray
        Histogram of bond angles, shape (nbin,). Available after :meth:`compute`.
    r_length : np.ndarray
        Bin centers for bond length histogram in Angstroms, shape (nbin,).
        Available after :meth:`compute`.
    r_angle : np.ndarray
        Bin centers for bond angle histogram in degrees, shape (nbin,).
        Available after :meth:`compute`.
    rc : float
        Cutoff radius used.
    nbin : int
        Number of bins used.

    Notes
    -----
    **Bond Length Distribution:**

    For each atom i, counts all neighbors j within cutoff rc. Each distance
    r_ij is binned into a histogram with bins from 0 to rc.

    **Bond Angle Distribution:**

    For each atom i (central atom), considers all pairs of neighbors (j, k)
    within cutoff rc. The angle θ is calculated as:

    .. math::

        \\theta = \\arccos\\left(\\frac{\\vec{r}_{ij} \\cdot \\vec{r}_{ik}}
                  {|\\vec{r}_{ij}| |\\vec{r}_{ik}|}\\right)

    where :math:`\\vec{r}_{ij} = \\vec{r}_j - \\vec{r}_i` is the distance vector
    from atom i to atom j. Angles are binned from 0° to 180°.

    Examples
    --------
    Analyze bond structure in a system:

    >>> from mdapy import System
    >>> system = System("structure.xyz")
    >>> system.build_neighbor(rc=5.0, max_neigh=50)
    >>> from mdapy.bond_analysis import BondAnalysis
    >>> ba = BondAnalysis(
    ...     system.data,
    ...     system.box,
    ...     rc=5.0,
    ...     nbin=50,
    ...     verlet_list=system.verlet_list,
    ...     distance_list=system.distance_list,
    ...     neighbor_number=system.neighbor_number,
    ... )
    >>> ba.compute()
    >>> # Plot distributions
    >>> fig1, ax1 = ba.plot_bond_length_distribution()
    >>> fig2, ax2 = ba.plot_bond_angle_distribution()
    """

    def __init__(
        self,
        data: pl.DataFrame,
        box: Box,
        rc: float,
        nbin: int,
        verlet_list: np.ndarray,
        distance_list: np.ndarray,
        neighbor_number: np.ndarray,
    ):
        self.data = data
        self.box = box
        self.rc = rc
        self.nbin = nbin
        self.verlet_list = verlet_list
        self.distance_list = distance_list
        self.neighbor_number = neighbor_number

    def compute(self):
        """
        Compute bond length and bond angle distributions.

        This method calculates histograms of all bond lengths and bond angles
        in the system. Results are stored in :attr:`bond_length_distribution`
        and :attr:`bond_angle_distribution`, with corresponding bin centers
        in :attr:`r_length` and :attr:`r_angle`.

        Both distributions use equal bin spacing: bond lengths from 0 to rc,
        bond angles from 0° to 180°.
        """
        delta_r = self.rc / self.nbin
        delta_theta = 180.0 / self.nbin
        self.bond_length_distribution = np.zeros(self.nbin, np.int32)
        self.bond_angle_distribution = np.zeros(self.nbin, np.int32)
        _bond_analysis.compute_bond(
            self.data["x"].to_numpy(allow_copy=False),
            self.data["y"].to_numpy(allow_copy=False),
            self.data["z"].to_numpy(allow_copy=False),
            self.box.box,
            self.box.origin,
            self.box.boundary,
            self.verlet_list,
            self.distance_list,
            self.neighbor_number,
            self.bond_length_distribution,
            self.bond_angle_distribution,
            delta_r,
            delta_theta,
            self.rc,
            self.nbin,
        )
        r = np.linspace(0, self.rc, self.nbin + 1)
        self.r_length = (r[1:] + r[:-1]) / 2
        r = np.linspace(0, 180.0, self.nbin + 1)
        self.r_angle = (r[1:] + r[:-1]) / 2

    def plot_bond_length_distribution(
        self, fig: Optional[Figure] = None, ax: Optional[Axes] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot bond length distribution.

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

        """
        assert hasattr(self, "bond_length_distribution"), "call compute first."
        if fig is None and ax is None:
            from mdapy import set_figure

            fig, ax = set_figure()

        ax.plot(self.r_length, self.bond_length_distribution)
        ax.fill_between(self.r_length, self.bond_length_distribution, alpha=0.3)
        ax.set_xlabel(r"Bond length ($\mathregular{\AA}$)")
        ax.set_ylabel("Count")
        ax.set_xlim(0, self.rc)
        ax.set_ylim(0, self.bond_length_distribution.max() * 1.05)

        return fig, ax

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

        """
        assert hasattr(self, "bond_angle_distribution"), "call compute first."
        if fig is None and ax is None:
            from mdapy import set_figure

            fig, ax = set_figure()

        ax.plot(self.r_angle, self.bond_angle_distribution)
        ax.fill_between(self.r_angle, self.bond_angle_distribution, alpha=0.3)
        ax.set_xlabel(r"Bond angle ($\mathregular{\theta}$)")
        ax.set_ylabel("Count")
        ax.set_xlim(0, 180)
        ax.set_ylim(0, self.bond_angle_distribution.max() * 1.05)
        ax.set_xticks([0, 60, 120, 180])

        return fig, ax


if __name__ == "__main__":
    pass
