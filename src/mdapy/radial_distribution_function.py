# Copyright (c) 2022-2025, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
from __future__ import annotations
from typing import Optional, Tuple, List, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray
from mdapy import _rdf
from mdapy.box import Box

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes


class RadialDistributionFunction:
    r"""Calculate the radial distribution function (RDF) for atomic systems.

    The radial distribution function g(r) describes the probability of finding
    an atom at distance r from a reference atom, normalized by the probability
    expected for a completely random distribution at the same density.

    For multi-component systems, the total RDF is a weighted sum of partial RDFs:

    .. math::
        g(r) = c_{\alpha}^2 g_{\alpha\alpha}(r) + 2c_{\alpha}c_{\beta}g_{\alpha\beta}(r) + c_{\beta}^2 g_{\beta\beta}(r)

    where :math:`c_{\alpha}` and :math:`c_{\beta}` denote the concentration of
    atom types α and β, respectively, and :math:`g_{\alpha\beta}(r) = g_{\beta\alpha}(r)`.

    The RDF is computed as:

    .. math::
        g_{\alpha\beta}(r) = \frac{V}{N_{\alpha}N_{\beta}} \sum_{i \in \alpha} \sum_{j \in \beta} \delta(r - r_{ij})

    where V is the system volume, :math:`N_{\alpha}` and :math:`N_{\beta}` are
    the number of atoms of types α and β, and :math:`r_{ij}` is the distance
    between atoms i and j.

    Parameters
    ----------
    rc : float
        Cutoff distance for RDF calculation (in Angstroms).
    nbin : int
        Number of bins for histogram.
    box : Box
        System box object containing boundary information.
    verlet_list : NDArray[np.int32]
        Shape (N_particles, max_neigh). Verlet neighbor list where
        verlet_list[i, j] gives the index of the j-th neighbor of particle i.
        Value of -1 indicates no neighbor at that position.
    distance_list : NDArray[np.float64]
        Shape (N_particles, max_neigh). Distance list where distance_list[i, j]
        gives the distance between particle i and its j-th neighbor.
    neighbor_number : NDArray[np.int32]
        Shape (N_particles,). Number of neighbors for each particle.
    type_list : NDArray[np.int32]
        Shape (N_particles,). Atom type indices for each particle.

    Attributes
    ----------
    r : NDArray[np.float64]
        Shape (nbin,). Distance values at bin centers.
    g_total : NDArray[np.float64]
        Shape (nbin,). Total (averaged) radial distribution function.
    Ntype : int
        Number of distinct atom types in the system.
    g : NDArray[np.float64]
        Shape (Ntype, Ntype, nbin). Partial RDFs for each pair of atom types.
        g[i, j, :] gives the RDF between type i and type j atoms.
    vol : float
        Volume of the simulation box.
    N : int
        Total number of particles in the system.

    """

    def __init__(
        self,
        rc: float,
        nbin: int,
        box: Box,
        verlet_list: NDArray[np.int32],
        distance_list: NDArray[np.float64],
        neighbor_number: NDArray[np.int32],
        type_list: NDArray[np.int32],
    ) -> None:
        self.rc = rc
        self.nbin = nbin
        self.box = box
        self.verlet_list = verlet_list
        self.distance_list = distance_list
        self.neighbor_number = neighbor_number
        self.vol = self.box.volume

        # Process type list
        type_list = np.asarray(type_list, np.int32)
        unitype = np.sort(np.unique(type_list))
        self.Ntype = len(unitype)
        self._old_type = unitype

        # Remap types to consecutive integers starting from 0
        new_type = type_list.copy()
        for i, j in enumerate(unitype):
            new_type[type_list == j] = i
        self.type_list = new_type
        self.N = self.verlet_list.shape[0]

    def compute(self) -> None:
        """Compute the radial distribution function.

        This method calculates both the total RDF and partial RDFs for all
        pair combinations of atom types. The results are stored in the
        `g_total`, `g`, and `r` attributes.

        """
        r = np.linspace(0, self.rc, self.nbin + 1)
        # Normalization constant: shell volume divided by total volume
        const = (4.0 * np.pi / 3.0 * (r[1:] ** 3 - r[:-1] ** 3)) / self.vol

        if self.Ntype > 1:
            # Multi-component system
            number_per_type = np.array(
                [len(self.type_list[self.type_list == i]) for i in range(self.Ntype)]
            )
            self.g = np.zeros((self.Ntype, self.Ntype, self.nbin), dtype=np.float64)

            # Calculate partial RDFs using Cython backend
            _rdf._rdf(
                self.verlet_list,
                self.distance_list,
                self.neighbor_number,
                self.type_list,
                self.g,
                self.rc,
                self.nbin,
            )

            self.r = (r[1:] + r[:-1]) / 2
            self.g_total = np.zeros_like(self.r)

            # Sum all partial RDFs to get total RDF
            for i in range(self.Ntype):
                for j in range(self.Ntype):
                    self.g_total += self.g[i, j]

            self.g_total = self.g_total / const / (self.N) ** 2

            # Normalize partial RDFs
            for i in range(self.Ntype):
                for j in range(self.Ntype):
                    self.g[i, j] = (
                        self.g[i, j] / number_per_type[i] / number_per_type[j]
                    )
            self.g = self.g / const
        else:
            # Single-component system
            self.g_total = np.zeros(self.nbin, dtype=np.float64)

            # Calculate RDF using optimized single-species routine
            _rdf._rdf_single_species(
                self.verlet_list,
                self.distance_list,
                self.neighbor_number,
                self.g_total,
                self.rc,
                self.nbin,
            )

            self.g_total = self.g_total / const / (self.N) ** 2
            self.r = (r[1:] + r[:-1]) / 2

            # Store as partial RDF for consistency
            self.g = np.zeros((1, 1, self.nbin), dtype=np.float64)
            self.g[0, 0] = self.g_total

    def plot(self, fig: Optional[Figure] = None, ax: Optional[Axes] = None) -> Tuple:
        """Plot the total (global) radial distribution function.

        Parameters
        ----------
        fig : Optional[Figure]
            Existing matplotlib figure.
        ax : Optional[Axes]
            Existing matplotlib axes.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib figure object.
        ax : matplotlib.axes.Axes
            The matplotlib axes object.

        """
        if fig is None and ax is None:
            from mdapy.plotset import set_figure

            fig, ax = set_figure()
        ax.plot(self.r, self.g_total, "o-", ms=3)
        ax.set_xlabel(r"r ($\AA$)")
        ax.set_ylabel("g(r)")
        ax.set_xlim(0, self.rc)

        return fig, ax

    def plot_partial(
        self,
        elements_list: Optional[List[str]] = None,
        fig: Optional[Figure] = None,
        ax: Optional[Axes] = None,
    ) -> Tuple:
        """Plot partial radial distribution functions for all atom type pairs.

        Parameters
        ----------
        elements_list : list of str, optional
            List of element symbols corresponding to each atom type,
            e.g., ['Al', 'Ni']. If None, numeric labels are used.
            Length must match the number of atom types (Ntype).
        fig : Optional[Figure]
            Existing matplotlib figure.
        ax : Optional[Axes]
            Existing matplotlib axes.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib figure object.
        ax : matplotlib.axes.Axes
            The matplotlib axes object containing the partial RDF plot.

        Raises
        ------
        AssertionError
            If length of elements_list does not match Ntype.

        """
        if elements_list is not None:
            assert len(elements_list) == self.Ntype, (
                f"Length of elements_list ({len(elements_list)}) must match "
                f"number of atom types ({self.Ntype})"
            )
        if fig is None and ax is None:
            from mdapy.plotset import set_figure

            fig, ax = set_figure()

        import matplotlib.pyplot as plt

        lw = 1.0
        # Select appropriate colormap based on number of types
        if self.Ntype > 3:
            colorlist = plt.cm.get_cmap("tab20").colors[::2]
        else:
            colorlist = [i["color"] for i in plt.rcParams["axes.prop_cycle"]]

        n = 0
        for i in range(self.Ntype):
            for j in range(self.Ntype):
                if j >= i:  # Only plot upper triangle (including diagonal)
                    if n > len(colorlist) - 1:
                        n = 0

                    if elements_list is not None:
                        label = f"{elements_list[i]}-{elements_list[j]}"
                    else:
                        label = f"{self._old_type[i]}-{self._old_type[j]}"

                    ax.plot(
                        self.r,
                        self.g[i, j],
                        c=colorlist[n],
                        lw=lw,
                        label=label,
                    )
                    n += 1

        ax.legend(ncol=2, fontsize=6)
        ax.set_xlabel(r"r ($\AA$)")
        ax.set_ylabel("g(r)")
        ax.set_xlim(0, self.rc)

        return fig, ax


if __name__ == "__main__":
    pass
