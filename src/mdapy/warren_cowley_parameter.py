# Copyright (c) 2022-2025, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

from mdapy import _wcp
from typing import Optional, List, Tuple, TYPE_CHECKING
import polars as pl
import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes


class WarrenCowleyParameter:
    """
    Calculate the Warren-Cowley Parameter (WCP) for chemical short-range order analysis.

    The Warren-Cowley Parameter quantifies chemical ordering in multi-component systems
    by measuring the deviation of local atomic arrangements from random mixing. It is
    particularly useful for analyzing segregation, clustering, and ordering tendencies
    in alloys.

    Parameters
    ----------
    verlet_list : np.ndarray
        Neighbor list array of shape (N, max_neigh).
    neighbor_number : np.ndarray
        Number of neighbors for each atom, shape (N,).
    data : pl.DataFrame
        Atomic data containing either 'element' or 'type' column.

    Attributes
    ----------
    verlet_list : np.ndarray
        Neighbor indices.
    neighbor_number : np.ndarray
        Neighbor counts.
    data : pl.DataFrame
        Input atomic data.
    type_list : np.ndarray
        Atom types (0-indexed) derived from data.
    Ntype : int
        Number of distinct atom types.
    ele2type : dict, optional
        Mapping from element symbols to type indices (if using element column).
    WCP : np.ndarray
        WCP matrix of shape (Ntype, Ntype) after calling compute().
        Element (i,j) represents WCP for type-j atoms around type-i atoms.

    Notes
    -----
    The Warren-Cowley Parameter α_ij is defined as:

    .. math::

        \\alpha_{ij} = 1 - \\frac{P_{ij}}{c_j}

    where P_ij is the probability of finding a type-j atom around a type-i atom,
    and c_j is the global concentration of type-j atoms.

    Interpretation:
    - α_ij < 0: Preference for unlike neighbors (ordering)
    - α_ij = 0: Random mixing
    - α_ij > 0: Preference for like neighbors (clustering/segregation)
    - α_ij = 1: Complete segregation (no unlike neighbors)

    References
    ----------
    .. [1] Warren, B. E. (1990). X-ray Diffraction. Dover Publications.
    .. [2] Cowley, J. M. (1950). An approximate theory of order in alloys.
           Physical Review, 77(5), 669.
    """

    def __init__(
        self, verlet_list: np.ndarray, neighbor_number: np.ndarray, data: pl.DataFrame
    ) -> None:
        self.verlet_list = verlet_list
        self.neighbor_number = neighbor_number
        self.data = data
        if "element" in self.data.columns:
            self.ele2type = {
                j: i for i, j in enumerate(self.data["element"].unique().sort())
            }
            self.type_list = self.data.with_columns(
                pl.col("element").replace_strict(self.ele2type).rechunk().alias("type")
            )["type"].to_numpy(allow_copy=False)
            self.Ntype = len(self.ele2type)
        else:
            assert "type" in self.data.columns
            self.type_list = self.data["type"].to_numpy() - 1
            self.Ntype = len(np.unique(self.type_list))
            assert self.type_list.max() + 1 == self.Ntype

    def compute(self) -> None:
        """
        Compute the Warren-Cowley Parameter matrix.

        This method calculates the WCP for all type pairs and stores the result
        in the ``WCP`` attribute as a symmetric matrix.

        Notes
        -----
        After calling this method, the ``WCP`` attribute will contain an
        (Ntype, Ntype) array with WCP values for each type pair.
        """
        self.WCP = np.zeros((self.Ntype, self.Ntype), float)
        _wcp.get_wcp(
            self.verlet_list, self.neighbor_number, self.type_list, self.Ntype, self.WCP
        )

    def plot(
        self,
        elements_list: Optional[List[str]] = None,
        fig: Optional["Figure"] = None,
        ax: Optional["Axes"] = None,
        vmin: float = -2,
        vmax: float = 1,
        cmap: str = "GnBu",
    ) -> Tuple["Figure", "Axes"]:
        """
        Visualize the Warren-Cowley Parameter matrix as a heatmap.

        Parameters
        ----------
        elements_list : list of str, optional
            Element symbols for axis labels. If None and ele2type exists,
            uses element symbols from ele2type.
        fig : matplotlib.figure.Figure, optional
            Existing figure to plot on. If None, creates new figure.
        ax : matplotlib.axes.Axes, optional
            Existing axes to plot on. If None, creates new axes.
        vmin : float, default=-2
            Minimum value for colormap scaling.
        vmax : float, default=1
            Maximum value for colormap scaling.
        cmap : str, default='GnBu'
            Matplotlib colormap name.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object containing the plot.
        ax : matplotlib.axes.Axes
            Axes object containing the plot.

        Notes
        -----
        The plot displays WCP values as a colored matrix with numerical annotations.
        Rows represent central atom types, columns represent neighboring atom types.
        """
        import matplotlib.pyplot as plt

        if fig is None and ax is None:
            from mdapy.plotset import set_figure

            fig, ax = set_figure()

        h = ax.imshow(self.WCP[::-1], vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_xticks(np.arange(self.Ntype))
        ax.set_yticks(np.arange(self.Ntype))

        if elements_list is None:
            if hasattr(self, "ele2type"):
                elements_list = list(self.ele2type.keys())
        if elements_list is not None:
            assert len(elements_list) == self.Ntype
            ax.set_xticklabels(elements_list)
            ax.set_yticklabels(elements_list[::-1])
        else:
            ax.set_yticklabels(np.arange(self.Ntype)[::-1])

        for i in range(self.Ntype):
            for j in range(self.Ntype):
                if self.WCP[i, j] == 0:
                    name = "0.00"
                else:
                    name = f"{np.round(self.WCP[::-1][i, j], 2)}"
                ax.text(j, i, name, ha="center", va="center", color="k")

        ax.set_xlabel("Central element")
        ax.set_ylabel("Neighboring element")
        bar = plt.colorbar(h, ax=ax)
        bar.set_ticks(ticks=[vmin, 0, vmax])
        bar.set_label("WCP")

        return fig, ax


if __name__ == "__main__":
    pass
