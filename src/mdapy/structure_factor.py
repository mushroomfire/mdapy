# Copyright (c) 2022-2025, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

from __future__ import annotations
from typing import Optional, Tuple, TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes

from mdapy import _sfc
import numpy as np
import polars as pl
from mdapy.box import Box
import mdapy.tool_function as tool
from mdapy.data import xray_form_factor


class StructureFactor:
    r"""Compute the static structure factor S(k) of an atomic system.

    The static structure factor characterizes the spatial distribution of particles
    in reciprocal space and is related to the Fourier transform of the pair
    correlation function.

    Two computational methods are available:

    1. **Direct method**: Computes S(k) by directly evaluating the structure factor
       in reciprocal space using discrete k-space bins:

       .. math::

        S(\vec{k}) = \frac{1}{N}  \sum_{i=0}^{N} \sum_{j=0}^N e^{i\vec{k} \cdot
        \vec{r}_{ij}}


    2. **Debye method**: Uses the Debye scattering equation, which computes S(k)
       for specific k values. Better for large systems or when specific k values
       are needed.

       .. math::
           S(k) = \frac{1}{N} \sum_{i,j} \frac{\sin(k r_{ij})}{k r_{ij}}

    where N is the number of particles, k is the wavenumber magnitude, and r_ij
    is the distance between particles i and j.

    Parameters
    ----------
    data : pl.DataFrame
        Polars DataFrame containing atomic positions. Must have 'x', 'y', 'z' columns.
        For partial structure factors, must also contain 'element' or 'type' column.
    box : Box
        Simulation box object defining boundary conditions and box dimensions.
    k_min : float
        Minimum wavenumber for S(k) calculation (must be >= 0).
    k_max : float
        Maximum wavenumber for S(k) calculation (must be > k_min).
    nbins : int
        Number of bins for discretizing the k-space (must be > 0).
        - For 'direct' mode: creates nbins+1 bin edges, returns nbins values
        - For 'debye' mode: creates nbins linearly spaced k values
    cal_partial : bool, default=False
        If True, compute partial structure factors S_αβ(k) for each pair of
        atom types α and β. Requires 'element' or 'type' column in data.
    atomic_form_factors : bool, default=False
        If True, use atomic form factors f to weigh the atoms' individual contributions to S(k). Atomic form factors are taken from `TU Graz <https://lampz.tugraz.at/~hadley/ss1/crystaldiffraction/atomicformfactors/formfactors.php>`_.
    mode : {'direct', 'debye'}, default='direct'
        Computational method:
        - 'direct': Direct reciprocal space summation
        - 'debye': Debye scattering equation

    Attributes
    ----------
    k : np.ndarray
        Array of wavenumber values where S(k) is computed.
    Sk : np.ndarray
        Total static structure factor values.
    Sk_partial : dict[str, np.ndarray], optional
        Dictionary of partial structure factors (only if cal_partial=True).
        Keys are formatted as 'element1-element2'.
    Sk_partial_xray : dict[str, np.ndarray], optional
        Dictionary of partial structure factors weighted by atomic_form (only if cal_partial=True and atomic_form_factors=True).
        This is helpful to directly compare with experimental results.
        Keys are formatted as 'element1-element2'.
    Sk_xray : np.ndarray
        Total static structure factor values weighted by atomic_form  (only if cal_partial=True and atomic_form_factors=True).


    References
    ----------
    .. [1] Frenkel, D. & Smit, B. Understanding Molecular Simulation.
           Academic Press (2002)
    .. [2] Hansen, J.-P. & McDonald, I. R. Theory of Simple Liquids.
           Academic Press (2006)
    .. [3] freud.diffraction.StaticStructureFactorDirect
           https://freud.readthedocs.io/en/stable/modules/diffraction.html
    """

    def __init__(
        self,
        data: pl.DataFrame,
        box: Box,
        k_min: float,
        k_max: float,
        nbins: int,
        cal_partial: bool = False,
        atomic_form_factors: bool = False,
        mode: Literal["direct", "debye"] = "direct",
    ) -> None:
        self.data = data
        self.box = box
        self.k_min = float(k_min)
        assert k_min >= 0, "k_min must be non-negative"
        self.k_max = float(k_max)
        assert k_max > k_min, "k_max must be greater than k_min"
        self.nbins = int(nbins)
        assert nbins > 0, "nbins must be positive"
        self.cal_partial = cal_partial
        self.atomic_form_factors = atomic_form_factors
        self.mode = mode.lower()
        assert self.mode in ["direct", "debye"], "mode must be 'direct' or 'debye'"

    def _get_xray_form_factor(self, element: str):
        para = xray_form_factor[element]
        factors = np.zeros(len(self.k))
        for i in range(0, 4):
            factors += para[2 * i] * np.exp(
                -para[2 * i + 1] * (self.k / (np.pi * 4)) ** 2
            )
        return factors + para[-1]

    def compute(self) -> None:
        """Compute the static structure factor S(k).

        This method performs the actual calculation of S(k) using the specified
        computational mode. Results are stored in the `Sk` attribute, and if
        `cal_partial=True`, partial structure factors are stored in `Sk_partial`.

        Raises
        ------
        AssertionError
            If required columns ('x', 'y', 'z') are missing from data, or if
            'element' or 'type' column is missing when cal_partial=True.

        Notes
        -----
        For partial structure factors in a binary system with species A and B,
        three partial S(k) are computed: S_AA(k), S_AB(k), and S_BB(k).
        The total structure factor is:

        .. math::
            S(k) = S_{AA}(k) + 2S_{AB}(k) + S_{BB}(k)
        """
        for i in ["x", "y", "z"]:
            assert i in self.data.columns, f"Column '{i}' must be present in data"

        if self.mode == "direct":
            k = np.linspace(self.k_min, self.k_max, self.nbins + 1)
            self.k = (k[1:] + k[:-1]) / 2.0
        else:
            self.k = np.linspace(self.k_min, self.k_max, self.nbins)

        data, box = self.data, self.box
        rNum = 200  # safe atom number for statistical accuracy
        N = self.data.shape[0]
        repeat = [1, 1, 1]
        if N < rNum:
            if sum(self.box.boundary) > 0:
                while np.prod(repeat) * N < rNum:
                    for i in range(3):
                        if self.box.boundary[i] == 1:
                            repeat[i] += 1

        if sum(repeat) != 3:
            # Small box: replicate atoms to ensure sufficient statistics
            data, box = tool._replicate_pos(data, box, *repeat)

        if self.cal_partial:
            if self.atomic_form_factors:
                assert "element" in self.data.columns
            name = "element"
            if name not in self.data.columns:
                name = "type"
                assert name in self.data.columns, (
                    "Must have 'element' or 'type' column for partial structure factors"
                )
            data = data.sort(name)
            uniele = data[name].unique().sort()
            assert len(uniele) > 1, (
                "Need at least 2 species for partial structure factors"
            )
            self.Sk_partial = {}
            self.Sk = np.zeros(self.nbins)
            for i in range(len(uniele)):
                for j in range(i, len(uniele)):
                    Sk = np.zeros(self.nbins)
                    data1 = data.filter(pl.col(name) == uniele[i]).rechunk()
                    data2 = data.filter(pl.col(name) == uniele[j]).rechunk()
                    if self.mode == "direct":
                        _sfc.compute_sfc_direct(
                            data1["x"].to_numpy(allow_copy=False),
                            data1["y"].to_numpy(allow_copy=False),
                            data1["z"].to_numpy(allow_copy=False),
                            box.box,
                            box.origin,
                            box.boundary,
                            Sk,
                            self.nbins,
                            self.k_max,
                            self.k_min,
                            data2["x"].to_numpy(allow_copy=False),
                            data2["y"].to_numpy(allow_copy=False),
                            data2["z"].to_numpy(allow_copy=False),
                            data.shape[0],
                        )
                    else:
                        _sfc.compute_sfc_debye(
                            data1["x"].to_numpy(allow_copy=False),
                            data1["y"].to_numpy(allow_copy=False),
                            data1["z"].to_numpy(allow_copy=False),
                            box.box,
                            box.origin,
                            box.boundary,
                            self.k,
                            Sk,
                            data2["x"].to_numpy(allow_copy=False),
                            data2["y"].to_numpy(allow_copy=False),
                            data2["z"].to_numpy(allow_copy=False),
                            data.shape[0],
                        )
                    self.Sk_partial[f"{uniele[i]}-{uniele[j]}"] = Sk
                    self.Sk += Sk
                    if i != j:
                        self.Sk += Sk  # Account for symmetry (S_ij = S_ji)
            if self.atomic_form_factors:
                self.Sk_partial_xray = {}
                self.Sk_xray = np.zeros(self.nbins)
                concentration = []
                factor = []
                for i in range(len(uniele)):
                    concentration.append(
                        data.filter(pl.col("element") == uniele[i]).shape[0]
                        / data.shape[0]
                    )
                    assert uniele[i] in xray_form_factor.keys(), (
                        f"Unrecognized element: {uniele[i]}."
                    )
                    factor.append(self._get_xray_form_factor(uniele[i]))
                normalization = np.zeros(self.k.shape[0])
                for i in range(len(uniele)):
                    for j in range(i, len(uniele)):
                        pair = f"{uniele[i]}-{uniele[j]}"
                        if i == j:
                            self.Sk_partial_xray[pair] = (
                                self.Sk_partial[pair]
                                / concentration[i]
                                / concentration[j]
                                + 1
                                - 1 / concentration[i]
                            )
                            self.Sk_xray += (
                                self.Sk_partial_xray[pair]
                                * factor[i]
                                * factor[j]
                                * concentration[i]
                                * concentration[j]
                            )
                        else:
                            self.Sk_partial_xray[pair] = (
                                self.Sk_partial[pair]
                                / concentration[i]
                                / concentration[j]
                                + 1
                            )
                            self.Sk_xray += (
                                2
                                * self.Sk_partial_xray[pair]
                                * factor[i]
                                * factor[j]
                                * concentration[i]
                                * concentration[j]
                            )
                    normalization += concentration[i] * factor[i]
                self.Sk_xray /= normalization**2
        else:
            self.Sk = np.zeros(self.nbins)
            if self.mode == "direct":
                _sfc.compute_sfc_direct(
                    data["x"].to_numpy(allow_copy=False),
                    data["y"].to_numpy(allow_copy=False),
                    data["z"].to_numpy(allow_copy=False),
                    box.box,
                    box.origin,
                    box.boundary,
                    self.Sk,
                    self.nbins,
                    self.k_max,
                    self.k_min,
                )
            else:
                _sfc.compute_sfc_debye(
                    data["x"].to_numpy(allow_copy=False),
                    data["y"].to_numpy(allow_copy=False),
                    data["z"].to_numpy(allow_copy=False),
                    box.box,
                    box.origin,
                    box.boundary,
                    self.k,
                    self.Sk,
                )

    def plot(
        self, fig: Optional[Figure] = None, ax: Optional[Axes] = None
    ) -> Tuple[Figure, Axes]:
        """Plot the total static structure factor S(k).

        Creates a line plot of S(k) versus wavenumber k with markers.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            Matplotlib figure object. If None, a new figure is created.
        ax : matplotlib.axes.Axes, optional
            Matplotlib axes object. If None, new axes are created.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib figure object containing the plot.
        ax : matplotlib.axes.Axes
            The matplotlib axes object containing the plot.

        """
        if fig is None and ax is None:
            from mdapy.plotset import set_figure

            fig, ax = set_figure()
        if self.atomic_form_factors:
            ax.plot(self.k, self.Sk_xray, "o-", ms=3)
        else:
            ax.plot(self.k, self.Sk, "o-", ms=3)
        ax.set_xlabel(r"k (1/$\mathregular{\AA}$)")
        ax.set_ylabel("S(k)")
        ax.set_xlim(self.k_min, self.k_max)

        return fig, ax

    def plot_partial(
        self, fig: Optional[Figure] = None, ax: Optional[Axes] = None
    ) -> Tuple[Figure, Axes]:
        """Plot partial structure factors S_αβ(k) for multi-component systems.

        Creates a line plot showing all partial structure factors computed for
        different atom type pairs. Each partial S(k) is plotted with a different
        color and labeled accordingly.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            Matplotlib figure object. If None, a new figure is created.
        ax : matplotlib.axes.Axes, optional
            Matplotlib axes object. If None, new axes are created.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib figure object containing the plot.
        ax : matplotlib.axes.Axes
            The matplotlib axes object containing the plot.

        Raises
        ------
        AttributeError
            If compute() was not called with cal_partial=True.
        """
        if fig is None and ax is None:
            from mdapy.plotset import set_figure

            fig, ax = set_figure()
        import matplotlib.pyplot as plt

        if len(self.Sk_partial) > 6:
            colorlist = plt.cm.get_cmap("tab20").colors[::2]
        else:
            colorlist = [i["color"] for i in plt.rcParams["axes.prop_cycle"]]
        for i, j in enumerate(self.Sk_partial.keys()):
            if self.atomic_form_factors:
                ax.plot(
                    self.k, self.Sk_partial_xray[j], "o-", c=colorlist[i], label=j, ms=3
                )
            else:
                ax.plot(self.k, self.Sk_partial[j], "o-", c=colorlist[i], label=j, ms=3)

        ax.set_xlabel(r"k (1/$\mathregular{\AA}$)")
        ax.set_ylabel("S(k)")
        ax.set_xlim(self.k_min, self.k_max)
        ax.legend()
        return fig, ax


if __name__ == "__main__":
    from mdapy.build_lattice import build_crystal
    import matplotlib.pyplot as plt

    Cu = build_crystal("Cu", "fcc", 3.615)
    sfc = Cu.cal_structure_factor(0.1, 10, 50, mode="direct")
    sfc.plot()
    plt.show()
