# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
from __future__ import annotations
from typing import Optional, Tuple, List, Dict, Any, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray
from mdapy import _rdf
from mdapy.box import Box

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes


class RadialDistributionFunction:
    r"""Radial distribution function :math:`g(r)` of an atomic system.

    For a multi-component system, the total RDF is the
    concentration-weighted sum of the partial RDFs:

    .. math::

        g(r) = \sum_{\alpha \beta} c_{\alpha} c_{\beta} g_{\alpha \beta}(r)

    where :math:`c_{\alpha}` is the concentration of species :math:`\alpha`
    and :math:`g_{\alpha \beta}(r) = g_{\beta \alpha}(r)`. The partial RDFs
    are normalised so that :math:`g_{\alpha \beta}(r) \to 1` for an ideal
    homogeneous system.

    Two execution paths are supported:

    * **Verlet-list path** (default): supply a pre-built neighbour list via
      ``verlet_list`` / ``distance_list`` / ``neighbor_number``. Best when
      the cutoff is small relative to the box.
    * **Streaming path** (``streaming=True``): supply atomic positions via
      ``x`` / ``y`` / ``z``. Pair distances are binned directly into the
      histogram with no neighbour list materialised, so memory stays
      :math:`\mathcal{O}(N)` regardless of cutoff.

    Parameters
    ----------
    rc : float
        Cutoff distance for RDF calculation (in Angstroms).
    nbin : int
        Number of histogram bins.
    box : Box
        Simulation box.
    verlet_list : NDArray[np.int32], optional
        Required when ``streaming=False``.
    distance_list : NDArray[np.float64], optional
        Required when ``streaming=False``.
    neighbor_number : NDArray[np.int32], optional
        Required when ``streaming=False``.
    type_list : NDArray, optional
        Per-atom species labels (int type ids or str element symbols). Used
        as keys in :attr:`g_partial`. If omitted, all atoms share label 0.
    streaming : bool, default=False
        Use the streaming kernel (no Verlet list).
    x, y, z : NDArray[np.float64], optional
        Atomic Cartesian coordinates. Required when ``streaming=True``.

    Attributes
    ----------
    r : NDArray[np.float64]
        Bin centres, shape ``(nbin,)``.
    g_total : NDArray[np.float64]
        Total :math:`g(r)`, shape ``(nbin,)``.
    g_partial : dict
        Mapping ``(species_a, species_b) -> NDArray`` for :math:`a \leq b`
        (upper triangle). Each value is the partial :math:`g_{\alpha\beta}(r)`.
    elements : list
        Sorted list of unique species labels — the same labels used as keys
        in :attr:`g_partial`.
    Ntype : int
        Number of distinct species.
    vol : float
        Simulation box volume.
    N : int
        Total number of particles.
    """

    def __init__(
        self,
        rc: float,
        nbin: int,
        box: Box,
        verlet_list: Optional[NDArray[np.int32]] = None,
        distance_list: Optional[NDArray[np.float64]] = None,
        neighbor_number: Optional[NDArray[np.int32]] = None,
        type_list: Optional[NDArray] = None,
        streaming: bool = False,
        x: Optional[NDArray[np.float64]] = None,
        y: Optional[NDArray[np.float64]] = None,
        z: Optional[NDArray[np.float64]] = None,
    ) -> None:
        self.rc = float(rc)
        self.nbin = int(nbin)
        self.box = box
        self.vol = self.box.volume
        self.streaming = bool(streaming)

        if self.streaming:
            if x is None or y is None or z is None:
                raise ValueError(
                    "streaming=True requires x, y, z position arrays."
                )
            self._x = np.ascontiguousarray(x, dtype=np.float64)
            self._y = np.ascontiguousarray(y, dtype=np.float64)
            self._z = np.ascontiguousarray(z, dtype=np.float64)
            assert self._x.shape == self._y.shape == self._z.shape, (
                "x, y, z must have the same shape"
            )
            self.N = int(self._x.shape[0])
            self.verlet_list = None
            self.distance_list = None
            self.neighbor_number = None
        else:
            if verlet_list is None or distance_list is None or neighbor_number is None:
                raise ValueError(
                    "streaming=False requires verlet_list, distance_list, "
                    "neighbor_number."
                )
            self.verlet_list = verlet_list
            self.distance_list = distance_list
            self.neighbor_number = neighbor_number
            self.N = int(self.verlet_list.shape[0])

        # Species labels: keep the user-supplied identifiers (int or str) as
        # the dict keys in g_partial, while internally remapping to a dense
        # 0..Ntype-1 range for the C++ kernel.
        if type_list is None:
            raw = np.zeros(self.N, dtype=np.int32)
        else:
            raw = np.asarray(type_list)
        unique_sorted = sorted(set(raw.tolist()))
        self.elements: List[Any] = list(unique_sorted)
        self.Ntype = len(self.elements)
        label_to_idx = {label: i for i, label in enumerate(self.elements)}
        self.type_list = np.array(
            [label_to_idx[v] for v in raw.tolist()], dtype=np.int32
        )

    def compute(self) -> None:
        """Run the kernel and populate :attr:`g_total`, :attr:`g_partial`,
        and :attr:`r`."""
        edges = np.linspace(0, self.rc, self.nbin + 1)
        const = (4.0 * np.pi / 3.0 * (edges[1:] ** 3 - edges[:-1] ** 3)) / self.vol
        self.r = (edges[1:] + edges[:-1]) / 2

        # Raw histogram counts in shape (Ntype, Ntype, nbin).
        counts = np.zeros((self.Ntype, self.Ntype, self.nbin), dtype=np.float64)

        if self.streaming:
            _rdf._rdf_streaming(
                self._x, self._y, self._z, self.type_list,
                self.box.box, self.box.origin, self.box.boundary,
                counts, self.rc, self.nbin,
            )
        else:
            if self.Ntype > 1:
                _rdf._rdf(
                    self.verlet_list, self.distance_list, self.neighbor_number,
                    self.type_list, counts, self.rc, self.nbin,
                )
            else:
                # Single-species fast path returns 2× the upper-triangle count
                # in a 1D array; expand into the (1, 1, nbin) histogram.
                flat = np.zeros(self.nbin, dtype=np.float64)
                _rdf._rdf_single_species(
                    self.verlet_list, self.distance_list, self.neighbor_number,
                    flat, self.rc, self.nbin,
                )
                counts[0, 0] = flat

        number_per_type = np.bincount(self.type_list, minlength=self.Ntype)

        # Total g(r). Sum of all (alpha, beta) histogram entries divided by
        # N^2 and the shell volume gives the concentration-weighted total
        # (matches the textbook g(r) = sum_{ab} c_a c_b g_ab(r)).
        total = np.zeros(self.nbin, dtype=np.float64)
        for a in range(self.Ntype):
            for b in range(self.Ntype):
                total += counts[a, b]
        self.g_total = total / const / self.N**2

        # Partial g_{ab}(r) keyed by (label_a, label_b) for a <= b.
        self.g_partial: Dict[Tuple[Any, Any], NDArray[np.float64]] = {}
        for a in range(self.Ntype):
            n_a = number_per_type[a]
            for b in range(a, self.Ntype):
                n_b = number_per_type[b]
                if a == b:
                    raw = counts[a, b]
                else:
                    # Symmetric pair: both ordered halves of the histogram
                    # exist; combining gives counts(unordered pair) without
                    # double-counting in the normalisation below.
                    raw = counts[a, b] + counts[b, a]
                if n_a > 0 and n_b > 0:
                    g_ab = raw / (n_a * n_b) / const
                    if a != b:
                        # raw above already counted both (a→b) and (b→a)
                        # contributions; the standard partial-RDF
                        # normalisation expects the unordered count, hence
                        # the factor 1/2.
                        g_ab *= 0.5
                else:
                    g_ab = np.zeros_like(self.r)
                self.g_partial[(self.elements[a], self.elements[b])] = g_ab

    def plot(self, fig: Optional["Figure"] = None, ax: Optional["Axes"] = None) -> Tuple:
        """Plot the total RDF :math:`g(r)`."""
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
        fig: Optional["Figure"] = None,
        ax: Optional["Axes"] = None,
    ) -> Tuple:
        """Plot all partial RDFs :math:`g_{\\alpha\\beta}(r)`.

        Parameters
        ----------
        elements_list : list of str, optional
            Override the element labels used in the legend. Length must
            match :attr:`Ntype`. If ``None``, uses the keys of
            :attr:`g_partial` directly.
        """
        if elements_list is not None:
            assert len(elements_list) == self.Ntype, (
                f"Length of elements_list ({len(elements_list)}) must match "
                f"number of species ({self.Ntype})"
            )
        if fig is None and ax is None:
            from mdapy.plotset import set_figure

            fig, ax = set_figure()

        import matplotlib.pyplot as plt

        if self.Ntype > 3:
            colorlist = plt.cm.get_cmap("tab20").colors[::2]
        else:
            colorlist = [i["color"] for i in plt.rcParams["axes.prop_cycle"]]

        n = 0
        for (a, b), g_ab in self.g_partial.items():
            if elements_list is not None:
                ia = self.elements.index(a)
                ib = self.elements.index(b)
                label = f"{elements_list[ia]}-{elements_list[ib]}"
            else:
                label = f"{a}-{b}"
            color = colorlist[n % len(colorlist)]
            ax.plot(self.r, g_ab, c=color, lw=1.0, label=label)
            n += 1

        ax.legend(ncol=2, fontsize=6)
        ax.set_xlabel(r"r ($\AA$)")
        ax.set_ylabel("g(r)")
        ax.set_xlim(0, self.rc)

        return fig, ax


if __name__ == "__main__":
    pass
