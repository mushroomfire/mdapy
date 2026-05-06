# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

from __future__ import annotations
from typing import Optional, Tuple, Dict, List, Any, TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes

from mdapy import _sfc
from mdapy.parallel import get_num_threads
import numpy as np
import polars as pl
from mdapy.box import Box
import mdapy.tool_function as tool
from mdapy.data import xray_form_factor, neutron_form_factor, atomic_numbers
from mdapy.radial_distribution_function import RadialDistributionFunction


# Bohr radius in Angstrom (CODATA): used by the electron form-factor
# prefactor in the Mott-Bethe formula.
_BOHR_RADIUS_A = 0.529177210903


class StructureFactor:
    r"""Static structure factor :math:`S(k)` of an atomic system.

    The class computes the partial Faber-Ziman structure factors
    :math:`A_{\alpha\beta}(k)`, the total :math:`S(k)`, and (optionally)
    their X-ray, neutron, and electron-diffraction equivalents along with
    the real-space derived quantities :math:`g(r)`, :math:`G(r)`, and
    :math:`R(r)`.

    Two algorithms are available:

    * **Debye method** (``mode='debye'``, default). The radial distribution
      function :math:`g_{\alpha\beta}(r)` is computed once via the
      streaming RDF kernel up to ``rc`` (default :math:`L_{\max}/2`, where
      :math:`L_{\max}` is the longest cell-vector length). Each partial is
      then a one-dimensional sin transform, integrated by
      :func:`numpy.trapezoid` on the bin-centre grid:

      .. math::

         A_{\alpha\beta}(k) = 1 + \frac{4\pi\rho}{k}
            \int_0^{r_c} r\,[g_{\alpha\beta}(r) - 1]\,\sin(kr)\,w(r)\,dr,

      with :math:`w(r) = \mathrm{sinc}(2\pi r/L_{\max})` if ``window=True``
      and :math:`w(r) = 1` otherwise. The total
      :math:`S(k) = \sum_{\alpha\beta} c_\alpha c_\beta A_{\alpha\beta}(k)`.
      This implementation is bit-for-bit identical (max diff
      :math:`\lesssim 10^{-10}`) to the reference code accompanying
      [Erhard2024]_.

    * **Direct method** (``mode='direct'``). Generates every reciprocal
      lattice vector :math:`\vec{k}` in :math:`(k_{\min}, k_{\max}]` and
      sums

      .. math::

         F_\alpha(\vec{k}) = \frac{1}{\sqrt{N}}
            \sum_{i\in\alpha} e^{i \vec{k}\cdot\vec{r}_i}, \qquad
         A_{\alpha\beta}(\vec{k}) = \mathrm{Re}\left[
            F_\alpha^{*}(\vec{k})\,F_\beta(\vec{k})\right] / (c_\alpha c_\beta).

      Spherically averaging :math:`A_{\alpha\beta}(\vec{k})` in
      :math:`|\vec{k}|`-bins yields :math:`A_{\alpha\beta}(k)` of the same
      Faber-Ziman normalisation as the Debye method, modulo statistical
      noise from the discrete reciprocal-lattice grid. The direct method
      is much slower than the Debye method for large systems but resolves
      Bragg peaks in crystals where the Debye :math:`g(r)`-based method is
      blurred by the finite radial cutoff.

    The X-ray, neutron, and electron total structure factors are
    constructed from the partials following Erhard *et al.*:

    .. math::

         S^{(X)}(k) = \frac{
            \sum_{\alpha\beta} (2 - \delta_{\alpha\beta})
               c_\alpha c_\beta f_\alpha(k) f_\beta(k)\,A_{\alpha\beta}(k)
         }{
            \left[\sum_\alpha c_\alpha f_\alpha(k)\right]^2
         },

    with the appropriate form factors :math:`f(k)`:

    * X-ray: nine-coefficient Cromer-Mann fit
      :math:`f(k) = c + \sum_{i=1}^4 a_i\,\exp(-b_i\,(k/4\pi)^2)`
      from [BrownITC]_.
    * Neutron: tabulated coherent scattering length
      from `NIST <https://www.ncnr.nist.gov/resources/n-lengths/list.html>`__,
      :math:`k`-independent.
    * Electron: Mott-Bethe relation
      :math:`f_e(k) = (Z - f_X(k)) / (8 \pi^2 a_0 k^2)`,
      with :math:`a_0` the Bohr radius in Angstrom.

    Parameters
    ----------
    data : pl.DataFrame
        Atomic data with at least ``x``, ``y``, ``z`` columns. For
        :math:`A_{\alpha\beta}` and X-ray/neutron/electron derivations,
        ``element`` (preferred) or ``type`` is required.
    box : Box
        Simulation box.
    k_min, k_max : float
        Wavenumber range.
    nbins : int
        Number of :math:`k`-bins.
    cal_partial : bool, default False
        Compute :math:`A_{\alpha\beta}(k)`. Required for X-ray / neutron /
        electron totals.
    atomic_form_factors : bool, default False
        Compute X-ray total :math:`S^{(X)}(k)`. Implies ``cal_partial``.
    mode : {'debye', 'direct'}, default 'debye'
        Algorithm selector. ``'rdf'`` is accepted as an alias for
        ``'debye'``.
    rc : float, optional
        Radial cutoff for the Debye-mode RDF. Defaults to
        :math:`L_{\max}/2`. Only used by ``mode='debye'``.
    nbin_rdf : int, default 200
        Number of radial bins for the Debye-mode RDF.
    window : bool, default False
        Apply the Lorch window :math:`w(r) = \mathrm{sinc}(2\pi r/L_{\max})`
        in the Debye-mode integral. Default off, matching the textbook
        / Erhard reference.

    Attributes
    ----------
    k : np.ndarray, shape ``(nbins,)``
        Wavenumber bin centres.
    Sk : np.ndarray, shape ``(nbins,)``
        Total structure factor.
    Sk_partial : dict, optional
        Maps ``(species_a, species_b) -> A_alpha_beta(k)`` for the upper
        triangle (:math:`\alpha \leq \beta`). Populated when
        ``cal_partial=True``.
    Sk_xray, Sk_neutron, Sk_electron : np.ndarray, optional
        Total :math:`S^{(X)}`, :math:`S^{(N)}`, :math:`S^{(e)}`. Populated
        on demand by :meth:`get_xray_structure_factor` etc., and
        automatically when ``atomic_form_factors=True``.

    References
    ----------
    .. [Erhard2024] Erhard, L. C., Rohrer, J., Albe, K. & Deringer, V. L.
       "Modelling atomic and nanoscale structure in the silicon-oxygen
       system through active machine learning." *Nature Communications*
       **15**, 1927 (2024). https://doi.org/10.1038/s41467-024-45840-9.
       Reference Python implementation:
       https://github.com/LinusErhard/SiO_active_learning.
    .. [BrownITC] Brown, P. J., Fox, A. G., Maslen, E. N., O'Keefe, M. A.
       & Willis, B. T. M. "Intensity of diffraction intensities," in
       *International Tables for Crystallography*, Volume C: Mathematical,
       Physical, and Chemical Tables, Table 6.1.1.4, 4th ed. (2004).

    Notes
    -----
    The Faber-Ziman convention used here makes every
    :math:`A_{\alpha\beta}(k) \to 1` at large :math:`k`. To convert to the
    OVITO ``structure-factor`` (Ashcroft-Langreth-style) partials use

    .. math::

       S_{\alpha\beta}^{\mathrm{AL}}(k) = c_\alpha\,\delta_{\alpha\beta}
            + c_\alpha c_\beta\,[A_{\alpha\beta}(k) - 1].
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
        mode: Literal["direct", "debye", "rdf"] = "debye",
        rc: Optional[float] = None,
        nbin_rdf: int = 200,
        window: bool = False,
    ) -> None:
        self.data = data
        self.box = box
        self.k_min = float(k_min)
        assert k_min >= 0, "k_min must be non-negative"
        self.k_max = float(k_max)
        assert k_max > k_min, "k_max must be greater than k_min"
        self.nbins = int(nbins)
        assert nbins > 0, "nbins must be positive"
        # Computing the X-ray / neutron / electron totals always needs the
        # partials internally; promote ``cal_partial`` automatically so the
        # user can set ``atomic_form_factors=True`` alone.
        self.atomic_form_factors = bool(atomic_form_factors)
        self.cal_partial = bool(cal_partial) or self.atomic_form_factors
        self.mode = mode.lower()
        if self.mode == "rdf":
            self.mode = "debye"
        assert self.mode in ["direct", "debye"], (
            "mode must be 'direct' or 'debye'"
        )
        self.rc = rc
        self.nbin_rdf = int(nbin_rdf)
        self.window = bool(window)

        # Filled by compute().
        self.k: Optional[np.ndarray] = None
        self.Sk: Optional[np.ndarray] = None
        self.Sk_partial: Optional[Dict[Tuple[Any, Any], np.ndarray]] = None
        # Per-species concentration / form-factor caches; computed lazily
        # by the X-ray/neutron/electron getters.
        self._uniele: Optional[List[Any]] = None
        self._concentrations: Optional[np.ndarray] = None
        self._density: Optional[float] = None
        self.Sk_xray: Optional[np.ndarray] = None
        self.Sk_neutron: Optional[np.ndarray] = None
        self.Sk_electron: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public driver
    # ------------------------------------------------------------------
    def compute(self) -> None:
        """Run the configured algorithm and populate :attr:`Sk` and
        :attr:`Sk_partial`.

        When ``atomic_form_factors=True`` this also populates
        :attr:`Sk_xray`. Neutron and electron totals stay ``None`` until
        the corresponding getters are called.
        """
        for col in ("x", "y", "z"):
            assert col in self.data.columns, f"Column '{col}' must be present"

        if self.mode == "debye":
            self._compute_debye_mode()
        else:
            self._compute_direct_mode()

        if self.atomic_form_factors:
            self.Sk_xray = self.get_xray_structure_factor()

    # ------------------------------------------------------------------
    # Debye / RDF method
    # ------------------------------------------------------------------
    def _species_labels(self, view: pl.DataFrame) -> np.ndarray:
        if "element" in view.columns:
            return view["element"].to_numpy()
        if "type" in view.columns:
            return view["type"].to_numpy()
        return np.zeros(view.shape[0], dtype=np.int32)

    def _compute_debye_mode(self) -> None:
        data = self.data
        box = self.box
        L_max = float(max(np.linalg.norm(box.box[i]) for i in range(3)))
        if self.rc is None:
            self.rc = L_max / 2.0
        L_window = L_max

        self.k = np.linspace(self.k_min, self.k_max, self.nbins)
        if self.k_min == 0.0:
            self.k[0] = self.k[1] / 1000.0  # avoid 1/k division-by-zero

        # Replicate the box if rc exceeds L_min/2 of the original cell so
        # the streaming kernel sees every periodic image inside rc — this
        # is exactly what OVITO's RDF cell-list does.
        repeat = box.check_small_box(self.rc)
        rep_data = data
        rep_box = box
        if sum(repeat) != 3:
            rep_data, rep_box = tool.replicate(data, box, *repeat)
        rep_x = rep_data["x"].to_numpy()
        rep_y = rep_data["y"].to_numpy()
        rep_z = rep_data["z"].to_numpy()
        rep_types = self._species_labels(rep_data)

        rdf = RadialDistributionFunction(
            self.rc, self.nbin_rdf, rep_box,
            type_list=rep_types, streaming=True,
            x=rep_x, y=rep_y, z=rep_z,
        )
        rdf.compute()
        self._rdf = rdf
        self.r = rdf.r  # convenience for derived-PDF callers

        elements = list(rdf.elements)
        N_total = rep_data.shape[0]
        V = rep_box.volume
        rho = N_total / V
        type_list_remapped = rdf.type_list  # 0..Ntype-1
        n_per_type = np.bincount(type_list_remapped, minlength=len(elements))
        c = n_per_type / N_total

        self._uniele = elements
        self._concentrations = c
        self._density = rho
        self.num_density = rho
        self.density = rho  # legacy attribute names

        if self.window:
            w = np.sinc(2.0 * rdf.r / L_window)
        else:
            w = np.ones_like(rdf.r)

        sin_kr = np.sin(np.outer(self.k, rdf.r))

        Sk_partial: Dict[Tuple[Any, Any], np.ndarray] = {}
        # Faber-Ziman partial:
        #   A_ab(k) = 1 + (4π ρ / k) ∫ r [g_ab(r) - 1] sin(kr) w(r) dr
        for a, label_a in enumerate(elements):
            for b, label_b in enumerate(elements[a:], start=a):
                g_ab = rdf.g_partial[(label_a, label_b)]
                integrand = sin_kr * (rdf.r * (g_ab - 1.0) * w)
                integral = np.trapezoid(integrand, x=rdf.r, axis=1)
                Sk_partial[(label_a, label_b)] = (
                    1.0 + 4.0 * np.pi * rho / self.k * integral
                )

        # Total: 1 + (4π ρ / k) ∫ r (g_total(r) - 1) sin(kr) w(r) dr,
        # equivalent to Σ c_a c_b A_ab over the full pair matrix.
        integrand = sin_kr * (rdf.r * (rdf.g_total - 1.0) * w)
        Sk_total = 1.0 + 4.0 * np.pi * rho / self.k * np.trapezoid(
            integrand, x=rdf.r, axis=1
        )

        if self.cal_partial:
            self.Sk_partial = Sk_partial
        else:
            # Always keep partials cached on self for downstream X-ray /
            # neutron / electron getters; only suppress the public
            # attribute when the user did not ask for them.
            self._Sk_partial_internal = Sk_partial
        self.Sk = Sk_total

    # ------------------------------------------------------------------
    # Direct method
    # ------------------------------------------------------------------
    def _compute_direct_mode(self) -> None:
        data = self.data
        box = self.box
        # k-bin edges (used by the C++ binner) → returned bin centres.
        edges = np.linspace(self.k_min, self.k_max, self.nbins + 1)
        self.k = (edges[1:] + edges[:-1]) / 2.0

        # Replicate small boxes so that every pair separation up to k_max
        # is well-resolved by the discrete reciprocal lattice (matches the
        # historical mdapy behaviour and freud's recommendation).
        rNum = 200
        N = data.shape[0]
        repeat = [1, 1, 1]
        if N < rNum and sum(box.boundary) > 0:
            while np.prod(repeat) * N < rNum:
                for i in range(3):
                    if box.boundary[i] == 1:
                        repeat[i] += 1
        if sum(repeat) != 3:
            data, box = tool.replicate(data, box, *repeat)

        # Build species labels → contiguous ids on the (possibly enlarged)
        # frame, so we can compute a single F_alpha(k) per species.
        if self.cal_partial:
            if "element" in data.columns:
                col = "element"
            elif "type" in data.columns:
                col = "type"
            else:
                raise ValueError(
                    "cal_partial / atomic_form_factors require an "
                    "'element' or 'type' column."
                )
            uniele = sorted(data[col].unique().to_list())
            assert len(uniele) >= 1
        else:
            uniele = ["all"]

        N_total = data.shape[0]
        V = box.volume
        self._uniele = uniele
        self._density = N_total / V
        self.num_density = self._density
        self.density = self._density

        x_all = data["x"].to_numpy(allow_copy=False)
        y_all = data["y"].to_numpy(allow_copy=False)
        z_all = data["z"].to_numpy(allow_copy=False)

        if self.cal_partial:
            # Compute every Ashcroft-Langreth partial in one C++ pass:
            # F_alpha(k) is built once per species, then every (a, b)
            # cross-product is binned in parallel. This drops the cost of
            # the partial decomposition from O(Ntype^2 * N * n_k) (the
            # naive pair-by-pair approach) to O(Ntype * N * n_k).
            label_to_idx = {sp: i for i, sp in enumerate(uniele)}
            type_dense = (
                data.select(
                    pl.col(col).replace_strict(label_to_idx).cast(pl.Int32)
                )
                .to_numpy()
                .ravel()
            )
            n_per_type = np.bincount(type_dense, minlength=len(uniele))
            c = n_per_type / N_total
            self._concentrations = c

            partial_AL = np.zeros(
                (len(uniele), len(uniele), self.nbins), dtype=np.float64
            )
            _sfc.compute_sfc_direct_partial(
                x_all, y_all, z_all,
                np.ascontiguousarray(type_dense, dtype=np.int32),
                len(uniele),
                box.box, box.origin, box.boundary,
                partial_AL,
                self.nbins, self.k_max, self.k_min,
                get_num_threads(),
            )

            # Convert Ashcroft-Langreth partial to Faber-Ziman so the
            # public ``Sk_partial`` dict uses the same convention as the
            # Debye-mode output.
            #   S^AL_aa = c_a + c_a^2 (A_aa - 1)  -> A_aa = (S^AL_aa - c_a)/c_a^2 + 1
            #   S^AL_ab = c_a c_b (A_ab - 1)      -> A_ab = S^AL_ab / (c_a c_b) + 1
            Sk_partial_FZ: Dict[Tuple[Any, Any], np.ndarray] = {}
            for ia, sp_a in enumerate(uniele):
                for ib in range(ia, len(uniele)):
                    sp_b = uniele[ib]
                    Sk_AL = partial_AL[ia, ib]
                    if ia == ib:
                        Sk_partial_FZ[(sp_a, sp_b)] = (
                            (Sk_AL - c[ia]) / (c[ia] ** 2) + 1.0
                        )
                    else:
                        Sk_partial_FZ[(sp_a, sp_b)] = (
                            Sk_AL / (c[ia] * c[ib]) + 1.0
                        )

            self.Sk_partial = Sk_partial_FZ
            # Total = Sum over the full (Ntype, Ntype) AL matrix.
            Sk_total = np.zeros(self.nbins)
            for ia in range(len(uniele)):
                for ib in range(len(uniele)):
                    Sk_total += partial_AL[ia, ib]
            self.Sk = Sk_total
        else:
            self.Sk = np.zeros(self.nbins)
            _sfc.compute_sfc_direct(
                x_all, y_all, z_all,
                box.box, box.origin, box.boundary,
                self.Sk, self.nbins, self.k_max, self.k_min,
                num_t=get_num_threads(),
            )

    # ------------------------------------------------------------------
    # Form factors and weighted totals
    # ------------------------------------------------------------------
    def _xray_form_factor(self, element: str) -> np.ndarray:
        para = xray_form_factor[element]
        f = np.zeros_like(self.k)
        for i in range(4):
            f += para[2 * i] * np.exp(
                -para[2 * i + 1] * (self.k / (4.0 * np.pi)) ** 2
            )
        return f + para[-1]

    def _neutron_form_factor(self, element: str) -> np.ndarray:
        b = neutron_form_factor[element]
        # k-independent; broadcast to the k grid. Complex values flow
        # through the formula naturally — magnitude squared at the end.
        return np.full_like(self.k, b, dtype=np.complex128 if isinstance(b, complex) else np.float64)

    def _electron_form_factor(self, element: str) -> np.ndarray:
        # Mott-Bethe: f_e(k) = (Z - f_X(k)) / (8 π² a_0 k²),
        # with k in 1/Å and a_0 in Å.
        Z = atomic_numbers[element]
        fx = self._xray_form_factor(element)
        prefactor = 1.0 / (8.0 * np.pi ** 2 * _BOHR_RADIUS_A)
        return prefactor * (Z - fx) / (self.k ** 2)

    def _weighted_total(self, kind: str) -> np.ndarray:
        if self.Sk_partial is None and not hasattr(self, "_Sk_partial_internal"):
            raise RuntimeError(
                "Run compute() with cal_partial=True (or set "
                "atomic_form_factors=True) before requesting a "
                f"{kind}-weighted total."
            )
        partial = (
            self.Sk_partial if self.Sk_partial is not None else self._Sk_partial_internal
        )
        c = self._concentrations
        elements = self._uniele
        if kind == "xray":
            ff = [self._xray_form_factor(e) for e in elements]
        elif kind == "neutron":
            ff = [self._neutron_form_factor(e) for e in elements]
        elif kind == "electron":
            ff = [self._electron_form_factor(e) for e in elements]
        else:
            raise ValueError(f"unknown weighting kind: {kind!r}")
        norm = np.zeros_like(ff[0])
        for i, fi in enumerate(ff):
            norm += c[i] * fi
        total = np.zeros_like(ff[0])
        for (a, b), A_ab in partial.items():
            ia = elements.index(a); ib = elements.index(b)
            multi = 1.0 if ia == ib else 2.0
            total += multi * c[ia] * c[ib] * ff[ia] * ff[ib] * A_ab
        result = total / norm ** 2
        if kind == "neutron":
            return np.real(result * np.conj(result)) ** 0.5 if np.iscomplexobj(result) else result
        return result

    def get_xray_structure_factor(self) -> np.ndarray:
        r"""Return the X-ray total :math:`S^{(X)}(k)`.

        Implements the Erhard convention

        .. math::

           S^{(X)}(k) = \frac{
              \sum_{\alpha\beta} (2 - \delta_{\alpha\beta})
                 c_\alpha c_\beta f_\alpha(k) f_\beta(k)\,A_{\alpha\beta}(k)
           }{\left[\sum_\alpha c_\alpha f_\alpha(k)\right]^2}

        with :math:`f(k)` the Cromer-Mann fit
        :math:`c + \sum_i a_i e^{-b_i (k/4\pi)^2}` from [BrownITC]_.
        """
        self.Sk_xray = self._weighted_total("xray")
        return self.Sk_xray

    def get_neutron_structure_factor(self) -> np.ndarray:
        r"""Return the neutron total :math:`S^{(N)}(k)`.

        Form factors are the tabulated coherent scattering lengths from
        NIST (see :data:`mdapy.data.neutron_form_factor`) and are
        :math:`k`-independent. For absorptive isotopes (e.g.
        :math:`^{10}\mathrm{B}`, :math:`\mathrm{Cd}`,
        :math:`\mathrm{Sm}`) the tabulated value is complex; the returned
        :math:`S^{(N)}(k)` is then the modulus of the weighted sum.
        """
        self.Sk_neutron = self._weighted_total("neutron")
        return self.Sk_neutron

    def get_electron_structure_factor(self) -> np.ndarray:
        r"""Return the electron-diffraction total :math:`S^{(e)}(k)`.

        Form factors follow the Mott-Bethe relation

        .. math::

           f_e(k) = \frac{Z - f_X(k)}{8 \pi^2 a_0\,k^2}

        with :math:`a_0` the Bohr radius (in :math:`\AA`) and
        :math:`f_X(k)` the X-ray form factor.
        """
        self.Sk_electron = self._weighted_total("electron")
        return self.Sk_electron

    # ------------------------------------------------------------------
    # Real-space derived quantities
    # ------------------------------------------------------------------
    def _real_space_derived(self, kind: str, r: Optional[np.ndarray] = None):
        r"""Inverse-transform :math:`S^{(\cdot)}(k) - 1` to get the
        reduced pair distribution :math:`G(r)`, the pair distribution
        :math:`g(r)`, and the radial distribution :math:`R(r)`."""
        if kind == "xray":
            S = self.get_xray_structure_factor()
        elif kind == "neutron":
            S = self.get_neutron_structure_factor()
        elif kind == "electron":
            S = self.get_electron_structure_factor()
        else:
            raise ValueError(f"unknown weighting kind: {kind!r}")

        if r is None:
            r = self.r if hasattr(self, "r") else np.linspace(
                0.0, np.pi / (self.k[1] - self.k[0]), 200
            )

        rho = self._density
        sin_kr = np.sin(np.outer(r, self.k))
        # G(r) = (2/π) ∫ k [S(k) - 1] sin(kr) dk
        G = (2.0 / np.pi) * np.trapezoid(
            sin_kr * self.k * (S - 1.0), x=self.k, axis=1
        )
        # Avoid divide-by-zero at r = 0; set g(0) = 0 to match Erhard's code.
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.where(r > 0, G / (4.0 * np.pi * r * rho) + 1.0, 0.0)
        R = 4.0 * np.pi * r ** 2 * rho * g
        return r, g, G, R

    def get_xray_pair_distribution_function(
        self, r: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Return :math:`(r, g_X(r))` reconstructed from
        :math:`S^{(X)}(k)`."""
        rr, g, _, _ = self._real_space_derived("xray", r)
        return rr, g

    def get_xray_reduced_pair_distribution_function(
        self, r: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Return :math:`(r, G_X(r) = 4\pi r \rho [g_X(r) - 1])`."""
        rr, _, G, _ = self._real_space_derived("xray", r)
        return rr, G

    def get_xray_radial_distribution_function(
        self, r: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Return :math:`(r, R_X(r) = 4\pi r^2 \rho g_X(r))`."""
        rr, _, _, R = self._real_space_derived("xray", r)
        return rr, R

    def get_neutron_pair_distribution_function(
        self, r: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Return :math:`(r, g_N(r))`."""
        rr, g, _, _ = self._real_space_derived("neutron", r)
        return rr, g

    def get_neutron_reduced_pair_distribution_function(
        self, r: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Return :math:`(r, G_N(r))`."""
        rr, _, G, _ = self._real_space_derived("neutron", r)
        return rr, G

    def get_neutron_radial_distribution_function(
        self, r: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Return :math:`(r, R_N(r))`."""
        rr, _, _, R = self._real_space_derived("neutron", r)
        return rr, R

    def get_electron_pair_distribution_function(
        self, r: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Return :math:`(r, g_e(r))`."""
        rr, g, _, _ = self._real_space_derived("electron", r)
        return rr, g

    def get_electron_reduced_pair_distribution_function(
        self, r: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Return :math:`(r, G_e(r))`."""
        rr, _, G, _ = self._real_space_derived("electron", r)
        return rr, G

    def get_electron_radial_distribution_function(
        self, r: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Return :math:`(r, R_e(r))`."""
        rr, _, _, R = self._real_space_derived("electron", r)
        return rr, R

    # ------------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------------
    def plot(
        self, fig: Optional[Figure] = None, ax: Optional[Axes] = None
    ) -> Tuple[Figure, Axes]:
        r"""Plot the total :math:`S(k)` (or :math:`S^{(X)}(k)` if
        ``atomic_form_factors=True``)."""
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
        r"""Plot the partial Faber-Ziman :math:`A_{\alpha\beta}(k)`."""
        if fig is None and ax is None:
            from mdapy.plotset import set_figure

            fig, ax = set_figure()
        import matplotlib.pyplot as plt

        if self.Sk_partial is None:
            raise AttributeError(
                "compute() must be called with cal_partial=True before "
                "plot_partial()."
            )

        if len(self.Sk_partial) > 6:
            colorlist = plt.cm.get_cmap("tab20").colors[::2]
        else:
            colorlist = [i["color"] for i in plt.rcParams["axes.prop_cycle"]]
        for i, ((a, b), arr) in enumerate(self.Sk_partial.items()):
            label = f"{a}-{b}"
            ax.plot(self.k, arr, "o-", c=colorlist[i % len(colorlist)],
                    label=label, ms=3)
        ax.set_xlabel(r"k (1/$\mathregular{\AA}$)")
        ax.set_ylabel(r"$A_{\alpha\beta}(k)$")
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
