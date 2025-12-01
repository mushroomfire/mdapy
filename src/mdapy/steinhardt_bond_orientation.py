# Copyright (c) 2022-2025, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

import numpy as np
import polars as pl
from mdapy.box import Box
from mdapy import _sbo


class SteinhardtBondOrientation:
    r"""Compute Steinhardt bond orientation order parameters.
    The Steinhardt bond orientation order parameters provide a way to characterize
    local structural order in molecular dynamics simulations. This class computes
    the spherical harmonics-based order parameters :math:`q_l` and :math:`w_l` for
    identifying crystal structures and distinguishing between solid and liquid phases.

    Parameters
    ----------
    box : Box
        Simulation box object containing boundary conditions and dimensions.
    data : pl.DataFrame
        Polars DataFrame containing particle positions with columns 'x', 'y', 'z'.
    llist : np.ndarray
        Array of spherical harmonic degrees :math:`l` to compute (e.g., [4, 6, 8]).
    nnn : int
        Number of nearest neighbors to consider (0 to use cutoff distance instead).
    rc : float
        Cutoff radius for neighbor search (ignored if nnn > 0 or use_voronoi=True).
    average : bool
        Whether to compute averaged order parameters :math:`\bar{q}_l` over neighbor shells.
    use_voronoi : bool
        Use Voronoi tessellation to determine neighbors.
    use_weight : bool
        Apply weights to neighbor contributions.
    weight : np.ndarray
        Weight array for each neighbor pair (must match verlet_list shape if use_weight=True).
    verlet_list : np.ndarray
        Precomputed neighbor list array.
    distance_list : np.ndarray
        Precomputed distances for neighbor pairs.
    neighbor_number : np.ndarray
        Number of neighbors for each particle.
    wl : bool
        Compute third-order invariant :math:`w_l` parameters.
    wlhat : bool
        Compute normalized third-order invariant :math:`\hat{w}_l` parameters.
    identify_liquid : bool
        Enable solid-liquid classification (requires :math:`l=6` in llist).
    threshold : float
        Threshold for solid-liquid identification (default: 0.7 for normalized :math:`q_6`).
    n_bond : int
        Minimum number of "solid-like" bonds for solid classification.

    Attributes
    ----------
    qlm_r : np.ndarray
        Real part of spherical harmonics :math:`q_{lm}`, shape (:math:`N_{particles}`, :math:`N_l`, :math:`2 l_{max} + 1`).
    qlm_i : np.ndarray
        Imaginary part of spherical harmonics :math:`q_{lm}`, shape (:math:`N_{particles}`, :math:`N_l`, :math:`2 l_{max} + 1`).
    qnarray : np.ndarray
        Computed order parameters, shape (:math:`N_{particles}` :math:`N_{columns}`) where columns contain
        :math:`q_l`, and optionally :math:`w_l` and :math:`\hat{w}_l` values.
    solidliquid : np.ndarray, optional
        Solid-liquid classification array (0=liquid, 1=solid), only computed if
        identify_liquid=True.
    nbond : np.ndarray, optional
        Number of solid-like bonds for each particle, only computed if identify_liquid=True.

    Notes
    -----
    For a particle :math:`i`, we calculate the quantity :math:`q_{lm}` by summing the spherical harmonics
    between particle :math:`i` and its neighbors :math:`j` in a local region:

    .. math::

        q_{lm}(i) = \frac{1}{N_b} \sum \limits_{j=1}^{N_b}
        Y_{lm}(\theta(\vec{r}_{ij}), \phi(\vec{r}_{ij}))

    Then the :math:`q_l` order parameter is computed by combining the :math:`q_{lm}`
    in a rotationally invariant fashion to remove local orientational order:

    .. math::

        q_l(i) = \sqrt{\frac{4\pi}{2l+1} \sum \limits_{m=-l}^{l}
        |q_{lm}(i)|^2 }

    If the ``wl`` parameter is ``True``, this class computes the quantity
    :math:`w_l`, defined as a weighted average over the
    :math:`q_{lm}(i)` values using `Wigner 3-j symbols
    <https://en.wikipedia.org/wiki/3-j_symbol>`__ (related to `Clebsch-Gordan
    coefficients
    <https://en.wikipedia.org/wiki/Clebsch%E2%80%93Gordan_coefficients>`__).
    The resulting combination is rotationally invariant:

    .. math::

        w_l(i) = \sum \limits_{m_1 + m_2 + m_3 = 0} \begin{pmatrix}
            l & l & l \\
            m_1 & m_2 & m_3
        \end{pmatrix}
        q_{lm_1}(i) q_{lm_2}(i) q_{lm_3}(i)

    If ``wlhat`` parameter to ``True`` will
    normalize the :math:`w_l` order parameter as follows:

    .. math::

        w_l(i) = \frac{
            \sum \limits_{m_1 + m_2 + m_3 = 0} \begin{pmatrix}
                l & l & l \\
                m_1 & m_2 & m_3
            \end{pmatrix}
            q_{lm_1}(i) q_{lm_2}(i) q_{lm_3}(i)}
            {\left(\sum \limits_{m=-l}^{l} |q_{lm}(i)|^2 \right)^{3/2}}

    If ``average`` is ``True``, the class computes a variant of this order
    parameter that performs an average over the first and second shell combined
    . To compute this parameter, we perform a second
    averaging over the first neighbor shell of the particle to implicitly
    include information about the second neighbor shell. This averaging is
    performed by replacing the value :math:`q_{lm}(i)` in the original
    definition by :math:`\overline{q}_{lm}(i)`, the average value of
    :math:`q_{lm}(k)` over all the :math:`N_b` neighbors :math:`k`
    of particle :math:`i`, including particle :math:`i` itself:

    .. math::
        \overline{q}_{lm}(i) = \frac{1}{N_b} \sum \limits_{k=0}^{N_b}
        q_{lm}(k)

    If ``use_weight`` is True, the contributions of each neighbor are weighted.
    Neighbor weights :math:`w_{ij}` are defined for a
    Vornoi face area obtained from Voronoi neighbor or one with user-provided weights, and
    default to 1 if not otherwise provided. The formulas are modified as
    follows, replacing :math:`q_{lm}(i)` with the weighted value
    :math:`q'_{lm}(i)`:

    .. math::

        q'_{lm}(i) = \frac{1}{\sum_{j=1}^{N_b} w_{ij}}
        \sum \limits_{j=1}^{N_b} w_{ij} Y_{lm}(\theta(\vec{r}_{ij}),
        \phi(\vec{r}_{ij}))
    
    For solid-liquid classification (only performed when :math:`l=6` is included), 
    particles are identified as solid if they have at least :math:`n_{\text{bond}}` 
    "solid-like" bonds. A bond between particles :math:`i` and :math:`j` is considered 
    solid-like if their local bond orientations are correlated:

    .. math::

        \mathrm{dot}_{6}(i,j) = \frac{\sum_{m=-6}^{6} q_{6m}(i) \cdot q_{6m}^*(j)}{\sqrt{\sum_{m=-6}^{6} |q_{6m}(i)|^2} \sqrt{\sum_{m=-6}^{6} |q_{6m}(j)|^2}} > \text{threshold}

    where the default threshold is 0.7 for normalized :math:`q_6` values. When solid bond is larger than ``n_bond``, the atom is treated as solid atom.
    The default ``n_bond`` is 7, where 6-8 is generally good for FCC and BCC strutures.

    References
    ----------
    .. [1] Paul J. Steinhardt. Bond-orientational order in liquids and glasses. Physical Review B, 28(2):784-805, 1983. doi:10.1103/PhysRevB.28.784.

    .. [2] Wolfgang Lechner and Christoph Dellago. Accurate determination of crystal structures based on averaged local bond order parameters. The Journal of Chemical Physics, 129(11):114707, 2008. http://dx.doi.org/10.1063/1.2977970

    .. [3] L. Filion, M. Hermes, R. Ni, and M. Dijkstra. Crystal nucleation of hard spheres using molecular dynamics, umbrella sampling, and forward flux sampling: a comparison of simulation techniques. The Journal of Chemical Physics, 133(24):244115, http://dx.doi.org/10.1063/1.3506838.
    """

    def __init__(
        self,
        box: Box,
        data: pl.DataFrame,
        llist: np.ndarray,
        nnn: int,
        rc: float,
        average: bool,
        use_voronoi: bool,
        use_weight: bool,
        weight: np.ndarray,
        verlet_list: np.ndarray,
        distance_list: np.ndarray,
        neighbor_number: np.ndarray,
        wl: bool,
        wlhat: bool,
        identify_liquid: bool,
        threshold: float,
        n_bond: int,
    ) -> None:
        self.box = box
        self.data = data
        self.llist = llist
        self.nnn = nnn
        self.rc = rc
        self.average = average
        self.use_voronoi = use_voronoi
        self.use_weight = use_weight
        self.weight = weight
        self.verlet_list = verlet_list
        self.distance_list = distance_list
        self.neighbor_number = neighbor_number
        self.wl = wl
        self.wlhat = wlhat
        self.identify_liquid = identify_liquid
        self.threshold = threshold
        self.n_bond = n_bond

    def compute(self) -> None:
        """Compute the Steinhardt bond orientation order parameters.

        This method calculates the bond orientation order parameters and stores
        the results as instance attributes. If identify_liquid is True, it also
        performs solid-liquid classification.

        Returns
        -------
        None
            Results are stored in the following instance attributes:

            - qlm_r : Real part of spherical harmonics :math:`q_{lm}`
            - qlm_i : Imaginary part of spherical harmonics :math:`q_{lm}`
            - qnarray : Computed order parameters array
            - solidliquid : Solid-liquid classification (if identify_liquid=True)
            - nbond : Number of solid-like bonds per particle (if identify_liquid=True)

        Raises
        ------
        AssertionError
            If identify_liquid is True but l=6 is not in llist.
        AssertionError
            If identify_liquid is True but threshold or n_bond are not positive.
        AssertionError
            If rc is not positive when using cutoff-based neighbor search.
        AssertionError
            If use_weight is True but weight array shape doesn't match verlet_list.
        """
        if self.identify_liquid:
            assert 6 in self.llist
            assert self.threshold > 0
            assert self.n_bond > 0
        lmax = self.llist.max()
        self.qlm_r = np.zeros((self.data.shape[0], self.llist.shape[0], 2 * lmax + 1))
        self.qlm_i = np.zeros_like(self.qlm_r)
        ncol = self.llist.shape[0]
        if self.wl:
            ncol += self.llist.shape[0]
        if self.wlhat:
            ncol += self.llist.shape[0]
        self.qnarray = np.zeros((self.data.shape[0], ncol))
        if self.use_voronoi:
            self.rc = 10000000000.0
        else:
            if self.nnn > 0:
                self.rc = 1000000000.0
            else:
                assert self.rc > 0
        if not self.use_weight:
            self.weight = np.zeros((2, 2))
        else:
            assert self.weight.shape == self.verlet_list.shape
        _sbo.get_sq(
            self.data["x"].to_numpy(allow_copy=False),
            self.data["y"].to_numpy(allow_copy=False),
            self.data["z"].to_numpy(allow_copy=False),
            self.box.box,
            self.box.origin,
            self.box.boundary,
            self.verlet_list,
            self.distance_list,
            self.neighbor_number,
            self.weight,
            self.llist,
            self.nnn,
            lmax,
            self.wl,
            self.wlhat,
            self.average,
            self.use_voronoi,
            self.rc,
            self.use_weight,
            self.qlm_r,
            self.qlm_i,
            self.qnarray,
        )
        if self.identify_liquid:
            Q6index = int(np.where(self.llist == 6)[0][0])
            Q6 = np.ascontiguousarray(self.qnarray[:, Q6index])
            self.solidliquid = np.zeros(self.data.shape[0], np.int32)
            self.nbond = np.zeros(self.data.shape[0], np.int32)
            _sbo.identifySolidLiquid(
                Q6index,
                Q6,
                self.verlet_list,
                self.distance_list,
                self.neighbor_number,
                self.qlm_r,
                self.qlm_i,
                float(self.threshold),
                int(self.n_bond),
                self.solidliquid,
                self.nbond,
                self.use_voronoi,
                self.nnn,
                self.rc,
            )
