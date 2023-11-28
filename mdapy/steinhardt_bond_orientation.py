# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import numpy as np
import taichi as ti

try:
    from tool_function import _check_repeat_cutoff, _check_repeat_nearest
    from replicate import Replicate
    from neighbor import Neighbor
    from nearest_neighbor import NearestNeighbor
except Exception:
    from .tool_function import _check_repeat_cutoff, _check_repeat_nearest
    from .replicate import Replicate
    from .neighbor import Neighbor
    from .nearest_neighbor import NearestNeighbor


nfac_table_numpy = np.array(
    [
        1,
        1,
        2,
        6,
        24,
        120,
        720,
        5040,
        40320,
        362880,
        3628800,
        39916800,
        479001600,
        6227020800,
        87178291200,
        1307674368000,
        20922789888000,
        355687428096000,
        6.402373705728e15,
        1.21645100408832e17,
        2.43290200817664e18,
        5.10909421717094e19,
        1.12400072777761e21,
        2.5852016738885e22,
        6.20448401733239e23,
        1.5511210043331e25,
        4.03291461126606e26,
        1.08888694504184e28,
        3.04888344611714e29,
        8.8417619937397e30,
        2.65252859812191e32,
        8.22283865417792e33,
        2.63130836933694e35,
        8.68331761881189e36,
        2.95232799039604e38,
        1.03331479663861e40,
        3.71993326789901e41,
        1.37637530912263e43,
        5.23022617466601e44,
        2.03978820811974e46,
        8.15915283247898e47,
        3.34525266131638e49,
        1.40500611775288e51,
        6.04152630633738e52,
        2.65827157478845e54,
        1.1962222086548e56,
        5.50262215981209e57,
        2.58623241511168e59,
        1.24139155925361e61,
        6.08281864034268e62,
        3.04140932017134e64,
        1.55111875328738e66,
        8.06581751709439e67,
        4.27488328406003e69,
        2.30843697339241e71,
        1.26964033536583e73,
        7.10998587804863e74,
        4.05269195048772e76,
        2.35056133128288e78,
        1.3868311854569e80,
        8.32098711274139e81,
        5.07580213877225e83,
        3.14699732603879e85,
        1.98260831540444e87,
        1.26886932185884e89,
        8.24765059208247e90,
        5.44344939077443e92,
        3.64711109181887e94,
        2.48003554243683e96,
        1.71122452428141e98,
        1.19785716699699e100,
        8.50478588567862e101,
        6.12344583768861e103,
        4.47011546151268e105,
        3.30788544151939e107,
        2.48091408113954e109,
        1.88549470166605e111,
        1.45183092028286e113,
        1.13242811782063e115,
        8.94618213078297e116,
        7.15694570462638e118,
        5.79712602074737e120,
        4.75364333701284e122,
        3.94552396972066e124,
        3.31424013456535e126,
        2.81710411438055e128,
        2.42270953836727e130,
        2.10775729837953e132,
        1.85482642257398e134,
        1.65079551609085e136,
        1.48571596448176e138,
        1.3520015276784e140,
        1.24384140546413e142,
        1.15677250708164e144,
        1.08736615665674e146,
        1.03299784882391e148,
        9.91677934870949e149,
        9.61927596824821e151,
        9.42689044888324e153,
        9.33262154439441e155,
        9.33262154439441e157,
        9.42594775983835e159,
        9.61446671503512e161,
        9.90290071648618e163,
        1.02990167451456e166,
        1.08139675824029e168,
        1.14628056373471e170,
        1.22652020319614e172,
        1.32464181945183e174,
        1.44385958320249e176,
        1.58824554152274e178,
        1.76295255109024e180,
        1.97450685722107e182,
        2.23119274865981e184,
        2.54355973347219e186,
        2.92509369349301e188,
        3.3931086844519e190,
        3.96993716080872e192,
        4.68452584975429e194,
        5.5745857612076e196,
        6.68950291344912e198,
        8.09429852527344e200,
        9.8750442008336e202,
        1.21463043670253e205,
        1.50614174151114e207,
        1.88267717688893e209,
        2.37217324288005e211,
        3.01266001845766e213,
        3.8562048236258e215,
        4.97450422247729e217,
        6.46685548922047e219,
        8.47158069087882e221,
        1.118248651196e224,
        1.48727070609069e226,
        1.99294274616152e228,
        2.69047270731805e230,
        3.65904288195255e232,
        5.01288874827499e234,
        6.91778647261949e236,
        9.61572319694109e238,
        1.34620124757175e241,
        1.89814375907617e243,
        2.69536413788816e245,
        3.85437071718007e247,
        5.5502938327393e249,
        8.04792605747199e251,
        1.17499720439091e254,
        1.72724589045464e256,
        2.55632391787286e258,
        3.80892263763057e260,
        5.71338395644585e262,
        8.62720977423323e264,
        1.31133588568345e267,
        2.00634390509568e269,
        3.08976961384735e271,
        4.78914290146339e273,
        7.47106292628289e275,
        1.17295687942641e278,
        1.85327186949373e280,
        2.94670227249504e282,
        4.71472363599206e284,
        7.59070505394721e286,
        1.22969421873945e289,
        2.0044015765453e291,
        3.28721858553429e293,
        5.42391066613159e295,
        9.00369170577843e297,
        1.503616514865e300,
    ]
)


@ti.data_oriented
class SteinhardtBondOrientation:
    """This class is used to calculate a set of bond-orientational order parameters :math:`Q_{\\ell}` to characterize the local orientational order in atomic structures. We first compute the local order parameters as averages of the spherical harmonics :math:`Y_{\ell m}` for each neighbor:

    .. math:: \\bar{Y}_{\\ell m} = \\frac{1}{nnn}\\sum_{j = 1}^{nnn} Y_{\\ell m}\\bigl( \\theta( {\\bf r}_{ij} ), \\phi( {\\bf r}_{ij} ) \\bigr),

    where the summation goes over the :math:`nnn` nearest neighbor and the :math:`\\theta` and the :math:`\\phi` are the azimuthal and polar
    angles. Then we can obtain a rotationally invariant non-negative amplitude by summing over all the components of degree :math:`l`:

    .. math:: Q_{\\ell}  = \\sqrt{\\frac{4 \\pi}{2 \\ell  + 1} \\sum_{m = -\\ell }^{m = \\ell } \\bar{Y}_{\\ell m} \\bar{Y}^*_{\\ell m}}.

    For a FCC lattice with :math:`nnn=12`, :math:`Q_4 = \\sqrt{\\frac{7}{192}} \\approx 0.19094`. More numerical values for commonly encountered high-symmetry structures are listed in Table 1 of `J. Chem. Phys. 138, 044501 (2013) <https://aip.scitation.org/doi/abs/10.1063/1.4774084>`_, and all data can be reproduced by this class.

    If :math:`wlflag` is True, this class will compute the third-order invariants :math:`W_{\\ell}` for the same degrees as for the :math:`Q_{\\ell}` parameters:

    .. math:: W_{\\ell} = \\sum \\limits_{m_1 + m_2 + m_3 = 0} \\begin{pmatrix}\\ell & \\ell & \\ell \\\m_1 & m_2 & m_3\\end{pmatrix}\\bar{Y}_{\\ell m_1} \\bar{Y}_{\\ell m_2} \\bar{Y}_{\\ell m_3}.

    For FCC lattice with :math:`nnn=12`, :math:`W_4 = -\\sqrt{\\frac{14}{143}} \\left(\\frac{49}{4096}\\right) \\pi^{-3/2} \\approx -0.0006722136`.

    If :math:`wlhatflag` is true, the normalized third-order invariants :math:`\\hat{W}_{\\ell}` will be computed:

    .. math:: \\hat{W}_{\\ell} = \\frac{\\sum \\limits_{m_1 + m_2 + m_3 = 0} \\begin{pmatrix}\\ell & \\ell & \\ell \\\m_1 & m_2 & m_3\\end{pmatrix}\\bar{Y}_{\\ell m_1} \\bar{Y}_{\\ell m_2} \\bar{Y}_{\\ell m_3}}{\\left(\\sum \\limits_{m=-l}^{l} |\\bar{Y}_{\ell m}|^2 \\right)^{3/2}}.

    For FCC lattice with :math:`nnn=12`, :math:`\\hat{W}_4 = -\\frac{7}{3} \\sqrt{\\frac{2}{429}} \\approx -0.159317`. More numerical values of :math:`\\hat{W}_{\\ell}` can be found in Table 1 of `Phys. Rev. B 28, 784 <https://doi.org/10.1103/PhysRevB.28.784>`_, and all data can be reproduced by this class.

    .. hint:: If you use this class in your publication, you should cite the original paper:

      `Steinhardt P J, Nelson D R, Ronchetti M. Bond-orientational order in liquids and glasses[J]. Physical Review B, 1983, 28(2): 784. <https://doi.org/10.1103/PhysRevB.28.784>`_

    .. note:: This class is translated from that in `LAMMPS <https://docs.lammps.org/compute_orientorder_atom.html>`_.

    We also further implement the bond order to identify the solid or liquid state for lattice structure. For FCC structure, one can compute the normalized cross product:

    .. math:: s_\\ell(i,j) = \\frac{4\\pi}{2\\ell + 1} \\frac{\\sum_{m=-\\ell}^{\\ell} \\bar{Y}_{\\ell m}(i) \\bar{Y}_{\\ell m}^*(j)}{Q_\\ell(i) Q_\\ell(j)}.

    According to `J. Chem. Phys. 133, 244115 (2010) <https://doi.org/10.1063/1.3506838>`_, when :math:`s_6(i, j)` is larger than a threshold value (typically 0.7), the bond is regarded as a solid bond. Id the number of solid bond is larger than a threshold (6-8), the atom is considered as solid phase.

    .. hint:: If you use `identifySolidLiquid` function in this class in your publication, you should cite the original paper:

      `Filion L, Hermes M, Ni R, et al. Crystal nucleation of hard spheres using molecular dynamics, umbrella sampling, and forward flux sampling: A comparison of simulation techniques[J]. The Journal of chemical physics, 2010, 133(24): 244115. <https://doi.org/10.1063/1.3506838>`_

    Args:

        pos (np.ndarray): (:math:`N_p, 3`) particles positions.
        box (np.ndarray): (:math:`3, 2`) system box, must be rectangle.
        boundary (list, optional): boundary conditions, 1 is periodic and 0 is free boundary. Defaults to [1, 1, 1].
        verlet_list (np.ndarray, optional): (:math:`N_p, max\_neigh`) verlet_list[i, j] means j atom is a neighbor of i atom if j > -1. Defaults to None.
        distance_list (np.ndarray, optional): (:math:`N_p, max\_neigh`) distance_list[i, j] means distance between i and j atom. Defaults to None.
        neighbor_number (np.ndarray, optional): (:math:`N_p`) neighbor atoms number. Defaults to None.
        rc (float, optional): cutoff distance to find neighbors. Defaults to 0.0.
        qlist (list|int, optional): the list of order parameters to be computed, which should be a non-negative integer. Defaults to np.array([4, 6, 8, 10, 12], int).
        nnn (int, optional): the number of nearest neighbors used to calculate :math:`Q_{\ell}`. If :math:`nnn > 0`, the :math:`rc` has no effects, otherwise the summation will go over all neighbors within :math:`rc`. Defaults to 12.
        wlflag (bool, optional): whether calculate the third-order invariants :math:`W_{\ell}`. Defaults to False.
        wlhatflag (bool, optional): whether calculate the normalized third-order invariants :math:`\hat{W}_{\ell}`. If :math:`wlflag` is False, this parameter has no effect. Defaults to False.
        max_neigh (int, optional): a given maximum neighbor number per atoms. Defaults to 60.

    Outputs:
        - **qnarray** (np.ndarray) - (math:`N_p, len(qlist)*(1+wlflag+wlhatflag)`) consider the :math:`qlist=[4, 6]` and :math:`wlflag` and :math:`wlhatflag` is True, the columns of :math:`qnarray` are [:math:`Q_4, Q_6, W_4, W_6, \hat{W}_4, \hat{W}_6`].
        - **solidliquid** (np.ndarray) - (math:`N_p`), 1 indicates solid state and 0 indicates liquid state.

    Examples:
        >>> import mdapy as mp

        >>> mp.init()

        >>> FCC = mp.LatticeMaker(3.615, 'FCC', 10, 10, 10) # Create a FCC structure

        >>> FCC.compute() # Get atom positions

        >>> BO = SteinhardtBondOrientation(
                    FCC.pos,
                    FCC.box,
                    [1, 1, 1],
                    None,
                    None,
                    None,
                    0.0,
                    [4, 6, 8, 10, 12],
                    12,
                    wlflag=False,
                    wlhatflag=False,
                ) # Initialize BondOrder class

        >>> BO.compute() # Do the BondOrder computation.

        >>> BO.qnarray[0] # Check qnarray, it should be [0.19094067 0.57452428 0.40391458 0.01285704 0.60008306].

        >>> BO.identifySolidLiquid() # Identify solid/liquid state.

        >>> BO.solidliquid[0] # Should be 1, that is solid.
    """

    def __init__(
        self,
        pos,
        box,
        boundary=[1, 1, 1],
        verlet_list=None,
        distance_list=None,
        neighbor_number=None,
        rc=0.0,
        qlist=np.array([4, 6, 8, 10, 12], int),
        nnn=12,
        wlflag=False,
        wlhatflag=False,
        max_neigh=60,
    ):
        self.rc = rc
        self.nnn = nnn
        if self.nnn > 0:
            self.rc = 1000000000.0
            repeat = _check_repeat_nearest(pos, box, boundary)
        else:
            assert self.rc > 0
            repeat = _check_repeat_cutoff(box, boundary, self.rc)

        if pos.dtype != np.float64:
            pos = pos.astype(np.float64)
        if box.dtype != np.float64:
            box = box.astype(np.float64)
        self.old_N = None
        if sum(repeat) == 3:
            self.pos = pos
            if box.shape == (3, 2):
                self.box = np.zeros((4, 3), dtype=box.dtype)
                self.box[0, 0], self.box[1, 1], self.box[2, 2] = box[:, 1] - box[:, 0]
                self.box[-1] = box[:, 0]
            elif box.shape == (4, 3):
                self.box = box
        else:
            self.old_N = pos.shape[0]
            repli = Replicate(pos, box, *repeat)
            repli.compute()
            self.pos = repli.pos
            self.box = repli.box

        assert self.box[0, 1] == 0
        assert self.box[0, 2] == 0
        assert self.box[1, 2] == 0
        self.box_length = ti.Vector([np.linalg.norm(self.box[i]) for i in range(3)])
        self.rec = True
        if self.box[1, 0] != 0 or self.box[2, 0] != 0 or self.box[2, 1] != 0:
            self.rec = False
        self.boundary = ti.Vector([int(boundary[i]) for i in range(3)])

        self.verlet_list = verlet_list
        self.distance_list = distance_list
        self.neighbor_number = neighbor_number
        if isinstance(qlist, int):
            self.qlist = np.array([qlist], int)
        elif isinstance(qlist, list):
            self.qlist = np.array(qlist, int)
        elif isinstance(qlist, tuple):
            self.qlist = np.array(qlist, int)
        elif isinstance(qlist, np.ndarray):
            self.qlist = qlist.astype(int)
        else:
            raise "qlist should be a non-negative integer or List[int]|Tuple[int]|np.array[int]."
        for i in self.qlist:
            assert (
                i >= 0
            ), "qlist should be a non-negative integer or List[int]|Tuple[int]|np.array[int]."

        self.wlflag = wlflag
        self.wlhatflag = wlhatflag
        self.nqlist = self.qlist.shape[0]
        ncol = self.qlist.shape[0]
        if self.wlflag:
            ncol += self.nqlist
            if self.wlhatflag:
                ncol += self.nqlist
        self.ncol = ncol
        self.nfac_table = ti.field(dtype=ti.f64, shape=nfac_table_numpy.shape)
        self.nfac_table.from_numpy(nfac_table_numpy)
        self.if_compute = False
        self.max_neigh = max_neigh

    @ti.func
    def _factorial(self, n: int) -> ti.f64:
        # if n >= 0 and n < self.nfac_table.shape[0]:
        return self.nfac_table[n]

    @ti.kernel
    def _init_clebsch_gordan(
        self, cglist: ti.types.ndarray(), qlist: ti.types.ndarray()
    ):
        idxcg_count = 0
        for il in range(self.nqlist):
            l = qlist[il]
            for m1 in range(2 * l + 1):
                aa2 = m1 - l
                for m2 in range(ti.max(0, l - m1), ti.min(2 * l + 1, 3 * l - m1 + 1)):
                    bb2 = m2 - l
                    m = aa2 + bb2 + l
                    sums = ti.f64(0.0)
                    for z in range(
                        ti.max(0, ti.max(-aa2, bb2)),
                        ti.min(l, ti.min(l - aa2, l + bb2)) + 1,
                    ):
                        ifac = 1
                        if z % 2:
                            ifac = -1
                        sums += ifac / (
                            self._factorial(z)
                            * self._factorial(l - z)
                            * self._factorial(l - aa2 - z)
                            * self._factorial(l + bb2 - z)
                            * self._factorial(aa2 + z)
                            * self._factorial(-bb2 + z)
                        )
                    cc2 = m - l
                    sfaccg = ti.sqrt(
                        self._factorial(l + aa2)
                        * self._factorial(l - aa2)
                        * self._factorial(l + bb2)
                        * self._factorial(l - bb2)
                        * self._factorial(l + cc2)
                        * self._factorial(l - cc2)
                        * (2 * l + 1)
                    )
                    sfac1 = self._factorial(3 * l + 1)
                    sfac2 = self._factorial(l)
                    dcg = ti.sqrt(sfac2 * sfac2 * sfac2 / sfac1)
                    cglist[idxcg_count] = sums * dcg * sfaccg
                    idxcg_count += 1

    @ti.func
    def _associated_legendre(self, l: int, m: int, x: float) -> float:
        res = ti.f64(0.0)
        if l >= m:
            p, pm1, pm2 = ti.f64(1.0), ti.f64(0.0), ti.f64(0.0)
            if m != 0:
                sqx = ti.sqrt(1.0 - x * x)
                for i in range(1, m + 1):
                    p *= (2 * i - 1) * sqx
            for i in range(m + 1, l + 1):
                pm2 = pm1
                pm1 = p
                p = ((2 * i - 1) * x * pm1 - (i + m - 1) * pm2) / (i - m)
            res = p
        return res

    @ti.func
    def _polar_prefactor(self, l: int, m: int, costheta: float) -> float:
        mabs = ti.abs(m)
        prefactor = ti.f64(1.0)
        for i in range(l - mabs + 1, l + mabs + 1):
            prefactor *= i
        prefactor = ti.sqrt(
            (2 * l + 1) / (4 * ti.math.pi * prefactor)
        ) * self._associated_legendre(l, mabs, costheta)
        if (m < 0) and (m % 2):
            prefactor = -prefactor
        return prefactor

    @ti.func
    def _pbc_rec(self, rij):
        for m in ti.static(range(3)):
            if self.boundary[m]:
                dx = rij[m]
                x_size = self.box_length[m]
                h_x_size = x_size * 0.5
                if dx > h_x_size:
                    dx = dx - x_size
                if dx <= -h_x_size:
                    dx = dx + x_size
                rij[m] = dx
        return rij

    @ti.func
    def _pbc(self, rij, box: ti.types.ndarray(element_dim=1)) -> ti.math.vec3:
        nz = rij[2] / box[2][2]
        ny = (rij[1] - nz * box[2][1]) / box[1][1]
        nx = (rij[0] - ny * box[1][0] - nz * box[2][0]) / box[0][0]
        n = ti.Vector([nx, ny, nz])
        for i in ti.static(range(3)):
            if self.boundary[i] == 1:
                if n[i] > 0.5:
                    n[i] -= 1
                elif n[i] < -0.5:
                    n[i] += 1
        return n[0] * box[0] + n[1] * box[1] + n[2] * box[2]

    @ti.kernel
    def _get_idx(self, qlist: ti.types.ndarray()) -> int:
        idxcg_count = 0
        ti.loop_config(serialize=True)
        for il in range(self.nqlist):
            l = qlist[il]
            for m1 in range(2 * l + 1):
                for _ in range(ti.max(0, l - m1), ti.min(2 * l + 1, 3 * l - m1 + 1)):
                    idxcg_count += 1
        return idxcg_count

    @ti.kernel
    def _compute(
        self,
        pos: ti.types.ndarray(dtype=ti.math.vec3),
        box: ti.types.ndarray(element_dim=1),
        verlet_list: ti.types.ndarray(),
        distance_list: ti.types.ndarray(),
        neighbor_number: ti.types.ndarray(),
        qlist: ti.types.ndarray(),
        qnm_r: ti.types.ndarray(),
        qnm_i: ti.types.ndarray(),
        qnarray: ti.types.ndarray(),
        cglist: ti.types.ndarray(),
    ):
        MY_EPSILON = 2.220446049250313e-15
        N = pos.shape[0]
        nqlist = self.nqlist
        K = 0
        for i in range(N):
            nneigh = 0
            # Make sure only iterate the nnn neighbors!
            if self.nnn > 0:
                K = self.nnn
            else:
                K = neighbor_number[i]
            for jj in range(K):
                j = verlet_list[i, jj]
                r = pos[i] - pos[j]
                if ti.static(self.rec):
                    r = self._pbc_rec(r)
                else:
                    r = self._pbc(r, box)
                rmag = distance_list[i, jj]
                if rmag > MY_EPSILON and rmag <= self.rc:
                    nneigh += 1
                    costheta = r[2] / rmag
                    expphi_r = r[0]
                    expphi_i = r[1]
                    rxymag = ti.sqrt(expphi_r * expphi_r + expphi_i * expphi_i)
                    if rxymag <= MY_EPSILON:
                        expphi_r = 1.0
                        expphi_i = 0.0
                    else:
                        rxymaginv = 1.0 / rxymag
                        expphi_r *= rxymaginv
                        expphi_i *= rxymaginv
                    for il in range(nqlist):
                        l = qlist[il]
                        qnm_r[i, il, l] += self._polar_prefactor(l, 0, costheta)
                        expphim_r = expphi_r
                        expphim_i = expphi_i
                        for m in range(1, l + 1):
                            prefactor = self._polar_prefactor(l, m, costheta)
                            c_r = prefactor * expphim_r
                            c_i = prefactor * expphim_i
                            qnm_r[i, il, m + l] += c_r
                            qnm_i[i, il, m + l] += c_i
                            if m & 1:
                                qnm_r[i, il, -m + l] -= c_r
                                qnm_i[i, il, -m + l] += c_i
                            else:
                                qnm_r[i, il, -m + l] += c_r
                                qnm_i[i, il, -m + l] -= c_i
                            tmp_r = expphim_r * expphi_r - expphim_i * expphi_i
                            tmp_i = expphim_r * expphi_i + expphim_i * expphi_r
                            expphim_r = tmp_r
                            expphim_i = tmp_i

            facn = 1.0 / nneigh
            for il in range(nqlist):
                l = qlist[il]
                for m in range(2 * l + 1):
                    qnm_r[i, il, m] *= facn
                    qnm_i[i, il, m] *= facn

            for il in range(nqlist):
                l = qlist[il]
                qnormfac = ti.sqrt(4 * ti.math.pi / (2 * l + 1))
                qm_sum = ti.f64(0.0)
                for m in range(2 * l + 1):
                    qm_sum += (
                        qnm_r[i, il, m] * qnm_r[i, il, m]
                        + qnm_i[i, il, m] * qnm_i[i, il, m]
                    )
                qnarray[i, il] = qnormfac * ti.sqrt(qm_sum)

            if self.wlflag:
                idxcg_count = 0
                for il in range(nqlist):
                    l = qlist[il]
                    wlsum = ti.f64(0.0)
                    for m1 in range(2 * l + 1):
                        for m2 in range(
                            ti.max(0, l - m1), ti.min(2 * l + 1, 3 * l - m1 + 1)
                        ):
                            m = m1 + m2 - l
                            qm1qm2_r = (
                                qnm_r[i, il, m1] * qnm_r[i, il, m2]
                                - qnm_i[i, il, m1] * qnm_i[i, il, m2]
                            )
                            qm1qm2_i = (
                                qnm_r[i, il, m1] * qnm_i[i, il, m2]
                                + qnm_i[i, il, m1] * qnm_r[i, il, m2]
                            )
                            wlsum += (
                                qm1qm2_r * qnm_r[i, il, m] + qm1qm2_i * qnm_i[i, il, m]
                            ) * cglist[idxcg_count]
                            idxcg_count += 1
                    qnarray[i, nqlist + il] = wlsum / ti.sqrt(2 * l + 1)

                if self.wlhatflag:
                    idxcg_count = 0
                    for il in range(nqlist):
                        l = qlist[il]
                        wlsum = ti.f64(0.0)
                        for m1 in range(2 * l + 1):
                            for m2 in range(
                                ti.max(0, l - m1), ti.min(2 * l + 1, 3 * l - m1 + 1)
                            ):
                                m = m1 + m2 - l
                                qm1qm2_r = (
                                    qnm_r[i, il, m1] * qnm_r[i, il, m2]
                                    - qnm_i[i, il, m1] * qnm_i[i, il, m2]
                                )
                                qm1qm2_i = (
                                    qnm_r[i, il, m1] * qnm_i[i, il, m2]
                                    + qnm_i[i, il, m1] * qnm_r[i, il, m2]
                                )
                                wlsum += (
                                    qm1qm2_r * qnm_r[i, il, m]
                                    + qm1qm2_i * qnm_i[i, il, m]
                                ) * cglist[idxcg_count]
                                idxcg_count += 1
                        if qnarray[i, il] >= 1e-6:
                            qnormfac = ti.sqrt(4 * ti.math.pi / (2 * l + 1))
                            qnfac = qnormfac / qnarray[i, il]
                            qnarray[i, nqlist + nqlist + il] = (
                                wlsum / ti.sqrt(2 * l + 1) * (qnfac * qnfac * qnfac)
                            )

    def compute(self):
        """Do the Steinhardt Bondorder calculation."""
        qmax = self.qlist.max()
        self.qnm_r = np.zeros((self.pos.shape[0], self.nqlist, 2 * qmax + 1))
        self.qnm_i = np.zeros_like(self.qnm_r)
        idxcg_count = self._get_idx(self.qlist)
        cglist = np.zeros(idxcg_count)
        if self.wlflag or self.wlhatflag:
            self._init_clebsch_gordan(cglist, self.qlist)
        self.qnarray = np.zeros((self.pos.shape[0], self.ncol))

        if self.verlet_list is None or self.distance_list is None:
            if self.nnn > 0:
                kdt = NearestNeighbor(self.pos, self.box, self.boundary)
                self.distance_list, self.verlet_list = kdt.query_nearest_neighbors(
                    self.nnn
                )
                self.neighbor_number = np.ones(self.pos.shape[0], int) * self.nnn
            else:
                assert self.rc > 0.0, "one of rc and nnn at least is positive."
                neigh = Neighbor(
                    self.pos, self.box, self.rc, self.boundary, max_neigh=self.max_neigh
                )
                neigh.compute()
                self.distance_list, self.verlet_list, self.neighbor_number = (
                    neigh.verlet_list,
                    neigh.distance_list,
                    neigh.neighbor_number,
                )
        else:
            if self.nnn > 0:
                assert (
                    self.neighbor_number.min() >= self.nnn
                ), "The minimum of neighbor_number should be larger than nnn."
                # self.neighbor_number = np.ones(self.pos.shape[0], int) * self.nnn
        self._compute(
            self.pos,
            self.box,
            self.verlet_list,
            self.distance_list,
            self.neighbor_number,
            self.qlist,
            self.qnm_r,
            self.qnm_i,
            self.qnarray,
            cglist,
        )
        if self.old_N is not None:
            self.old_qnarray = self.qnarray.copy()
            self.qnarray = np.ascontiguousarray(self.qnarray[: self.old_N])
        self.if_compute = True

    @ti.kernel
    def _identifySolidLiquid(
        self,
        Q6index: int,
        Q6: ti.types.ndarray(),
        verlet_list: ti.types.ndarray(),
        distance_list: ti.types.ndarray(),
        qnm_r: ti.types.ndarray(),
        qnm_i: ti.types.ndarray(),
        neighbor_number: ti.types.ndarray(),
        threshold: float,
        n_bond: int,
        solidliquid: ti.types.ndarray(),
    ):
        for i in range(verlet_list.shape[0]):
            n_solid_bond = 0
            for jj in range(neighbor_number[i]):
                j = verlet_list[i, jj]
                r = distance_list[i, jj]
                sij_sum = ti.f64(0.0)
                if r <= self.rc:
                    for m in range(13):
                        sij_sum += (
                            qnm_r[i, Q6index, m] * qnm_r[j, Q6index, m]
                            + qnm_i[i, Q6index, m] * qnm_i[j, Q6index, m]
                        )

                sij_sum = sij_sum / Q6[i] / Q6[j] * 4 * ti.math.pi / 13
                if sij_sum > threshold:
                    n_solid_bond += 1
            if n_solid_bond >= n_bond:
                solidliquid[i] = 1

        for i in range(verlet_list.shape[0]):
            if solidliquid[i] == 1:
                n_solid = 0
                for jj in range(neighbor_number[i]):
                    j = verlet_list[i, jj]
                    if solidliquid[j] == 1:
                        n_solid = 1
                        break
                if n_solid == 0:
                    solidliquid[i] = 0

    def identifySolidLiquid(self, threshold=0.7, n_bond=7):
        """Identify the solid/liquid phase. Make sure you computed the 6 in qlist.

        Args:
            threshold (float, optional): threshold value to determine the solid bond. Defaults to 0.7.
            n_bond (int, optional): threshold to determine the solid atoms. Defaults to 7.
        """
        assert 6 in self.qlist, "You must calculate Q6 bond order."
        if not self.if_compute:
            self.compute()
        Q6index = int(np.where(self.qlist == 6)[0][0])
        if self.old_N is not None:
            Q6 = np.ascontiguousarray(self.old_qnarray[:, Q6index])
        else:
            Q6 = np.ascontiguousarray(self.qnarray[:, Q6index])
        self.solidliquid = np.zeros(self.pos.shape[0], int)
        self._identifySolidLiquid(
            Q6index,
            Q6,
            self.verlet_list,
            self.distance_list,
            self.qnm_r,
            self.qnm_i,
            self.neighbor_number,
            threshold,
            n_bond,
            self.solidliquid,
        )
        if self.old_N is not None:
            self.solidliquid = np.ascontiguousarray(self.solidliquid[: self.old_N])


if __name__ == "__main__":
    from lattice_maker import LatticeMaker
    from time import time

    ti.init()
    start = time()
    FCC = LatticeMaker(3.615, "FCC", 10, 10, 10)
    FCC.compute()
    print(f"Build FCC time cost: {time()-start} s. Atom number: {FCC.N}.")

    start = time()
    BO = SteinhardtBondOrientation(
        FCC.pos,
        FCC.box,
        [1, 1, 1],
        None,
        None,
        None,
        0.0,
        [4, 6, 8, 10],
        12,
        wlflag=True,
        wlhatflag=False,
    )
    BO.compute()
    print(f"BO time cost: {time()-start} s.")
    print(BO.qnarray[0])
    start = time()
    BO.identifySolidLiquid()
    print(f"SolidLiquid time cost: {time()-start} s.")
    print(BO.solidliquid[:10])
