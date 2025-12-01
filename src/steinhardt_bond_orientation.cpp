#include "type.h"
#include "box.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <vector>
#include <cmath>
#include <omp.h>
#include <stdio.h>

static constexpr double h_factorial[168] = {
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
};

inline double _factorial(int n)
{
    return h_factorial[n];
}

void _init_clebsch_gordan(std::vector<double> &cglist, const int *llist, const int ndegrees_)
{
    int idxcg_count{0};
    for (int il = 0; il < ndegrees_; il++)
    {
        const int l = llist[il];
        for (int m1 = 0; m1 < 2 * l + 1; m1++)
        {
            int aa2 = m1 - l;
            for (int m2 = std::max(0, l - m1); m2 < std::min(2 * l + 1, 3 * l - m1 + 1); m2++)
            {
                int bb2 = m2 - l;
                int m = aa2 + bb2 + l;
                double sums{0.0};
                for (int z = std::max(0, std::max(-aa2, bb2)); z < std::min(l, std::min(l - aa2, l + bb2)) + 1; z++)
                {
                    int ifac = 1;
                    if (z % 2)
                    {
                        ifac = -1;
                    }
                    sums += ifac / (_factorial(z) * _factorial(l - z) * _factorial(l - aa2 - z) *
                                    _factorial(l + bb2 - z) * _factorial(aa2 + z) * _factorial(-bb2 + z));
                }
                int cc2 = m - l;
                double sfaccg = sqrt(
                    _factorial(l + aa2) * _factorial(l - aa2) * _factorial(l + bb2) * _factorial(l - bb2) *
                    _factorial(l + cc2) * _factorial(l - cc2) * (2 * l + 1));
                double sfac1 = _factorial(3 * l + 1);
                double sfac2 = _factorial(l);
                double dcg = sqrt(sfac2 * sfac2 * sfac2 / sfac1);
                cglist[idxcg_count] = sums * dcg * sfaccg;
                idxcg_count++;
            }
        }
    }
}

int _get_idx(const int *llist, const int ndegrees_)
{
    int idxcg_count{0};
    for (int il = 0; il < ndegrees_; il++)
    {
        const int l = llist[il];
        for (int m1 = 0; m1 < 2 * l + 1; m1++)
        {
            for (int m2 = std::max(0, l - m1); m2 < std::min(2 * l + 1, 3 * l - m1 + 1); m2++)
            {
                idxcg_count++;
            }
        }
    }
    return idxcg_count;
}

inline double _associated_legendre(int l, int m, double x)
{
    double res{0.0};
    if (l >= m)
    {
        double p{1.0};
        double pm1{0.0};
        double pm2{0.0};
        if (m != 0)
        {
            double sqx{sqrt(1.0 - x * x)};
            for (int i = 1; i < m + 1; i++)
            {
                p *= (2 * i - 1) * sqx;
            }
        }
        for (int i = m + 1; i < l + 1; i++)
        {
            pm2 = pm1;
            pm1 = p;
            p = ((2 * i - 1) * x * pm1 - (i + m - 1) * pm2) / (i - m);
        }
        res = p;
    }
    return res;
}

inline double _polar_prefactor(int l, int m, double costheta)
{
    constexpr double My_PI = 3.14159265358979323846;
    int mabs = abs(m);
    double prefactor{1.0};
    for (int i = l - mabs + 1; i < l + mabs + 1; i++)
    {
        prefactor *= i;
    }
    prefactor = sqrt((2 * l + 1) / (4 * My_PI * prefactor)) * _associated_legendre(l, mabs, costheta);

    if ((m < 0) & (m % 2))
    {
        prefactor = -prefactor;
    }
    return prefactor;
}

void _compute_ql(
    const int N,
    const Box &box,
    const int nnn,
    const int max_neigh,
    const int *NN,
    const int *NL,
    const double *distance_list,
    const double *x,
    const double *y,
    const double *z,
    const int *llist,
    const int ndegrees,
    const int lmax,
    double *qlm_r,
    double *qlm_i,
    const bool use_voronoi,
    const double rc,
    const bool use_weight,
    const double *weight,
    const bool average,
    double *aqlm_r, // 这个会在函数内部被赋值
    double *aqlm_i, // 这个会在函数内部被赋值
    const int ncol,
    double *qnarray,
    const bool wl,
    const bool wlhat,
    const double *cglist)
{
    constexpr double MY_EPSILON{1e-15};
    constexpr double My_PI = 3.14159265358979323846;
    const int nz{lmax * 2 + 1};

// 第一阶段：计算 qlm - 并行化
#pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < N; ++i)
    {
        double weight_val{0.0};
        const double x1 = x[i];
        const double y1 = y[i];
        const double z1 = z[i];
        int i_neigh = NN[i];
        if ((!use_voronoi) && (nnn > 0))
        {
            i_neigh = nnn;
        }

        const int base_idx = i * max_neigh;
        for (int jj = 0; jj < i_neigh; jj++)
        {
            const int index = base_idx + jj;
            const int j = NL[index];
            if (j < 0)
            {
                continue;
            }

            double xij = x[j] - x1;
            double yij = y[j] - y1;
            double zij = z[j] - z1;
            box.pbc(xij, yij, zij);
            const double rmag = distance_list[index];

            if ((rmag > MY_EPSILON) && (rmag <= rc))
            {
                double wij = use_weight ? weight[index] : 1.0;
                weight_val += wij;

                const double rmag_inv = 1.0 / rmag;
                double costheta = zij * rmag_inv;
                double expphi_r = xij;
                double expphi_i = yij;
                double rxymag_sq = expphi_r * expphi_r + expphi_i * expphi_i;

                if (rxymag_sq < MY_EPSILON * MY_EPSILON)
                {
                    expphi_r = 1.0;
                    expphi_i = 0.0;
                }
                else
                {
                    double rxymaginv = 1.0 / sqrt(rxymag_sq);
                    expphi_r *= rxymaginv;
                    expphi_i *= rxymaginv;
                }

                const int base_qlm_idx = i * (ndegrees * nz);
                for (int il = 0; il < ndegrees; il++)
                {
                    const int l = llist[il];
                    const int il_offset = base_qlm_idx + il * nz;

                    qlm_r[il_offset + l] += wij * _polar_prefactor(l, 0, costheta);

                    double expphim_r = expphi_r;
                    double expphim_i = expphi_i;

                    for (int m = 1; m < l + 1; m++)
                    {
                        double prefactor = _polar_prefactor(l, m, costheta);
                        double c_r = prefactor * expphim_r;
                        double c_i = prefactor * expphim_i;

                        int index1 = il_offset + m + l;
                        int index2 = il_offset - m + l;

                        const double w_c_r = wij * c_r;
                        const double w_c_i = wij * c_i;

                        qlm_r[index1] += w_c_r;
                        qlm_i[index1] += w_c_i;

                        if (m & 1)
                        {
                            qlm_r[index2] -= w_c_r;
                            qlm_i[index2] += w_c_i;
                        }
                        else
                        {
                            qlm_r[index2] += w_c_r;
                            qlm_i[index2] -= w_c_i;
                        }

                        double tmp_r = expphim_r * expphi_r - expphim_i * expphi_i;
                        double tmp_i = expphim_r * expphi_i + expphim_i * expphi_r;
                        expphim_r = tmp_r;
                        expphim_i = tmp_i;
                    }
                }
            }
        }

        // 归一化
        const double facn = 1.0 / weight_val;
        const int base_qlm_idx = i * (ndegrees * nz);
        for (int il = 0; il < ndegrees; il++)
        {
            const int l = llist[il];
            const int il_offset = base_qlm_idx + il * nz;
            const int mmax = 2 * l + 1;
            for (int m = 0; m < mmax; m++)
            {
                int index = il_offset + m;
                qlm_r[index] *= facn;
                qlm_i[index] *= facn;
            }
        }
    }

    // 第二阶段：如果需要平均化，先保存当前的 qlm 值到 aqlm
    if (average)
    {
        // ✅ 正确：在这里复制已经计算好的 qlm 值
        const int total_size = N * ndegrees * nz;
#pragma omp parallel for
        for (int i = 0; i < total_size; i++)
        {
            aqlm_r[i] = qlm_r[i];
            aqlm_i[i] = qlm_i[i];
        }

// 然后进行平均化操作
#pragma omp parallel for schedule(dynamic, 16)
        for (int i = 0; i < N; ++i)
        {
            int i_neigh = NN[i];
            if ((!use_voronoi) && (nnn > 0))
            {
                i_neigh = nnn;
            }

            int N_neigh = 1;
            const int base_idx = i * max_neigh;
            const int base_qlm_idx = i * (ndegrees * nz);

            for (int jj = 0; jj < i_neigh; jj++)
            {
                const int j = NL[base_idx + jj];
                if (j < 0)
                {
                    continue;
                }

                const int j_base = j * (ndegrees * nz);
                for (int il = 0; il < ndegrees; il++)
                {
                    const int l = llist[il];
                    const int i_offset = base_qlm_idx + il * nz;
                    const int j_offset = j_base + il * nz;
                    const int mmax = 2 * l + 1;

                    for (int m = 0; m < mmax; m++)
                    {
                        qlm_r[i_offset + m] += aqlm_r[j_offset + m];
                        qlm_i[i_offset + m] += aqlm_i[j_offset + m];
                    }
                }
                N_neigh++;
            }

            const double inv_N_neigh = 1.0 / N_neigh;
            for (int il = 0; il < ndegrees; il++)
            {
                const int l = llist[il];
                const int i_offset = base_qlm_idx + il * nz;
                const int mmax = 2 * l + 1;

                for (int m = 0; m < mmax; m++)
                {
                    qlm_r[i_offset + m] *= inv_N_neigh;
                    qlm_i[i_offset + m] *= inv_N_neigh;
                }
            }
        }
    }

// 第三阶段：计算 qn 和 wl - 并行化
#pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < N; ++i)
    {
        const int base_qlm_idx = i * (ndegrees * nz);
        const int base_qn_idx = i * ncol;

        for (int il = 0; il < ndegrees; il++)
        {
            const int l = llist[il];
            const double qnormfac = sqrt(4 * My_PI / (2 * l + 1));
            double qm_sum = 0.0;

            const int il_offset = base_qlm_idx + il * nz;
            const int mmax = 2 * l + 1;

            for (int m = 0; m < mmax; m++)
            {
                int index = il_offset + m;
                qm_sum += qlm_r[index] * qlm_r[index] + qlm_i[index] * qlm_i[index];
            }
            qnarray[base_qn_idx + il] = qnormfac * sqrt(qm_sum);
        }

        if (wl | wlhat)
        {
            int idxcg_count{0};
            for (int il = 0; il < ndegrees; il++)
            {
                const int l = llist[il];
                double wlsum{0.0};
                const int il_offset = base_qlm_idx + il * nz;

                for (int m1 = 0; m1 < 2 * l + 1; m1++)
                {
                    const int m2_start = std::max(0, l - m1);
                    const int m2_end = std::min(2 * l + 1, 3 * l - m1 + 1);

                    for (int m2 = m2_start; m2 < m2_end; m2++)
                    {
                        int m = m1 + m2 - l;
                        int index1 = il_offset + m1;
                        int index2 = il_offset + m2;
                        int index3 = il_offset + m;

                        double qm1qm2_r = qlm_r[index1] * qlm_r[index2] - qlm_i[index1] * qlm_i[index2];
                        double qm1qm2_i = qlm_r[index1] * qlm_i[index2] + qlm_i[index1] * qlm_r[index2];
                        wlsum += (qm1qm2_r * qlm_r[index3] + qm1qm2_i * qlm_i[index3]) * cglist[idxcg_count];
                        idxcg_count++;
                    }
                }

                const double wl_factor = wlsum / sqrt(2 * l + 1.0);
                if (wl)
                {
                    qnarray[base_qn_idx + il + ndegrees] = wl_factor;
                }
                if (wlhat)
                {
                    const double qn_val = qnarray[base_qn_idx + il];
                    if (qn_val > MY_EPSILON)
                    {
                        double qnormfac = sqrt(4 * My_PI / (2 * l + 1));
                        double qnfac = qnormfac / qn_val;
                        double qnfac3 = qnfac * qnfac * qnfac;
                        qnarray[base_qn_idx + il + wl * ndegrees + ndegrees] = wl_factor * qnfac3;
                    }
                }
            }
        }
    }
}

void identifySolidLiquid(
    const int Q6index,
    const ROneArrayD Q6_py,
    const RTwoArrayI verlet_list_py,
    const RTwoArrayD distance_list_py,
    const ROneArrayI neighbor_number_py,
    const RThreeArrayD qlm_r_py,
    const RThreeArrayD qlm_i_py,
    const double threshold,
    const int n_bond,
    OneArrayI solidliquid_py,
    OneArrayI nbond_py,
    const bool use_voronoi,
    const int nnn,
    const double rc)
{
    const int N = verlet_list_py.shape(0);
    auto verlet_list = verlet_list_py.view();
    auto distance_list = distance_list_py.view();
    const int *neighbor_number = neighbor_number_py.data();
    auto qlm_i = qlm_i_py.view();
    auto qlm_r = qlm_r_py.view();
    const double *Q6 = Q6_py.data();
    int *solidliquid = solidliquid_py.data();
    int *nbond = nbond_py.data();
    constexpr double My_PI = 3.14159265358979323846;
#pragma omp parallel for
    for (int i = 0; i < N; ++i)
    {
        int n_solid_bond = 0;

        int i_neigh = neighbor_number[i];
        if ((!use_voronoi) && (nnn > 0))
        {
            i_neigh = nnn;
        }
        for (int jj = 0; jj < i_neigh; ++jj)
        {
            const int j = verlet_list(i, jj);
            if (j < 0)
            {
                continue;
            }
            const double r = distance_list(i, jj);
            if (r > rc)
            {
                continue;
            }
            double sij_sum = 0.0;
            for (int m = 0; m < 13; ++m)
            {
                sij_sum += qlm_r(i, Q6index, m) * qlm_r(j, Q6index, m) + qlm_i(i, Q6index, m) * qlm_i(j, Q6index, m);
            }
            sij_sum = sij_sum / Q6[i] / Q6[j] * 4 * My_PI / 13;
            if (sij_sum > threshold)
            {
                n_solid_bond++;
            }
            // printf("i: %d, sijsum: %f, sb: %d\n", i, sij_sum, n_solid_bond);
        }
        if (n_solid_bond >= n_bond)
        {
            solidliquid[i] = 1;
        }
        nbond[i] = n_solid_bond;
    }
#pragma omp parallel for
    for (int i = 0; i < N; ++i)
    {
        if (solidliquid[i] == 1)
        {
            int n_solid = 0;
            int i_neigh = neighbor_number[i];
            if ((!use_voronoi) && (nnn > 0))
            {
                i_neigh = nnn;
            }
            for (int jj = 0; jj < i_neigh; ++jj)
            {
                const int j = verlet_list(i, jj);
                if (j < 0)
                {
                    continue;
                }
                if (solidliquid[j] == 1)
                {
                    n_solid = 1;
                    break;
                }
            }
            if (n_solid == 0)
            {
                solidliquid[i] = 0;
            }
        }
    }
}

void get_sq(
    const ROneArrayD x_py,
    const ROneArrayD y_py,
    const ROneArrayD z_py,
    const RTwoArrayD box_py,
    const ROneArrayD origin,
    const ROneArrayI boundary,
    const RTwoArrayI verlet_list_py,
    const RTwoArrayD distance_list_py,
    const ROneArrayI neighbor_number_py,
    const RTwoArrayD weight_py,
    const ROneArrayI llist_py,
    const int nnn_,
    const int lmax_,
    const bool wl_,
    const bool wlhat_,
    const bool average_,
    const bool use_voronoi,
    const double rc,
    const bool use_weight,
    ThreeArrayD qlm_r_py,
    ThreeArrayD qlm_i_py,
    TwoArrayD qnarray_py)
{
    const Box box = get_box(box_py, origin, boundary);
    const int ndegrees_ = llist_py.shape(0);
    const int num_atoms_ = x_py.shape(0);
    const int *llist = llist_py.data();
    const int *NN = neighbor_number_py.data();
    const int *NL = verlet_list_py.data();
    const double *distance_list = distance_list_py.data();
    const int max_neigh = verlet_list_py.shape(1);
    const double *x = x_py.data();
    const double *y = y_py.data();
    const double *z = z_py.data();
    const double *weight = weight_py.data();
    double *qlm_r = qlm_r_py.data();
    double *qlm_i = qlm_i_py.data();
    double *qnarray = qnarray_py.data();

    int ncol_ = ndegrees_;
    if (wl_)
    {
        ncol_ += ndegrees_;
    }
    if (wlhat_)
    {
        ncol_ += ndegrees_;
    }

    std::vector<double> cglist;
    if (wl_ | wlhat_)
    {
        int idxcg_count = _get_idx(llist, ndegrees_);
        cglist.resize(idxcg_count, 0.0);
        _init_clebsch_gordan(cglist, llist, ndegrees_);
    }
    else
    {
        cglist.resize(1, 0.0);
    }

    std::vector<double> aqlm_r;
    std::vector<double> aqlm_i;

    if (average_)
    {
        // ✅ 只分配空间，不初始化（会在 _compute_ql 内部赋值）
        aqlm_r.resize(num_atoms_ * ndegrees_ * (2 * lmax_ + 1));
        aqlm_i.resize(num_atoms_ * ndegrees_ * (2 * lmax_ + 1));
    }
    else
    {
        aqlm_r.resize(1, 0.0);
        aqlm_i.resize(1, 0.0);
    }

    _compute_ql(
        num_atoms_,
        box,
        nnn_,
        max_neigh,
        NN,
        NL,
        distance_list,
        x,
        y,
        z,
        llist,
        ndegrees_,
        lmax_,
        qlm_r,
        qlm_i,
        use_voronoi,
        rc,
        use_weight,
        weight,
        average_,
        aqlm_r.data(),
        aqlm_i.data(),
        ncol_,
        qnarray,
        wl_,
        wlhat_,
        cglist.data());
}

NB_MODULE(_sbo, m)
{
    m.def("get_sq", &get_sq);
    m.def("identifySolidLiquid", &identifySolidLiquid);
}