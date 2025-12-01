/*Copyright (c) 2022 PM Larsen

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include "ptm_alloy_types.h"
#include "ptm_constants.h"
#include "ptm_convex_hull_incremental.h"
#include "ptm_deformation_gradient.h"
#include "ptm_functions.h"
#include "ptm_graph_data.h"
#include "ptm_initialize_data.h"
#include "ptm_multishell.h"
#include "ptm_neighbour_ordering.h"
#include "ptm_normalize_vertices.h"
#include "ptm_polar.h"
#include "ptm_structure_matcher.h"
#include <cassert>
#include <cstring>


static double calculate_interatomic_distance(int type, double scale) {
    assert(type >= 1 && type <= 8);

    // these values should be equal to norm(template[1])
    double c[9] = {0,
                   1,
                   1,
                   (7. - 3.5 * sqrt(3)),
                   1,
                   1,
                   sqrt(3) * 4. / (6 * sqrt(2) + sqrt(3)),
                   sqrt(3) * 4. / (6 * sqrt(2) + sqrt(3)),
                   -3. / 11 + 6 * sqrt(3) / 11};
    return c[type] / scale;
}

static double calculate_lattice_constant(int type,
                     double interatomic_distance) {
    assert(type >= 1 && type <= 8);
    double c[9] = {0,
                   2 / sqrt(2),
                   2 / sqrt(2),
                   2 / sqrt(3),
                   2 / sqrt(2),
                   1,
                   4 / sqrt(3),
                   4 / sqrt(3),
                   sqrt(3)};
    return c[type] * interatomic_distance;
}

static void output_data(ptm::result_t *res, ptm_atomicenv_t* env,
                        bool calculate_deformation, ptm_result_t* result) {
    const ptm::refdata_t *ref = res->ref_struct;
    if (ref == NULL)
        return;

    result->structure_type = ref->type;
    result->ordering_type = ptm::find_alloy_type(ref, res->mapping, env->numbers);

    double qidentity[4] = {1, 0, 0, 0};
    result->template_index = ptm_remap_template(ref->type, 0,
                                                     qidentity, res->q, res->mapping);
    if (calculate_deformation && result->template_index >= 0) {

        int num_points = ref->num_nbrs + 1;
        double scaled_points[PTM_MAX_INPUT_POINTS][3];
        ptm::subtract_barycentre(num_points, env->points, scaled_points);
        for (int i = 0; i < num_points; i++) {
            scaled_points[i][0] *= res->scale;
            scaled_points[i][1] *= res->scale;
            scaled_points[i][2] *= res->scale;
        }

        ptm::calculate_deformation_gradient(num_points, res->mapping, scaled_points, ref->penrose[result->template_index],
                                            result->F);
        if (ref->type == PTM_MATCH_GRAPHENE) // hack for pseudo-2d structures
            result->F[8] = 1;
    }

    result->interatomic_distance = calculate_interatomic_distance(ref->type, res->scale);
    result->lattice_constant = calculate_lattice_constant(ref->type, result->interatomic_distance);
    result->rmsd = res->rmsd;
    result->scale = res->scale;
    memcpy(result->orientation, res->q, 4 * sizeof(double));
}

static void output_environment(ptm::result_t *res, ptm_atomicenv_t* env,
                               ptm_atomicenv_t* output_env) {
    if (output_env == NULL)
        return;

    const ptm::refdata_t *ref = res->ref_struct;
    if (ref == NULL)
        return;

    int num_nbrs = ref == NULL ? env->num - 1 : ref->num_nbrs;
    output_env->num = num_nbrs + 1;

    for (int i = 0; i < output_env->num; i++) {
        output_env->correspondences[i] = env->correspondences[res->mapping[i]];
        output_env->atom_indices[i] = env->atom_indices[res->mapping[i]];
        memcpy(output_env->points[i], env->points[res->mapping[i]], 3 * sizeof(double));
    }
}

extern bool ptm_initialized;

int ptm_index(ptm_local_handle_t local_handle,
                size_t atom_index, int (get_neighbours)(void* vdata, size_t _unused_lammps_variable, size_t atom_index, int num, ptm_atomicenv_t* env), void* nbrlist,
                int32_t flags, bool calculate_deformation,
                ptm_result_t* result, ptm_atomicenv_t* output_env)
{
    int ret = 0;
    assert(ptm_initialized);
    if (!ptm_initialized)    //assert is not active in OVITO release build
        return -1;

    //-------- initialize output values with failure case --------
    ptm::result_t res;
    res.ref_struct = NULL;
    res.rmsd = INFINITY;
    for (int i=0;i<PTM_MAX_POINTS;i++)
        res.mapping[i] = i;

    memset(result, 0, sizeof(ptm_result_t));
    //------------------------------------------------------------

    ptm::convexhull_t ch;
    double ch_points[PTM_MAX_INPUT_POINTS][3];
    ptm_atomicenv_t env, dmn_env, grp_env;
    env.num = dmn_env.num = grp_env.num = 0;

    if (flags & (PTM_CHECK_SC | PTM_CHECK_FCC | PTM_CHECK_HCP | PTM_CHECK_ICO | PTM_CHECK_BCC)) {

        const int num_inner = 1, num_outer = 0;
        ret = ptm::calculate_two_shell_neighbour_ordering(num_inner, num_outer, atom_index, get_neighbours, nbrlist, NULL, &env);
        if (ret == 0) {
            int num_points = env.num;
            ptm::normalize_vertices(num_points, env.points, ch_points);
            ch.ok = false;

            if ((flags & PTM_CHECK_SC) && num_points >= PTM_NUM_POINTS_SC)
                ret = match_general(&ptm::structure_sc, ch_points, env.points, &ch, &res);

            if ((flags & (PTM_CHECK_FCC | PTM_CHECK_HCP | PTM_CHECK_ICO)) && num_points >= PTM_NUM_POINTS_FCC)
                ret = match_fcc_hcp_ico(ch_points, env.points, flags, &ch, &res);

            if ((flags & PTM_CHECK_BCC) && num_points >= PTM_NUM_POINTS_BCC)
                ret = match_general(&ptm::structure_bcc, ch_points, env.points, &ch, &res);
        }
    }

    if (flags & (PTM_CHECK_DCUB | PTM_CHECK_DHEX)) {

        const int num_inner = 4, num_outer = 3;
        ret = ptm::calculate_two_shell_neighbour_ordering(num_inner, num_outer, atom_index, get_neighbours, nbrlist, &env, &dmn_env);
        if (ret == 0) {
            ptm::normalize_vertices(PTM_NUM_NBRS_DCUB + 1, dmn_env.points, ch_points);
            ch.ok = false;

            ret = match_dcub_dhex(ch_points, dmn_env.points, flags, &ch, &res);
        }
    }

    if (flags & PTM_CHECK_GRAPHENE) {

        const int num_inner = 3, num_outer = 2;
        ret = ptm::calculate_two_shell_neighbour_ordering(num_inner, num_outer, atom_index, get_neighbours, nbrlist, &env, &grp_env);
        if (ret == 0) {
            ret = match_graphene(grp_env.points, &res);
        }
    }

    if (res.ref_struct == NULL)
        return PTM_NO_ERROR;

    ptm_atomicenv_t* res_env = &env;
    if (res.ref_struct->type == PTM_MATCH_DCUB || res.ref_struct->type == PTM_MATCH_DHEX)
        res_env = &dmn_env;
    else if (res.ref_struct->type == PTM_MATCH_GRAPHENE)
        res_env = &grp_env;

    output_data(&res, res_env, calculate_deformation, result);
    output_environment(&res, res_env, output_env);
    return PTM_NO_ERROR;
}

