/*Copyright (c) 2022 PM Larsen

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include "ptm_constants.h"
#include "ptm_initialize_data.h"
#include "ptm_quat.h"
#include <cassert>
#include <cstring>


static int rotate_into_fundamental_zone(int type, double *q) {
    if (type == PTM_MATCH_SC
        || type == PTM_MATCH_FCC
        || type == PTM_MATCH_BCC
        || type == PTM_MATCH_DCUB)
        return ptm::rotate_quaternion_into_cubic_fundamental_zone(q);

    if (type == PTM_MATCH_HCP
        || type == PTM_MATCH_GRAPHENE
        || type == PTM_MATCH_DHEX) {
        return ptm::rotate_quaternion_into_hcp_conventional_fundamental_zone(q);
    }

    if (type == PTM_MATCH_ICO)
        return ptm::rotate_quaternion_into_icosahedral_fundamental_zone(q);

    return -1;
}

static int map_quaternion(int type, double *q, int i)
{
    if (type == PTM_MATCH_SC
        || type == PTM_MATCH_FCC
        || type == PTM_MATCH_BCC
        || type == PTM_MATCH_DCUB) {
        return ptm::map_quaternion_cubic(q, i);
    }

    if (type == PTM_MATCH_HCP
        || type == PTM_MATCH_GRAPHENE
        || type == PTM_MATCH_DHEX) {
            return ptm::map_quaternion_hcp_conventional(q, i);
    }

    if (type == PTM_MATCH_ICO)
        return ptm::map_quaternion_icosahedral(q, i);

    return -1;
}

// Map a quaternion into a crystallographically equivalent orientation which brings
// it as close as possible to a target orientation.
//
// Returns: index of symmetry group which minimizes distance between q and qtarget
static int map_quaternion_onto_target(int type,         // input
                                      double* qtarget,  // input
                                      double* q)        // input/output
{
    //argmin_g ||ag - b|| = argmin_g  ||binv.a.g - binv.b|| = argmin_g  ||binv.a.g - I||
    double qtemp[4];
    double invtarget[4] = {-qtarget[0], qtarget[1], qtarget[2], qtarget[3]};

    ptm::quat_rot(invtarget, q, qtemp);
    int bi = rotate_into_fundamental_zone(type, qtemp);
    if (bi < 0)
        return bi;

    map_quaternion(type, q, bi);
    return bi;
}

double ptm_map_and_calculate_disorientation(int type, double* qtarget, double* q)
{
    if (type == PTM_MATCH_NONE) {
        return INFINITY;
    }

    int bi = map_quaternion_onto_target(type, qtarget, q);
    if (bi < 0) {
        return INFINITY;
    }

    return ptm::quat_misorientation(q, qtarget);
}

static void permute_mapping(int num_points, const int8_t* permutation, int8_t* mapping)
{
    int8_t temp[PTM_MAX_POINTS];
    memset(temp, -1, PTM_MAX_POINTS * sizeof(int8_t));

    for (int i=0;i<num_points;i++)
        temp[permutation[i]] = mapping[i];

    memcpy(mapping, temp, num_points * sizeof(int8_t));
}

// Use the first template of the structure. Adjust the orientation (q) and mapping
// accordingly. For single-template structures (SC, FCC, BCC, ICO) this function
// has no effect.
//
// Returns: error code
static int undo_conventional_orientation(int type,                  // input
                                         int input_template_index,  // input
                                         double* q,                 // input/output
                                         int8_t* mapping)           // input/output
{
    if (input_template_index == 0)
        return 0;

    const ptm::refdata_t* ref = ptm::refdata[type];

    //this is an input error
    if (ref->template_indices == NULL)
        return -1;

    int mapping_index = -1;
    for (int i=0;i<ref->num_conventional_mappings;i++)
    {
        if (ref->template_indices[i] == input_template_index)
        {
            mapping_index = i;
            break;
        }
    }

    assert(mapping_index != -1);

    double qtemp[4];
    ptm::quat_rot(q, (double*)ref->qconventional[mapping_index], qtemp);
    memcpy(q, qtemp, 4 * sizeof(double));

    permute_mapping(ref->num_nbrs + 1, ref->mapping_conventional_inverse[mapping_index], mapping);
    return 0;
}

// Re-map a template towards a target orientation. If the misorientation between q and qtarget
// is small (less than the maximum crystallographic misorientation) nothing is changed. Otherwise
// the orientation (q) is moved to the (crystallographically equivalent) orientation which brings
// it closest to the target orientation (qtarget). The mapping (between template and simulated
// atoms) is updated accordingly.
//
// Returns: the template index of the re-mapped structure.
// If the structure type has multiple templates (HCP, DCUB, DHEX, graphene), the template index
// can change during re-mapping. For single-template structures (SC, FCC, BCC, ICO) the returned
// template index is the same as the input template index.
int ptm_remap_template(int type,                    // input
                       int input_template_index,    // input
                       double* qtarget,             // input
                       double* q,                   // input/output
                       int8_t* mapping)             // input/output
{
    if (type == PTM_MATCH_NONE)
        return -1;

    int ret = undo_conventional_orientation(type, input_template_index, q, mapping);
    if (ret != 0)
        return -1;

    int bi = map_quaternion_onto_target(type, qtarget, q);
    if (bi < 0)
        return -1;

    const ptm::refdata_t* ref = ptm::refdata[type];
    permute_mapping(ref->num_nbrs + 1, ref->mapping_conventional[bi], mapping);
    return ref->template_indices[bi];
}

