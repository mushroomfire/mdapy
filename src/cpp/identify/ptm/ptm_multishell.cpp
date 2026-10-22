/*Copyright (c) 2022 PM Larsen

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

//todo: normalize vertices

#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <unordered_set>
#include <cstdint>
#include "ptm_constants.h"
#include "ptm_voronoi_cell.h"
#include "ptm_multishell.h"
#include "ptm_normalize_vertices.h"


namespace ptm {

typedef struct
{
    int rank;
    int inner;
    int correspondences;
    size_t atom_index;
    int32_t number;
    double delta[3];
} atomorder_t;

static bool atomorder_compare(atomorder_t const& a, atomorder_t const& b)
{
    return a.rank < b.rank;
}

static void filter_neighbours(ptm_atomicenv_t* env)
{
    ptm_atomicenv_t temp;
    temp.num = 0;

    for (int i=0;i<env->num;i++) {
        if (env->correspondences[i] <= MAX_MULTISHELL_NEIGHBOURS) {

            temp.correspondences[temp.num] = env->correspondences[i];
            temp.atom_indices[temp.num] = env->atom_indices[i];
            temp.numbers[temp.num] = env->numbers[i];
            memcpy(temp.points[temp.num], env->points[i], 3 * sizeof(double));
            temp.num++;
        }
    }

    memcpy(env, &temp, sizeof(ptm_atomicenv_t));
}

static double distance(double* a, double* b) {
    double dx = a[0] - b[0];
    double dy = a[1] - b[1];
    double dz = a[2] - b[2];
    return sqrt(dx * dx + dy * dy + dz * dz);
}

static bool already_claimed(ptm_atomicenv_t* output, int num_inner, int num_outer, int* counts, size_t nbr_atom_index, double* delta, double tolerance)
{
    for (int i=0;i<num_inner+1;i++) {
        if (nbr_atom_index == output->atom_indices[i]) {
            double d = distance(delta, output->points[i]);
            if (d < tolerance)
                return true;
        }
    }

    for (int i=0;i<num_inner;i++) {
        for (int j=0;j<counts[i];j++) {
            size_t index = 1 + num_inner + num_outer * i + j;
            if (nbr_atom_index == output->atom_indices[index]) {
                double d = distance(delta, output->points[index]);
                if (d < tolerance)
                    return true;
            }
        }
    }

    return false;
}

#define MAX_INNER 4

int calculate_two_shell_neighbour_ordering( int num_inner, int num_outer,
                        size_t atom_index, int (get_neighbours)(void* vdata, size_t _unused_lammps_variable, size_t atom_index, int num, ptm_atomicenv_t* env), void* nbrlist,
                        ptm_atomicenv_t* central_env, ptm_atomicenv_t* output)
{
    assert(num_inner <= MAX_INNER);

    if (num_outer == 0) {
        get_neighbours(nbrlist, -1, atom_index, PTM_MAX_INPUT_POINTS, output);
        return 0;
    }

    ptm_atomicenv_t env;
    if (central_env != NULL && central_env->num >= num_inner) {
        memcpy(&env, central_env, sizeof(ptm_atomicenv_t));
    }
    else {
        get_neighbours(nbrlist, -1, atom_index, PTM_MAX_INPUT_POINTS, &env);
    }
    filter_neighbours(&env);
    if (env.num < num_inner + 1)
        return -1;

    for (int i=0;i<num_inner+1;i++)
    {
        output->correspondences[i] = env.correspondences[i];
        output->atom_indices[i] = env.atom_indices[i];
        output->numbers[i] = env.numbers[i];
        memcpy(output->points[i], env.points[i], 3 * sizeof(double));
    }

    double tolerance = 1E-5 * distance(output->points[0], output->points[1]);
    tolerance = std::max(tolerance, 1E-5);

    int num_inserted = 0;
    atomorder_t data[MAX_INNER * PTM_MAX_INPUT_POINTS];
    for (int i=0;i<num_inner;i++)
    {
        get_neighbours(nbrlist, -1, output->atom_indices[1 + i], PTM_MAX_INPUT_POINTS, &env);
        filter_neighbours(&env);
        if (env.num < num_inner + 1)
            return -1;

        for (int j=1;j<env.num;j++)
        {
            data[num_inserted].inner = i;
            data[num_inserted].rank = j;
            data[num_inserted].correspondences = env.correspondences[j];
            data[num_inserted].atom_index = env.atom_indices[j];
            data[num_inserted].number = env.numbers[j];

            memcpy(data[num_inserted].delta, env.points[j], 3 * sizeof(double));
            for (int k=0;k<3;k++)
                data[num_inserted].delta[k] += output->points[1 + i][k];

            num_inserted++;
        }
    }

    std::stable_sort(data, data + num_inserted, &atomorder_compare);

    int num_found = 0;
    int counts[MAX_INNER] = {0};
    for (int i=0;i<num_inserted;i++)
    {
        int inner = data[i].inner;
        if (counts[inner] >= num_outer)
            continue;

        if (already_claimed(output, num_inner, num_outer, counts, data[i].atom_index, data[i].delta, tolerance))
            continue;

        size_t index = 1 + num_inner + num_outer * inner + counts[inner];
        output->correspondences[index] = data[i].correspondences;
        output->atom_indices[index] = data[i].atom_index;
        output->numbers[index] = data[i].number;
        memcpy(output->points[index], &data[i].delta, 3 * sizeof(double));

        counts[inner]++;
        num_found++;
        if (num_found >= num_inner * num_outer)
            break;
    }

    if (num_found != num_inner * num_outer)
        return -1;

    return 0;
}

}

