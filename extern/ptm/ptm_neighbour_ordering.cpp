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
#include "ptm_neighbour_ordering.h"
#include "ptm_normalize_vertices.h"
#include "ptm_solid_angles.h"
#include "ptm_correspondences.h"


namespace ptm {

typedef struct
{
    double area;
    double dist;
    int ordering;
} sorthelper_t;

static bool sorthelper_compare(sorthelper_t const& a, sorthelper_t const& b)
{
    if (a.area > b.area)
        return true;

    if (a.area < b.area)
        return false;

    if (a.dist < b.dist)
        return true;

    return false;
}

static double norm_squared(double* x)
{
    return x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
}

//todo: change voronoi code to return errors rather than exiting
static int calculate_voronoi_face_areas(int num_points, const double (*_points)[3], double* normsq, double max_norm, ptm_voro::voronoicell_neighbor* v, std::vector<int>& nbr_indices, std::vector<double>& face_areas)
{
    const double k = 10 * max_norm;
    v->init(-k, k, -k, k, -k, k);

    for (int i=0;i<num_points;i++)
    {
        double x = _points[i][0];
        double y = _points[i][1];
        double z = _points[i][2];
        v->nplane(x, y, z, normsq[i], i);
    }

    v->neighbors(nbr_indices);

    std::vector<int> face_vertices;
    std::vector<double> vertices;

    v->face_vertices(face_vertices);
    v->vertices(0, 0, 0, vertices);

    size_t num_vertices = vertices.size() / 3;
    for (size_t i=0;i<num_vertices;i++)
    {
        double norm = sqrt(norm_squared(&vertices[i * 3]));
        vertices[i * 3 + 0] /= norm;
        vertices[i * 3 + 1] /= norm;
        vertices[i * 3 + 2] /= norm;
    }

    int num_faces = v->number_of_faces();

    size_t c = 0;
    for (int current_face=0;current_face<num_faces;current_face++)
    {
        int num = face_vertices[c++];

        int point_index = nbr_indices[current_face];
        if (point_index >= 0)
        {
            double solid_angle = 0;
            int u = face_vertices[c];
            int v = face_vertices[c + 1];
            for (int i=2;i<num;i++)
            {
                int w = face_vertices[c + i];
                double omega = calculate_solid_angle(&vertices[u * 3], &vertices[v * 3], &vertices[w * 3]);
                solid_angle += omega;

                v = w;
            }

            face_areas[current_face] = solid_angle;
        }

        c += num;
    }

    assert(c == face_vertices.size());
    return 0;
}

static int calculate_neighbour_ordering(void* _voronoi_handle, int num, double (*_points)[3], sorthelper_t* data)
{
    assert(num <= PTM_MAX_INPUT_POINTS);

    ptm_voro::voronoicell_neighbor* voronoi_handle = (ptm_voro::voronoicell_neighbor*)_voronoi_handle;

    double max_norm = 0;
    double points[PTM_MAX_INPUT_POINTS][3];
    double normsq[PTM_MAX_INPUT_POINTS];
    memcpy(points, _points, 3 * num * sizeof(double));

    for (int i=0;i<num;i++)
    {
        normsq[i] = norm_squared(points[i]);
        max_norm = std::max(max_norm, normsq[i]);
    }
    max_norm = sqrt(max_norm);

    std::vector<int> nbr_indices(num + 6);
    std::vector<double> face_areas(num + 6);
    int ret = calculate_voronoi_face_areas(num, points, normsq, max_norm, voronoi_handle, nbr_indices, face_areas);
    if (ret != 0)
        return ret;

    double areas[PTM_MAX_INPUT_POINTS] = {0};
    for (size_t i=0;i<nbr_indices.size();i++)
    {
        int index = nbr_indices[i];
        if (index >= 0)
            areas[index] = face_areas[i];
    }

    for (int i=0;i<num;i++)
    {
        assert(areas[i] == areas[i]);
        data[i].area = areas[i];
        data[i].dist = normsq[i];
        data[i].ordering = i;
    }

    std::stable_sort(data, data + num, &sorthelper_compare);
    return ret;
}

void* voronoi_initialize_local()
{
    ptm_voro::voronoicell_neighbor* ptr = new ptm_voro::voronoicell_neighbor;
    return (void*)ptr;
}

void voronoi_uninitialize_local(void* _ptr)
{
    ptm_voro::voronoicell_neighbor* ptr = (ptm_voro::voronoicell_neighbor*)_ptr;
    delete ptr;
}

int preorder_neighbours(void* _voronoi_handle, int num_input_points, double (*input_points)[3], uint64_t* res)
{
    ptm::sorthelper_t data[PTM_MAX_INPUT_POINTS - 1];
    int num = std::min(PTM_MAX_INPUT_POINTS - 1, num_input_points);

    int ret = ptm::calculate_neighbour_ordering(_voronoi_handle, num, input_points, data);
    if (ret != 0) {
        *res = 0;
        return ret;
    }

    int8_t correspondences[PTM_MAX_INPUT_POINTS];
    correspondences[0] = 0;
    for (int i=0;i<num;i++)
        correspondences[i + 1] = data[i].ordering + 1;

    complete_correspondences(num + 1, correspondences);
    *res = encode_correspondences(PTM_MATCH_NONE, num,    //this gives us default behaviour
                                  correspondences, 0);
    return PTM_NO_ERROR;
}

}

int ptm_preorder_neighbours(void* _voronoi_handle, int num_input_points, double (*input_points)[3], uint64_t* res)
{
    return ptm::preorder_neighbours(_voronoi_handle, num_input_points, input_points, res);
}

