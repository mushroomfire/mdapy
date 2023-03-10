/*Copyright (c) 2016 PM Larsen

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
	int ordering;
	size_t atom_index;
	int32_t number;
	double offset[3];
} sorthelper_t;

static bool sorthelper_compare(sorthelper_t const& a, sorthelper_t const& b)
{
	return a.rank < b.rank;
}

#define MAX_INNER 4

int calculate_two_shell_neighbour_ordering(	int num_inner, int num_outer,
						size_t atom_index, int (get_neighbours)(void* vdata, size_t _unused_lammps_variable, size_t atom_index, int num, int* ordering, size_t* nbr_indices, int32_t* numbers, double (*nbr_pos)[3]), void* nbrlist,
						ptm::atomicenv_t* output)
{
	assert(num_inner <= MAX_INNER);

	ptm::atomicenv_t central;
	int num_input_points = get_neighbours(nbrlist, -1, atom_index, PTM_MAX_INPUT_POINTS, central.ordering, central.nbr_indices, central.numbers, central.points);
	if (num_input_points < num_inner + 1)
		return -1;

	std::unordered_set<size_t> claimed;
	for (int i=0;i<num_inner+1;i++)
	{
		output->ordering[i] = central.ordering[i];
		output->nbr_indices[i] = central.nbr_indices[i];
		output->numbers[i] = central.numbers[i];
		memcpy(output->points[i], central.points[i], 3 * sizeof(double));

		claimed.insert(central.nbr_indices[i]);
	}

	int num_inserted = 0;
	sorthelper_t data[MAX_INNER * PTM_MAX_INPUT_POINTS];
	for (int i=0;i<num_inner;i++)
	{
		ptm::atomicenv_t inner;
		int num_points = get_neighbours(nbrlist, -1, central.nbr_indices[1 + i], PTM_MAX_INPUT_POINTS, inner.ordering, inner.nbr_indices, inner.numbers, inner.points);
		if (num_points < num_inner + 1)
			return -1;

		for (int j=0;j<num_points;j++)
		{
			size_t key = inner.nbr_indices[j];

			bool already_claimed = claimed.find(key) != claimed.end();
			if (already_claimed)
				continue;

			data[num_inserted].inner = i;
			data[num_inserted].rank = j;
			data[num_inserted].ordering = inner.ordering[j];
			data[num_inserted].atom_index = inner.nbr_indices[j];
			data[num_inserted].number = inner.numbers[j];

			memcpy(data[num_inserted].offset, inner.points[j], 3 * sizeof(double));
			for (int k=0;k<3;k++)
				data[num_inserted].offset[k] += central.points[1 + i][k];

			num_inserted++;
		}
	}

	std::sort(data, data + num_inserted, &sorthelper_compare);

	int num_found = 0;
	int counts[MAX_INNER] = {0};
	for (int i=0;i<num_inserted;i++)
	{
		int inner = data[i].inner;
		int nbr_atom_index = data[i].atom_index;

		bool already_claimed = claimed.find(nbr_atom_index) != claimed.end();
		if (counts[inner] >= num_outer || already_claimed)
			continue;

		output->ordering[1 + num_inner + num_outer * inner + counts[inner]] = data[i].ordering;

		output->nbr_indices[1 + num_inner + num_outer * inner + counts[inner]] = nbr_atom_index;
		output->numbers[1 + num_inner + num_outer * inner + counts[inner]] = data[i].number;
		memcpy(output->points[1 + num_inner + num_outer * inner + counts[inner]], &data[i].offset, 3 * sizeof(double));
		claimed.insert(nbr_atom_index);

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

