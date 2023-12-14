/*Copyright (c) 2016 PM Larsen

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef PTM_MULTISHELL_H
#define PTM_MULTISHELL_H

#include <cstddef>

namespace ptm {

typedef struct
{
	int ordering[PTM_MAX_INPUT_POINTS];
	size_t nbr_indices[PTM_MAX_INPUT_POINTS];
	int32_t numbers[PTM_MAX_INPUT_POINTS];
	double points[PTM_MAX_INPUT_POINTS][3];
} atomicenv_t;


int calculate_two_shell_neighbour_ordering(	int num_inner, int num_outer,
						size_t atom_index, int (get_neighbours)(void* vdata, size_t _unused_lammps_variable, size_t atom_index, int num, int* ordering, size_t* nbr_indices, int32_t* numbers, double (*nbr_pos)[3]), void* nbrlist,
						ptm::atomicenv_t* output);
}

#endif

