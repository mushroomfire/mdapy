/*Copyright (c) 2016 PM Larsen

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef PTM_FUNCTIONS_H
#define PTM_FUNCTIONS_H

#include <stdint.h>
#include <stdbool.h>
#include "ptm_initialize_data.h"
#include "ptm_constants.h"


//------------------------------------
//    function declarations
//------------------------------------
#ifdef __cplusplus
extern "C" {
#endif


int ptm_index(	ptm_local_handle_t local_handle,
		size_t atom_index, int (get_neighbours)(void* vdata, size_t _unused_lammps_variable, size_t atom_index, int num, int* ordering, size_t* nbr_indices, int32_t* numbers, double (*nbr_pos)[3]), void* nbrlist,
		int32_t flags, bool output_conventional_orientation, //inputs
		int32_t* p_type, int32_t* p_alloy_type, double* p_scale, double* p_rmsd, double* q, double* F, double* F_res, double* U, double* P, double* p_interatomic_distance, double* p_lattice_constant,
		int* p_best_template_index, const double (**p_best_template)[3], int8_t* output_indices);	//outputs


int ptm_remap_template(	int type, bool output_conventional_orientation, int input_template_index, double* qtarget, double* q,
			double* p_disorientation, int8_t* mapping, const double (**p_best_template)[3]);

int ptm_undo_conventional_orientation(int type, int input_template_index, double* q, int8_t* mapping);

int ptm_preorder_neighbours(void* _voronoi_handle, int num_input_points, double (*input_points)[3], uint64_t* res);
void ptm_index_to_permutation(int n, uint64_t k, int* permuted);


#ifdef __cplusplus
}
#endif

#endif

