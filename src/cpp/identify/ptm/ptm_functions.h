/*Copyright (c) 2022 PM Larsen

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
int ptm_index(  ptm_local_handle_t local_handle,
                size_t atom_index, int (get_neighbours)(void* vdata, size_t _unused_lammps_variable, size_t atom_index, int num, ptm_atomicenv_t* env), void* nbrlist,
                int32_t flags, bool calculate_deformation, //inputs
                ptm_result_t* result, ptm_atomicenv_t* output_env); //outputs

int ptm_remap_template(int type, int input_template_index, double* qtarget, double* q, int8_t* mapping);

double ptm_map_and_calculate_disorientation(int type, double* qtarget, double* q);

int ptm_preorder_neighbours(void* _voronoi_handle, int num_input_points, double (*input_points)[3], uint64_t* res);

uint64_t ptm_encode_correspondences(int type, int num, int8_t* correspondences, int best_template_index);
void ptm_decode_correspondences(int type, uint64_t encoded, int8_t* correspondences, int* p_best_template_index);

#endif

