// Voro++, a cell-based Voronoi library
// By Chris H. Rycroft and the Rycroft Group

/** \file worklist_2d.hh
 * \brief Header file for setting constants used in the block worklists that are
 * used during cell computation.
 *
 * This file is automatically generated by worklist_gen.pl and it is not
 * intended to be edited by hand. */

#ifndef VOROPP_WORKLIST_2D_HH
#define VOROPP_WORKLIST_2D_HH

namespace voro {

/** Each region is divided into a grid of subregions, and a worklist is
 * constructed for each. This parameter sets is set to half the number of
 * subregions that the block is divided into. */
const int wl_hgrid_2d=4;
/** The number of subregions that a block is subdivided into, which is twice
the value of hgrid. */
const int wl_fgrid_2d=8;
/** The total number of worklists, set to the cube of hgrid. */
const int wl_hgridsq_2d=16;
/** The number of elements in each worklist. */
const int wl_seq_length_2d=64;

}
#endif