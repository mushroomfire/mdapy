/* 
 * quadric.h - This file contains the defines for quadrics.
 *
 * (C) Copyright 1994-2022 John E. Stone
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * $Id: quadric.h,v 1.13 2022/02/18 17:55:28 johns Exp $
 *
 */

typedef struct {
  flt a; flt b; flt c;
  flt d; flt e; flt f;
  flt g; flt h; flt i; flt j;
} quadmatrix;

 
typedef struct {
  RT_OBJECT_HEAD
  vector ctr;      /**< center of quadric object            */
  quadmatrix mat;  /**< quadric function coefficient matrix */
} quadric; 


quadric * newquadric(void);
void quadric_intersect(const quadric *, ray *);
void quadric_normal(const quadric *, const vector *, const ray *, vector *);



