/* 
 * cylinder.h - This file contains the defines for cylinders etc.
 *
 * (C) Copyright 1994-2022 John E. Stone
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * $Id: cylinder.h,v 1.12 2022/02/18 17:55:28 johns Exp $
 *
 */

object * newcylinder(void *, vector, vector, flt);
object * newfcylinder(void *, vector, vector, flt);

#ifdef CYLINDER_PRIVATE

/**
 * Types for cylinder objects
 */
typedef struct {
  RT_OBJECT_HEAD
  vector ctr;       /**< starting endpoint of cylinder */
  vector axis;      /**< cylinder axis                 */
  flt rad;
} cylinder;

static void cylinder_intersect(const cylinder *, ray *);
static void fcylinder_intersect(const cylinder *, ray *);

static int cylinder_bbox(void * obj, vector * min, vector * max);
static int fcylinder_bbox(void * obj, vector * min, vector * max);

static void cylinder_normal(const cylinder *, const vector *, const ray *, vector *);
#endif

