/* 
 * sphere.h - This file contains the defines for spheres etc.
 *
 * (C) Copyright 1994-2022 John E. Stone
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * $Id: sphere.h,v 1.15 2022/02/18 17:55:28 johns Exp $
 *
 */

object * newsphere(void *, vector, flt);

#ifdef SPHERE_PRIVATE

typedef struct {
  RT_OBJECT_HEAD
  vector ctr;    /**< sphere center */
  flt rad;       /**< sphere radius */
} sphere;

static int sphere_bbox(void * obj, vector * min, vector * max);
static void sphere_intersect(const sphere *, ray *);
static void sphere_normal(const sphere *, const vector *, const ray *, vector *);

#endif /* SPHERE_PRIVATE */

