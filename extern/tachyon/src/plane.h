/* 
 * plane.h - This file contains the defines for planes etc.
 *
 * (C) Copyright 1994-2022 John E. Stone
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * $Id: plane.h,v 1.14 2022/02/18 17:55:28 johns Exp $
 *
 */
 
object * newplane(void * tex, vector ctr, vector norm);

#ifdef PLANE_PRIVATE
typedef struct {
  RT_OBJECT_HEAD
  flt d;            /**< plane distance along normal */
  vector norm;      /**< surface normal              */
} plane; 

static void plane_intersect(const plane *, ray *);
static int plane_bbox(void * obj, vector * min, vector * max);
static void plane_normal(const plane *, const vector *, const ray * incident, vector *);
#endif

