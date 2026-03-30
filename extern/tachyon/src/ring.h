/* 
 * ring.h - This file contains the defines for rings etc.
 *
 * (C) Copyright 1994-2022 John E. Stone
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * $Id: ring.h,v 1.12 2022/02/18 17:55:28 johns Exp $
 *
 */

object * newring(void * tex, vector ctr, vector norm, flt in, flt out);

#ifdef RING_PRIVATE 
typedef struct {
  RT_OBJECT_HEAD
  vector ctr;       /**< center of ring */
  vector norm;      /**< surface normal */
  flt inrad;        /**< inner ring radius (0.0 for disk) */
  flt outrad;       /**< outer ring raidus */
} ring; 

static int ring_bbox(void * obj, vector * min, vector * max);
static void ring_intersect(const ring *, ray *);
static void ring_normal(const ring *, const vector *, const ray * incident, vector *);
#endif

