/* 
 * box.h - This file contains the defines for boxes etc.
 *
 * (C) Copyright 1994-2022 John E. Stone
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * $Id: box.h,v 1.10 2022/02/18 17:55:28 johns Exp $
 *
 */

/**
 * axis-aligned box definition
 */ 
typedef struct {
  RT_OBJECT_HEAD
  vector min;     /**< minimum vertex coordinate */
  vector max;     /**< maximum vertex coordinate */
} box; 


box * newbox(void * tex, vector min, vector max);
void box_intersect(const box *, ray *);
void box_normal(const box *, const vector *, const ray *, vector *);
