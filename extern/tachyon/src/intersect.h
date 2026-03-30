/* 
 * intersect.h - This file contains the declarations and defines for the
 *               functions that manage intersection, bounding and CSG..
 *
 * (C) Copyright 1994-2022 John E. Stone
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * $Id: intersect.h,v 1.19 2022/02/18 17:55:28 johns Exp $
 *
 */

unsigned int new_objectid(scenedef *);
void free_objects(object *);
void intersect_objects(ray *);

void add_clipped_intersection(flt, const object *, ray *);
void add_regular_intersection(flt, const object *, ray *);
int closest_intersection(flt *, object const **, ray *);

void add_shadow_intersection(flt, const object *, ray *);
void add_clipped_shadow_intersection(flt, const object *, ray *);
int shadow_intersection(ray *);

#define reset_intersection(ry) \
	(ry)->intstruct.num = 0; \
	(ry)->intstruct.shadowfilter = 1.0;

