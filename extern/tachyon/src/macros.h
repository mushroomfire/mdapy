/* 
 * macros.h - This file contains macro versions of functions that would be best 
 *            used as inlined code rather than function calls.
 *
 * (C) Copyright 1994-2022 John E. Stone
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * $Id: macros.h,v 1.7 2022/02/18 17:55:28 johns Exp $
 *
 */

#define MYMAX(a , b) ((a) > (b) ? (a) : (b))
#define MYMIN(a , b) ((a) < (b) ? (a) : (b))

#define VDOT(return, a, b) 				\
 return=(a.x * b.x  +  a.y * b.y  +  a.z * b.z); 	\

#define RAYPNT(c, a, b)		\
c.x = a.o.x + ( a.d.x * b );	\
c.y = a.o.y + ( a.d.y * b );	\
c.z = a.o.z + ( a.d.z * b );	\


#define VSUB(a, b, c)		\
c.x = (a.x - b.x);		\
c.y = (a.y - b.y);		\
c.z = (a.z - b.z);		\


#define VCROSS(a, b, c) 				\
 c->x = (a->y * b->z) - (a->z * b->y);			\
 c->y = (a->z * b->x) - (a->x * b->z);			\
 c->z = (a->x * b->y) - (a->y * b->x);			\

