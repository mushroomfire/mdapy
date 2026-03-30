/*
 * coordsys.h - defines for coordinate system routines.
 *  
 * (C) Copyright 1994-2022 John E. Stone
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * $Id: coordsys.h,v 1.4 2022/02/18 17:55:28 johns Exp $
 *
 */

void xytopolar(flt, flt, flt, flt *, flt *);
void xyztocyl(vector, flt, flt *, flt *);
void xyztospr(vector, flt *, flt *);
