/*
 * vol.h - Volume rendering definitions etc.
 *
 * (C) Copyright 1994-2022 John E. Stone
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * $Id: vol.h,v 1.8 2022/02/18 17:55:28 johns Exp $
 *
 */

void * newscalarvol(void * intex, vector min, vector max, 
                    int xs, int ys, int zs, 
                    const char * fname, scalarvol * invol);

void  LoadVol(scalarvol *);
color scalar_volume_texture(const vector *, const texture *, ray *);

