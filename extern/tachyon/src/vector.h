/*
 * vector.h - This file contains declarations of vector functions
 *
 * (C) Copyright 1994-2022 John E. Stone
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * $Id: vector.h,v 1.6 2022/02/18 17:55:28 johns Exp $
 *
 */

flt VDot(const vector *, const vector *);
void VCross(const vector *, const vector *, vector *);
flt VLength(const vector *);
void VNorm(vector *);
void VAdd(const vector *, const vector *, vector *);
void VSub(const vector *, const vector *, vector *);
void VAddS(flt, const vector *, const vector *, vector *);
vector Raypnt(const ray *, flt);
void VScale(vector * a, flt s); 

void ColorAddS(color * a, const color * b, flt s); 
void ColorAccum(color * a, const color * b); 
void ColorScale(color * a, flt s); 

