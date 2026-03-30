/*
 * apitrigeom.h - header for functions to generate triangle tesselated 
 *                geometry for use with OpenGL, XGL, etc.
 *
 * (C) Copyright 1994-2022 John E. Stone
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * $Id: apitrigeom.h,v 1.6 2022/02/18 17:55:28 johns Exp $
 *
 */

void rt_tri_fcylinder(SceneHandle, void *, apivector, apivector, flt);
void rt_tri_cylinder(SceneHandle, void *, apivector, apivector, flt);
void rt_tri_ring(SceneHandle, void *, apivector, apivector, flt, flt);
void rt_tri_plane(SceneHandle, void *, apivector, apivector);
void rt_tri_box(SceneHandle, void *, apivector, apivector);

