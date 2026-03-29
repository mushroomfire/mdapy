/*
 * apitrigeom.h - header for functions to generate triangle tesselated 
 *                geometry for use with OpenGL, XGL, etc.
 *
 *  $Id: apitrigeom.h,v 1.5 2011/02/02 06:06:30 johns Exp $
 */

void rt_tri_fcylinder(SceneHandle, void *, apivector, apivector, flt);
void rt_tri_cylinder(SceneHandle, void *, apivector, apivector, flt);
void rt_tri_ring(SceneHandle, void *, apivector, apivector, flt, flt);
void rt_tri_plane(SceneHandle, void *, apivector, apivector);
void rt_tri_box(SceneHandle, void *, apivector, apivector);

