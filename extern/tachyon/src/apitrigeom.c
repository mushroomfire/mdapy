/*
 * apitrigeom.c - This file contains code for generating triangle tesselated
 *                geometry, for use with OpenGL, XGL, etc.
 * 
 *  $Id: apitrigeom.c,v 1.11 2012/10/17 04:25:57 johns Exp $
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TACHYON_INTERNAL 1
#include "tachyon.h"
#include "apitrigeom.h"

/* XXX Hack!  This needs cleanup.. */
void VNorm(apivector * a);
void VCross(apivector * a, apivector * b, apivector * c);
void VAddS(flt a, apivector * A, apivector * B, apivector * C);
void VAdd(apivector * a, apivector * b, apivector * c);
void VSub(apivector * a, apivector * b, apivector * c);
flt VDot(apivector *a, apivector *b);

#define CYLFACETS 36
#define RINGFACETS 36
#define SPHEREFACETS 25

void rt_tri_fcylinder(SceneHandle scene, void * tex, apivector ctr, apivector axis, flt rad) {
  apivector x, y, z, tmp;
  double u, v, u2, v2;
  int j;
  apivector p1, p2, p3, p4;
  apivector n1, n2;

  z = axis;
  VNorm(&z);
  tmp.x = z.y - 2.1111111;
  tmp.y = -z.z + 3.14159267;
  tmp.z = z.x - 3.915292342341;
  VNorm(&z);
  VNorm(&tmp);
  VCross(&z, &tmp, &x);
  VNorm(&x);
  VCross(&x, &z, &y);
  VNorm(&y);

  for (j=0; j<CYLFACETS; j++) {
     u = rad * SIN((6.28 * j) / (CYLFACETS - 1.0));
     v = rad * COS((6.28 * j) / (CYLFACETS - 1.0));
    u2 = rad * SIN((6.28 * (j + 1.0)) / (CYLFACETS - 1.0));
    v2 = rad * COS((6.28 * (j + 1.0)) / (CYLFACETS - 1.0));

    p1.x = p1.y = p1.z = 0.0;
    p4 = p3 = p2 = p1;

    VAddS(u, &x, &p1, &p1);
    VAddS(v, &y, &p1, &p1);
    n1 = p1;
    VNorm(&n1);
    VAddS(1.0, &ctr, &p1, &p1);
  

    VAddS(u2, &x, &p2, &p2);
    VAddS(v2, &y, &p2, &p2);
    n2 = p2;
    VNorm(&n2);
    VAddS(1.0, &ctr, &p2, &p2);

    VAddS(1.0, &axis, &p1, &p3);
    VAddS(1.0, &axis, &p2, &p4);

    rt_stri(scene, tex, p1, p2, p3, n1, n2, n1);
    rt_stri(scene, tex, p3, p2, p4, n1, n2, n2);
  }
}

void rt_tri_cylinder(SceneHandle scene, void * tex, apivector ctr, apivector axis, flt rad) {
  rt_fcylinder(scene, tex, ctr, axis, rad);
}

void rt_tri_ring(SceneHandle scene, void * tex, apivector ctr, apivector norm, flt a, flt b) {
  apivector x, y, z, tmp;
  double u, v, u2, v2;
  int j;
  apivector p1, p2, p3, p4;
  apivector n1, n2;

  z = norm;
  VNorm(&z);
  tmp.x = z.y - 2.1111111;
  tmp.y = -z.z + 3.14159267;
  tmp.z = z.x - 3.915292342341;
  VNorm(&z);
  VNorm(&tmp);
  VCross(&z, &tmp, &x);
  VNorm(&x);
  VCross(&x, &z, &y);
  VNorm(&y);

  for (j=0; j<RINGFACETS; j++) {
     u = SIN((6.28 * j) / (RINGFACETS - 1.0));
     v = COS((6.28 * j) / (RINGFACETS - 1.0));
    u2 = SIN((6.28 * (j + 1.0)) / (RINGFACETS - 1.0));
    v2 = COS((6.28 * (j + 1.0)) / (RINGFACETS - 1.0));

    p1.x = p1.y = p1.z = 0.0;
    p4 = p3 = p2 = p1;

    VAddS(u, &x, &p1, &p1);
    VAddS(v, &y, &p1, &p1);
    n1 = p1;
    VNorm(&n1);
    VAddS(a, &n1, &ctr, &p1);
    VAddS(b, &n1, &ctr, &p3);

    VAddS(u2, &x, &p2, &p2);
    VAddS(v2, &y, &p2, &p2);
    n2 = p2;
    VNorm(&n2);
    VAddS(a, &n2, &ctr, &p2);
    VAddS(b, &n2, &ctr, &p4);

    rt_stri(scene, tex, p1, p2, p3, norm, norm, norm);
    rt_stri(scene, tex, p3, p2, p4, norm, norm, norm);

  }
} 

void rt_tri_box(SceneHandle scene, void * tex, apivector min, apivector max) {
  /* -XY face */
  rt_tri(scene, tex, rt_vector(min.x, min.y, min.z),
                     rt_vector(min.x, max.y, min.z), 
                     rt_vector(max.x, max.y, min.z));
  rt_tri(scene, tex, rt_vector(min.x, min.y, min.z),
                     rt_vector(max.x, max.y, min.z), 
                     rt_vector(max.x, min.y, min.z));

  /* +XY face */
  rt_tri(scene, tex, rt_vector(min.x, min.y, max.z),
                     rt_vector(max.x, max.y, max.z),
                     rt_vector(min.x, max.y, max.z)); 
  rt_tri(scene, tex, rt_vector(min.x, min.y, max.z),
                     rt_vector(max.x, min.y, max.z),
                     rt_vector(max.x, max.y, max.z)); 

  /* -YZ face */
  rt_tri(scene, tex, rt_vector(min.x, min.y, min.z),
                     rt_vector(min.x, max.y, max.z),
                     rt_vector(min.x, min.y, max.z)); 
  rt_tri(scene, tex, rt_vector(min.x, min.y, min.z),
                     rt_vector(min.x, max.y, min.z),
                     rt_vector(min.x, max.y, max.z)); 

  /* +YZ face */
  rt_tri(scene, tex, rt_vector(max.x, min.y, min.z),
                     rt_vector(max.x, min.y, max.z),
                     rt_vector(max.x, max.y, max.z));
  rt_tri(scene, tex, rt_vector(max.x, min.y, min.z),
                     rt_vector(max.x, max.y, max.z),
                     rt_vector(max.x, max.y, min.z));

  /* -XZ face */
  rt_tri(scene, tex, rt_vector(min.x, min.y, min.z),
                     rt_vector(min.x, min.y, max.z), 
                     rt_vector(max.x, min.y, max.z));
  rt_tri(scene, tex, rt_vector(min.x, min.y, min.z),
                     rt_vector(max.x, min.y, max.z), 
                     rt_vector(max.x, min.y, min.z));

  /* +XZ face */
  rt_tri(scene, tex, rt_vector(min.x, max.y, min.z),
                     rt_vector(max.x, max.y, max.z),
                     rt_vector(min.x, max.y, max.z)); 
  rt_tri(scene, tex, rt_vector(min.x, max.y, min.z),
                     rt_vector(max.x, max.y, min.z),
                     rt_vector(max.x, max.y, max.z)); 
}

void rt_tri_sphere(SceneHandle scene, void * tex, apivector ctr, flt rad) {
}

void rt_tri_plane(SceneHandle scene, void * tex, apivector ctr, apivector norm) {
  rt_tri_ring(scene, tex, ctr, norm, 0.0, 10000.0);
} 

