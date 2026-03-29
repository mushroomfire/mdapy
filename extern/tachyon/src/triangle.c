/* 
 * triangle.c - This file contains the functions for dealing with triangles.
 *
 *  $Id: triangle.c,v 1.40 2012/10/17 04:25:57 johns Exp $
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TACHYON_INTERNAL 1
#include "tachyon.h"
#include "vector.h"
#include "macros.h"
#include "intersect.h"
#include "util.h"

#define TRIANGLE_PRIVATE
#include "triangle.h"

static object_methods tri_methods = {
  (void (*)(const void *, void *))(tri_intersect),
  (void (*)(const void *, const void *, const void *, void *))(tri_normal),
  tri_bbox, 
  free 
};

static object_methods stri_methods = {
  (void (*)(const void *, void *))(tri_intersect),
  (void (*)(const void *, const void *, const void *, void *))(stri_normal),
  tri_bbox, 
  free 
};

static object_methods stri_methods_reverse = {
  (void (*)(const void *, void *))(tri_intersect),
  (void (*)(const void *, const void *, const void *, void *))(stri_normal_reverse),
  tri_bbox, 
  free 
};

static object_methods stri_methods_guess = {
  (void (*)(const void *, void *))(tri_intersect),
  (void (*)(const void *, const void *, const void *, void *))(stri_normal_guess),
  tri_bbox, 
  free 
};

object * newtri(void * tex, vector v0, vector v1, vector v2) {
  tri * t;
  vector edge1, edge2, edge3;

  VSub(&v1, &v0, &edge1);
  VSub(&v2, &v0, &edge2);
  VSub(&v2, &v1, &edge3);

  /* check to see if this will be a degenerate triangle before creation */
  if ((VLength(&edge1) >= EPSILON) && 
      (VLength(&edge2) >= EPSILON) && 
      (VLength(&edge3) >= EPSILON)) {

    t=(tri *) malloc(sizeof(tri));

    t->nextobj = NULL;
    t->methods = &tri_methods;

    t->tex = tex;
    t->v0 = v0;
    t->edge1 = edge1;
    t->edge2 = edge2;
 
    return (object *) t;
  }
  
  return NULL; /* was a degenerate triangle */
}


object * newstri(void * tex, vector v0, vector v1, vector v2,
                           vector n0, vector n1, vector n2) {
  stri * t;
  vector edge1, edge2, edge3;

  VSub(&v1, &v0, &edge1);
  VSub(&v2, &v0, &edge2);
  VSub(&v2, &v1, &edge3);

  /* check to see if this will be a degenerate triangle before creation */
  if ((VLength(&edge1) >= EPSILON) && 
      (VLength(&edge2) >= EPSILON) &&
      (VLength(&edge3) >= EPSILON)) {

    t=(stri *) malloc(sizeof(stri));

    t->nextobj = NULL;
    t->methods = &stri_methods;
 
    t->tex = tex;
    t->v0 = v0;
    t->edge1 = edge1;
    t->edge2 = edge2;
    t->n0 = n0;
    t->n1 = n1;
    t->n2 = n2;

    return (object *) t;
  }

  return NULL; /* was a degenerate triangle */
}


void stri_normal_fixup(object *otri, int mode) {
  stri *t = (stri *) otri;

  switch (mode) {
/*    case RT_NORMAL_FIXUP_GUESS: */
    case 2:
      t->methods = &stri_methods_guess;
      break;

/*    case RT_NORMAL_FIXUP_FLIP: */
    case 1:
      t->methods = &stri_methods_reverse;
      break;

/*    case RT_NORMAL_FIXUP_OFF: */
    case 0:
    default:
      t->methods = &stri_methods;
      break;
  }
}


object * newvcstri(void * voidtex, vector v0, vector v1, vector v2,
                   vector n0, vector n1, vector n2,
                   color c0, color c1, color c2) {
  vcstri * t;
  vector edge1, edge2, edge3;
  vcstri_texture * tex = (vcstri_texture *) voidtex; 

  VSub(&v1, &v0, &edge1);
  VSub(&v2, &v0, &edge2);
  VSub(&v2, &v1, &edge3);

  /* check to see if this will be a degenerate triangle before creation */
  if ((VLength(&edge1) >= EPSILON) && 
      (VLength(&edge2) >= EPSILON) &&
      (VLength(&edge3) >= EPSILON)) {

    t=(vcstri *) malloc(sizeof(vcstri));

    t->nextobj = NULL;
    t->methods = &stri_methods;
 
    t->v0 = v0;
    t->edge1 = edge1;
    t->edge2 = edge2;
    t->n0 = n0;
    t->n1 = n1;
    t->n2 = n2;

    tex->c0 = c0;
    tex->c1 = c1;
    tex->c2 = c2;
    tex->obj = t; /* XXX hack to let the texture function get c0c1c2 data */
    tex->texfunc = (color(*)(const void *, const void *, void *))(vcstri_color);
    t->tex = (texture *) tex;

    return (object *) t;
  }

  return NULL; /* was a degenerate triangle */
}


void vcstri_normal_fixup(object *otri, int mode) {
  vcstri *t = (vcstri *) otri;

  switch (mode) {
/*    case RT_NORMAL_FIXUP_GUESS: */
    case 2:
      t->methods = &stri_methods_guess;
      break;

/*    case RT_NORMAL_FIXUP_FLIP: */
    case 1:
      t->methods = &stri_methods_reverse;
      break;

/*    case RT_NORMAL_FIXUP_OFF: */
    case 0:
    default:
      t->methods = &stri_methods;
      break;
  }
}


#define CROSS(dest,v1,v2) \
          dest.x=v1.y*v2.z-v1.z*v2.y; \
          dest.y=v1.z*v2.x-v1.x*v2.z; \
          dest.z=v1.x*v2.y-v1.y*v2.x;

#define DOT(v1,v2) (v1.x*v2.x+v1.y*v2.y+v1.z*v2.z)

#define SUB(dest,v1,v2) \
          dest.x=v1.x-v2.x; \
          dest.y=v1.y-v2.y; \
          dest.z=v1.z-v2.z;

static int tri_bbox(void * obj, vector * min, vector * max) {
  tri * t = (tri *) obj;
  vector v1, v2;

  VAdd(&t->v0, &t->edge1, &v1); 
  VAdd(&t->v0, &t->edge2, &v2); 

  min->x = MYMIN( t->v0.x , MYMIN( v1.x , v2.x ));
  min->y = MYMIN( t->v0.y , MYMIN( v1.y , v2.y ));
  min->z = MYMIN( t->v0.z , MYMIN( v1.z , v2.z ));

  max->x = MYMAX( t->v0.x , MYMAX( v1.x , v2.x ));
  max->y = MYMAX( t->v0.y , MYMAX( v1.y , v2.y ));
  max->z = MYMAX( t->v0.z , MYMAX( v1.z , v2.z ));

  return 1;
}

static void tri_intersect(const tri * trn, ray * ry) {
  vector tvec, pvec, qvec;
  flt det, inv_det, t, u, v;

  /* begin calculating determinant - also used to calculate U parameter */
  CROSS(pvec, ry->d, trn->edge2);

  /* if determinant is near zero, ray lies in plane of triangle */
  det = DOT(trn->edge1, pvec);

#if 0           /* define TEST_CULL if culling is desired */
   if (det < EPSILON)
      return;

   /* calculate distance from vert0 to ray origin */
   SUB(tvec, ry->o, trn->v0);

   /* calculate U parameter and test bounds */
   u = DOT(tvec, pvec);
   if (u < 0.0 || u > det)
      return;

   /* prepare to test V parameter */
   CROSS(qvec, tvec, trn->edge1);

   /* calculate V parameter and test bounds */
   v = DOT(ry->d, qvec);
   if (v < 0.0 || u + v > det)
      return;

   /* calculate t, scale parameters, ray intersects triangle */
   t = DOT(trn->edge2, qvec);
   inv_det = 1.0 / det;
   t *= inv_det;
   u *= inv_det;
   v *= inv_det;
#else                    /* the non-culling branch */
   if (det > -EPSILON && det < EPSILON)
     return;

   inv_det = 1.0 / det;

   /* calculate distance from vert0 to ray origin */
   SUB(tvec, ry->o, trn->v0);

   /* calculate U parameter and test bounds */
   u = DOT(tvec, pvec) * inv_det;
   if (u < 0.0 || u > 1.0)
     return;

   /* prepare to test V parameter */
   CROSS(qvec, tvec, trn->edge1);

   /* calculate V parameter and test bounds */
   v = DOT(ry->d, qvec) * inv_det;
   if (v < 0.0 || u + v > 1.0)
     return;

   /* calculate t, ray intersects triangle */
   t = DOT(trn->edge2, qvec) * inv_det;
#endif

  ry->add_intersection(t,(object *) trn, ry);
}


static void tri_normal(const tri * trn, const vector * hit, const ray * incident, vector * N) {
  flt invlen;

  CROSS((*N), trn->edge1, trn->edge2);

  invlen = 1.0 / SQRT(N->x*N->x + N->y*N->y + N->z*N->z);
  N->x *= invlen;
  N->y *= invlen;
  N->z *= invlen;

  /* Flip surface normal to point toward the viewer if necessary */
  if (VDot(N, &(incident->d)) > 0.0)  {
    N->x=-N->x;
    N->y=-N->y;
    N->z=-N->z;
  }
}


static void stri_normal(const stri * trn, const vector * hit, const ray * incident, vector * N) {
  flt U, V, W, lensqr, invlen;
  vector P, tmp, norm;
  
  CROSS(norm, trn->edge1, trn->edge2);
  lensqr = DOT(norm, norm); 

  VSUB((*hit), trn->v0, P);

  CROSS(tmp, P, trn->edge2);
  U = DOT(tmp, norm) / lensqr;   

  CROSS(tmp, trn->edge1, P);
  V = DOT(tmp, norm) / lensqr;   

  W = 1.0 - (U + V);

  N->x = W*trn->n0.x + U*trn->n1.x + V*trn->n2.x;
  N->y = W*trn->n0.y + U*trn->n1.y + V*trn->n2.y;
  N->z = W*trn->n0.z + U*trn->n1.z + V*trn->n2.z;

  invlen = 1.0 / SQRT(N->x*N->x + N->y*N->y + N->z*N->z);
  N->x *= invlen;
  N->y *= invlen;
  N->z *= invlen;

  /* Flip surface normal to point toward the viewer if necessary  */
  /* Note: unlike the normal routines for other objects, the code */
  /*       for triangles interpolated surface normals tests the   */
  /*       vertex winding order rather than using the resulting   */
  /*       interpolated normal.                                   */
  if (VDot(&norm, &(incident->d)) > 0.0)  {
    N->x=-N->x;
    N->y=-N->y;
    N->z=-N->z;
  }
}


color vcstri_color(const vector * hit, const texture * tx, const ray * incident) {
  vcstri_texture * tex = (vcstri_texture *) tx;
  const vcstri * trn = (vcstri *) tex->obj;
  flt U, V, W, lensqr;
  vector P, tmp, norm;
  color col;
  
  CROSS(norm, trn->edge1, trn->edge2);
  lensqr = DOT(norm, norm); 

  VSUB((*hit), trn->v0, P);

  CROSS(tmp, P, trn->edge2);
  U = DOT(tmp, norm) / lensqr;   

  CROSS(tmp, trn->edge1, P);
  V = DOT(tmp, norm) / lensqr;   

  W = 1.0 - (U + V);

  col.r = W*tex->c0.r + U*tex->c1.r + V*tex->c2.r;
  col.g = W*tex->c0.g + U*tex->c1.g + V*tex->c2.g;
  col.b = W*tex->c0.b + U*tex->c1.b + V*tex->c2.b;

  return col;
}


static void stri_normal_reverse(const stri * trn, const vector * hit, const ray * incident, vector * N) {
  flt U, V, W, lensqr, invlen;
  vector P, tmp, norm;
  
  CROSS(norm, trn->edge1, trn->edge2);
  lensqr = DOT(norm, norm); 

  VSUB((*hit), trn->v0, P);

  CROSS(tmp, P, trn->edge2);
  U = DOT(tmp, norm) / lensqr;   

  CROSS(tmp, trn->edge1, P);
  V = DOT(tmp, norm) / lensqr;   

  W = 1.0 - (U + V);

  N->x = W*trn->n0.x + U*trn->n1.x + V*trn->n2.x;
  N->y = W*trn->n0.y + U*trn->n1.y + V*trn->n2.y;
  N->z = W*trn->n0.z + U*trn->n1.z + V*trn->n2.z;

  invlen = 1.0 / SQRT(N->x*N->x + N->y*N->y + N->z*N->z);
  N->x *= invlen;
  N->y *= invlen;
  N->z *= invlen;

  /* Flip surface normal to point toward the viewer if necessary  */
  /* Note: unlike the normal routines for other objects, the code */
  /*       for triangles interpolated surface normals tests the   */
  /*       vertex winding order rather than using the resulting   */
  /*       interpolated normal.                                   */
  /* Note: This version is the reverse of the normal version      */
  if (VDot(&norm, &(incident->d)) < 0.0)  {
    N->x=-N->x;
    N->y=-N->y;
    N->z=-N->z;
  }
}


static void stri_normal_guess(const stri * trn, const vector * hit, const ray * incident, vector * N) {
  flt U, V, W, lensqr, invlen;
  vector P, tmp, norm;
  
  CROSS(norm, trn->edge1, trn->edge2);
  lensqr = DOT(norm, norm); 

  VSUB((*hit), trn->v0, P);

  CROSS(tmp, P, trn->edge2);
  U = DOT(tmp, norm) / lensqr;   

  CROSS(tmp, trn->edge1, P);
  V = DOT(tmp, norm) / lensqr;   

  W = 1.0 - (U + V);

  N->x = W*trn->n0.x + U*trn->n1.x + V*trn->n2.x;
  N->y = W*trn->n0.y + U*trn->n1.y + V*trn->n2.y;
  N->z = W*trn->n0.z + U*trn->n1.z + V*trn->n2.z;

  invlen = 1.0 / SQRT(N->x*N->x + N->y*N->y + N->z*N->z);
  N->x *= invlen;
  N->y *= invlen;
  N->z *= invlen;

  /* Flip surface normal to point toward the viewer if necessary  */
  /* XXX NOTE: this is actually incorrect, but will sorta work    */
  /* for surfaces with inconsistent winding order and mean vertex */
  /* normal directions.  This implementation is provided only for */
  /* cases where the incoming geometry can't be fixed and is      */
  /* randomly mixing winding order and normal direction.          */
  if (VDot(N, &(incident->d)) > 0.0)  {
    N->x=-N->x;
    N->y=-N->y;
    N->z=-N->z;
  }
}
