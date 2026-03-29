/* 
 * cylinder.c - This file contains the functions for dealing with cylinders.
 *
 *  $Id: cylinder.c,v 1.29 2012/10/17 04:25:57 johns Exp $
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TACHYON_INTERNAL 1
#include "tachyon.h"
#include "macros.h"
#include "vector.h"
#include "intersect.h"
#include "util.h"

#define CYLINDER_PRIVATE 
#include "cylinder.h"

static object_methods cylinder_methods = {
  (void (*)(const void *, void *))(cylinder_intersect),
  (void (*)(const void *, const void *, const void *, void *))(cylinder_normal),
  cylinder_bbox, 
  free 
};

static object_methods fcylinder_methods = {
  (void (*)(const void *, void *))(fcylinder_intersect),
  (void (*)(const void *, const void *, const void *, void *))(cylinder_normal),
  fcylinder_bbox, 
  free 
};


object * newcylinder(void * tex, vector ctr, vector axis, flt rad) {
  cylinder * c;
  
  c=(cylinder *) malloc(sizeof(cylinder));
  memset(c, 0, sizeof(cylinder));
  c->methods = &cylinder_methods;

  c->tex=(texture *) tex;
  c->ctr=ctr;
  c->axis=axis;
  c->rad=rad;
  return (object *) c;
}

static int cylinder_bbox(void * obj, vector * min, vector * max) {
  return 0; /* infinite / unbounded object */
}

static void cylinder_intersect(const cylinder * cyl, ray * ry) {
  vector rc, n, D, O;  
  flt t, s, tin, tout, ln, d; 

  rc.x = ry->o.x - cyl->ctr.x;
  rc.y = ry->o.y - cyl->ctr.y;
  rc.z = ry->o.z - cyl->ctr.z; 

  VCross(&ry->d, &cyl->axis, &n);

  ln=SQRT(n.x*n.x + n.y*n.y + n.z*n.z);    /* finish length calculation */

  if (ln == 0.0) {  /* ray is parallel to the cylinder.. */
    VDOT(d, rc, cyl->axis);         
    D.x = rc.x - d * cyl->axis.x; 
    D.y = rc.y - d * cyl->axis.y;
    D.z = rc.z - d * cyl->axis.z;
    VDOT(d, D, D);
    d = SQRT(d);
    tin = -FHUGE;
    tout = FHUGE;
    /* if (d <= cyl->rad) then ray is inside cylinder.. else outside */
  }

  n.x /= ln;
  n.y /= ln;
  n.z /= ln;

  VDOT(d, rc, n);
  d = FABS(d); 

  if (d <= cyl->rad) {  /* ray intersects cylinder.. */
    VCross(&rc, &cyl->axis, &O);
    VDOT(t, O, n);
    t = - t / ln;
    VCross(&n, &cyl->axis, &O); 

    ln = SQRT(O.x*O.x + O.y*O.y + O.z*O.z);
    O.x /= ln;
    O.y /= ln;
    O.z /= ln;

    VDOT(s, ry->d, O);
    s = FABS(SQRT(cyl->rad*cyl->rad - d*d) / s);
    tin = t - s;
    ry->add_intersection(tin, (object *) cyl, ry); 
    tout = t + s;
    ry->add_intersection(tout, (object *) cyl, ry);
  }
}

static void cylinder_normal(const cylinder * cyl, const vector * pnt, const ray * incident, vector * N) {
  vector a, b;
  flt t, invlen, invlen2;

  a.x = pnt->x - cyl->ctr.x;
  a.y = pnt->y - cyl->ctr.y;
  a.z = pnt->z - cyl->ctr.z;

  b=cyl->axis;

  invlen = 1.0 / SQRT(b.x*b.x + b.y*b.y + b.z*b.z);
  b.x *= invlen;
  b.y *= invlen;
  b.z *= invlen;
 
  VDOT(t, a, b);

  N->x = pnt->x - (b.x * t + cyl->ctr.x); 
  N->y = pnt->y - (b.y * t + cyl->ctr.y);
  N->z = pnt->z - (b.z * t + cyl->ctr.z);

  invlen2 = 1.0 / SQRT(N->x*N->x + N->y*N->y + N->z*N->z);
  N->x *= invlen2;
  N->y *= invlen2;
  N->z *= invlen2;

  /* Flip surface normal to point toward the viewer if necessary */
  if (VDot(N, &(incident->d)) > 0.0)  {
    N->x=-N->x;
    N->y=-N->y;
    N->z=-N->z;
  }
}

object * newfcylinder(void * tex, vector ctr, vector axis, flt rad) {
  cylinder * c;
  
  c=(cylinder *) malloc(sizeof(cylinder));
  memset(c, 0, sizeof(cylinder));
  c->methods = &fcylinder_methods;

  c->tex=(texture *) tex;
  c->ctr=ctr;
  c->axis=axis;
  c->rad=rad;

  return (object *) c;
}

static int fcylinder_bbox(void * obj, vector * min, vector * max) {
  cylinder * c = (cylinder *) obj;
  vector mintmp, maxtmp;

  mintmp.x = c->ctr.x;
  mintmp.y = c->ctr.y;
  mintmp.z = c->ctr.z;
  maxtmp.x = c->ctr.x + c->axis.x;
  maxtmp.y = c->ctr.y + c->axis.y;
  maxtmp.z = c->ctr.z + c->axis.z;

  min->x = MYMIN(mintmp.x, maxtmp.x);
  min->y = MYMIN(mintmp.y, maxtmp.y);
  min->z = MYMIN(mintmp.z, maxtmp.z);
  min->x -= c->rad;
  min->y -= c->rad;
  min->z -= c->rad;

  max->x = MYMAX(mintmp.x, maxtmp.x);
  max->y = MYMAX(mintmp.y, maxtmp.y);
  max->z = MYMAX(mintmp.z, maxtmp.z);
  max->x += c->rad;
  max->y += c->rad;
  max->z += c->rad;

  return 1;
}


static void fcylinder_intersect(const cylinder * cyl, ray * ry) {
  vector rc, n, O, hit, tmp2, ctmp4;
  flt t, s, tin, tout, ln, d, tmp, tmp3;
 
  rc.x = ry->o.x - cyl->ctr.x;  
  rc.y = ry->o.y - cyl->ctr.y;
  rc.z = ry->o.z - cyl->ctr.z;
 
  VCross(&ry->d, &cyl->axis, &n);
 
  ln=SQRT(n.x*n.x + n.y*n.y + n.z*n.z);    /* finish length calculation */
 
  if (ln == 0.0) {  /* ray is parallel to the cylinder.. */
    return;       /* in this case, we want to miss or go through the "hole" */
  }

  n.x /= ln;
  n.y /= ln;
  n.z /= ln;
 
  VDOT(d, rc, n);
  d = FABS(d);
 
  if (d <= cyl->rad) {  /* ray intersects cylinder.. */
    VCross(&rc, &cyl->axis, &O);
    VDOT(t, O, n);
    t = - t / ln;
    VCross(&n, &cyl->axis, &O);

    ln = SQRT(O.x*O.x + O.y*O.y + O.z*O.z);
    O.x /= ln;
    O.y /= ln;
    O.z /= ln;

    VDOT(s, ry->d, O);
    s = FABS(SQRT(cyl->rad*cyl->rad - d*d) / s);
    tin = t - s;

    RAYPNT(hit, (*ry), tin); 

    ctmp4=cyl->axis;
    VNorm(&ctmp4);

    tmp2.x = hit.x - cyl->ctr.x;   
    tmp2.y = hit.y - cyl->ctr.y;   
    tmp2.z = hit.z - cyl->ctr.z;   

    VDOT(tmp,  tmp2, ctmp4);
    VDOT(tmp3, cyl->axis, cyl->axis);

    if ((tmp > 0.0) && (tmp < SQRT(tmp3))) 
      ry->add_intersection(tin, (object *) cyl, ry);
    tout = t + s;

    RAYPNT(hit, (*ry), tout); 

    tmp2.x = hit.x - cyl->ctr.x;   
    tmp2.y = hit.y - cyl->ctr.y;   
    tmp2.z = hit.z - cyl->ctr.z;   

    VDOT(tmp,  tmp2, ctmp4); 
    VDOT(tmp3, cyl->axis, cyl->axis);

    if ((tmp > 0.0) && (tmp < SQRT(tmp3))) 
      ry->add_intersection(tout, (object *) cyl, ry);
  }
}

