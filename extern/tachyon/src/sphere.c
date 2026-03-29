/* 
 * sphere.c - This file contains the functions for dealing with spheres.
 *
 *  $Id: sphere.c,v 1.33 2012/10/17 04:25:57 johns Exp $
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

#define SPHERE_PRIVATE
#include "sphere.h"

static object_methods sphere_methods = {
  (void (*)(const void *, void *))(sphere_intersect),
  (void (*)(const void *, const void *, const void *, void *))(sphere_normal),
  sphere_bbox, 
  free 
};

object * newsphere(void * tex, vector ctr, flt rad) {
  sphere * s;
  
  s=(sphere *) malloc(sizeof(sphere));
  memset(s, 0, sizeof(sphere));
  s->methods = &sphere_methods;

  s->tex=tex;
  s->ctr=ctr;
  s->rad=rad;

  return (object *) s;
}

static int sphere_bbox(void * obj, vector * min, vector * max) {
  sphere * s = (sphere *) obj;

  min->x = s->ctr.x - s->rad;
  min->y = s->ctr.y - s->rad;
  min->z = s->ctr.z - s->rad;
  max->x = s->ctr.x + s->rad;
  max->y = s->ctr.y + s->rad;
  max->z = s->ctr.z + s->rad;

  return 1;
}

static void sphere_intersect(const sphere * spr, ray * ry) {
  flt b, disc, t1, t2, temp;
  vector V;

  VSUB(spr->ctr, ry->o, V);
  VDOT(b, V, ry->d); 
  VDOT(temp, V, V);  

  disc=b*b + spr->rad*spr->rad - temp;

  if (disc<=0.0) return;
  disc=SQRT(disc);

  t2=b+disc;
  if (t2 <= SPEPSILON) 
    return;
  ry->add_intersection(t2, (object *) spr, ry);  

  t1=b-disc;
  if (t1 > SPEPSILON) 
    ry->add_intersection(t1, (object *) spr, ry);  
}

static void sphere_normal(const sphere * spr, const vector * pnt, const ray * incident, vector * N) {
  flt invlen;

  N->x = pnt->x - spr->ctr.x;
  N->y = pnt->y - spr->ctr.y;
  N->z = pnt->z - spr->ctr.z;

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


