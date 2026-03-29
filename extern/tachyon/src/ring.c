/* 
 * ring.c - This file contains the functions for dealing with rings.
 *
 *  $Id: ring.c,v 1.22 2012/10/17 04:25:57 johns Exp $
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

#define RING_PRIVATE
#include "ring.h"

static object_methods ring_methods = {
  (void (*)(const void *, void *))(ring_intersect),
  (void (*)(const void *, const void *, const void *, void *))(ring_normal),
  ring_bbox, 
  free 
};

object * newring(void * tex, vector ctr, vector norm, flt inrad, flt outrad) {
  ring * r;
  
  r=(ring *) malloc(sizeof(ring));
  memset(r, 0, sizeof(ring));
  r->methods = &ring_methods;

  r->tex = tex;
  r->ctr = ctr;
  r->norm = norm;
  VNorm(&r->norm);
  r->inrad = inrad;
  r->outrad= outrad;

  return (object *) r;
}

static int ring_bbox(void * obj, vector * min, vector * max) {
  ring * r = (ring *) obj;

  min->x = r->ctr.x - r->outrad;
  min->y = r->ctr.y - r->outrad;
  min->z = r->ctr.z - r->outrad;
  max->x = r->ctr.x + r->outrad;
  max->y = r->ctr.y + r->outrad;
  max->z = r->ctr.z + r->outrad;

  return 1;
}

static void ring_intersect(const ring * rng, ray * ry) {
  flt d;
  flt t,td;
  vector hit, pnt;
  
  d = -VDot(&(rng->ctr), &(rng->norm));
   
  t=-(d+VDot(&(rng->norm), &(ry->o)));
  td=VDot(&(rng->norm),&(ry->d)); 
  if (td != 0.0) {
    t= t / td;
    if (t>=0.0) {
      hit=Raypnt(ry, t);
      VSUB(hit, rng->ctr, pnt);
      VDOT(td, pnt, pnt);
      td=SQRT(td);
      if ((td > rng->inrad) && (td < rng->outrad)) 
        ry->add_intersection(t,(object *) rng, ry);
    }
  }
}

static void ring_normal(const ring * rng, const vector * pnt, const ray * incident, vector * N) {
  *N=rng->norm;

  /* Flip surface normal to point toward the viewer if necessary */
  if (VDot(N, &(incident->d)) > 0.0)  {
    N->x=-N->x;
    N->y=-N->y;
    N->z=-N->z;
  }
}

