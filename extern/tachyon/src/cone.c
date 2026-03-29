/* 
 * cone.c - This file contains the functions for dealing with cones.
 *
 *  Added by Alexander Stukowski.
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

#define CONE_PRIVATE
#include "cone.h"

static object_methods cone_methods = {
  (void (*)(const void *, void *))(cone_intersect),
  (void (*)(const void *, const void *, const void *, void *))(cone_normal),
  cone_bbox,
  free 
};

object * newcone(void * tex, vector ctr, vector axis, flt rad) {
  cone * c;
  flt angle;
  
  c=(cone *) malloc(sizeof(cone));
  memset(c, 0, sizeof(cone));
  c->methods = &cone_methods;

  c->tex=(texture *) tex;
  c->ctr=ctr;
  c->axis=axis;
  c->rad=rad;
  c->height=VLength(&axis);
  angle = atan(rad/c->height);
  c->cos_angle=cos(angle);
  c->sin_angle=sin(angle);

  return (object *) c;
}

static void cone_normal(const cone * cone, const vector * pnt, const ray * incident, vector * N) {
  vector a, b;
  flt t, invlen, invlen2, r;

  a.x = pnt->x - cone->ctr.x;
  a.y = pnt->y - cone->ctr.y;
  a.z = pnt->z - cone->ctr.z;

  b=cone->axis;

  invlen = 1.0 / cone->height;
  b.x *= invlen;
  b.y *= invlen;
  b.z *= invlen;
 
  VDOT(t, a, b);

  N->x = pnt->x - (b.x * t + cone->ctr.x);
  N->y = pnt->y - (b.y * t + cone->ctr.y);
  N->z = pnt->z - (b.z * t + cone->ctr.z);

  r = (t * cone->sin_angle / cone->cos_angle * cone->sin_angle / cone->height);
  N->x -= cone->axis.x * r;
  N->y -= cone->axis.y * r;
  N->z -= cone->axis.z * r;

  invlen2 = 1.0 / sqrt(N->x*N->x + N->y*N->y + N->z*N->z);
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

static int cone_bbox(void * obj, vector * min, vector * max) {
  cone * c = (cone *) obj;
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


static void cone_intersect(const cone * cone, ray * ry) {
  vector E, hit;
  flt AdD, cosSqr, AdE, DdE, EdE, c2, c1, c0, dot, discr, root, invC2, t;

  VDOT(AdD, cone->axis, ry->d);
  AdD /= cone->height;
  cosSqr = cone->cos_angle * cone->cos_angle;
  VSub(&ry->o, &cone->ctr, &E);
  VDOT(AdE, cone->axis, E);
  AdE /= cone->height;
  VDOT(DdE, ry->d, E);
  VDOT(EdE, E, E);
  c2 = AdD*AdD - cosSqr;
  c1 = AdD*AdE - cosSqr*DdE;
  c0 = AdE*AdE - cosSqr*EdE;

  // Solve the quadratic.  Keep only those X for which Dot(A,X-V) >= 0.
  if(fabs(c2) >= 1e-9) {
      // c2 != 0
      discr = c1*c1 - c0*c2;
      if(discr < 0.0) {
          // Q(t) = 0 has no real-valued roots.  The line does not
          // intersect the double-sided cone.
    	  return;
      }
      else if(discr > 1e-9)
      {
          // Q(t) = 0 has two distinct real-valued roots.  However, one or
          // both of them might intersect the portion of the double-sided
          // cone "behind" the vertex.  We are interested only in those
          // intersections "in front" of the vertex.
          root = sqrt(discr);
          invC2 = 1.0/c2;

          t = (-c1 - root)*invC2;
          RAYPNT(hit, (*ry), t);
          VSub(&hit, &cone->ctr, &E);
          VDOT(dot, E, cone->axis);
          if(dot > 0.0 && dot < cone->height*cone->height)
        	  ry->add_intersection(t, (object *) cone, ry);

          t = (-c1 + root)*invC2;
          RAYPNT(hit, (*ry), t);
          VSub(&hit, &cone->ctr, &E);
          VDOT(dot, E, cone->axis);
          if(dot > 0.0 && dot < cone->height*cone->height)
        	  ry->add_intersection(t, (object *) cone, ry);
      }
      else {
          // One repeated real root (line is tangent to the cone).
          RAYPNT(hit, (*ry), -(c1/c2));
          VSub(&hit, &cone->ctr, &E);
          if(VDot(&E, &cone->axis) > 0.0)
        	  ry->add_intersection(-(c1/c2), (object *) cone, ry);
      }
  }
  else if(fabs(c1) >= 1e-9) {
      // c2 = 0, c1 != 0 (D is a direction vector on the cone boundary)
	  RAYPNT(hit, (*ry), -(0.5*c0/c1));
      VSub(&hit, &cone->ctr, &E);
      VDOT(dot, E, cone->axis);
      if(dot > 0.0)
    	  ry->add_intersection(-(0.5*c0/c1), (object *) cone, ry);
  }
  else if(fabs(c0) >= 1e-9) {
      // c2 = c1 = 0, c0 != 0
      return;
  }
  else {
      // c2 = c1 = c0 = 0, cone contains ray V+t*D where V is cone vertex
      // and D is the line direction.
	  ry->add_intersection(DdE, (object *) cone, ry);
  }
}

