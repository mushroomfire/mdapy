/*
 * extvol.c - Volume rendering helper routines etc.
 *
 *  $Id: extvol.c,v 1.30 2012/10/17 04:25:57 johns Exp $
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TACHYON_INTERNAL 1
#include "tachyon.h"
#include "macros.h"
#include "vector.h"
#include "util.h"
#include "parallel.h"
#include "threads.h"
#include "box.h"
#include "extvol.h"
#include "trace.h"
#include "sphere.h"
#include "light.h"
#include "shade.h"


int extvol_bbox(void * obj, vector * min, vector * max) {
  box * b = (box *) obj;

  *min = b->min;
  *max = b->max;

  return 1;
}

static object_methods extvol_methods = {
  (void (*)(const void *, void *))(box_intersect),
  (void (*)(const void *, const void *, const void *, void *))(box_normal),
  extvol_bbox, 
  free 
};

extvol * newextvol(void * voidtex, vector min, vector max, 
                   int samples, flt (* evaluator)(flt, flt, flt)) { 
  extvol * xvol;
  standard_texture * tex, * xvoltex;
  
  tex = (standard_texture *) voidtex;

  xvol = (extvol *) malloc(sizeof(extvol));
  memset(xvol, 0, sizeof(extvol));

  xvol->methods = &extvol_methods;

  xvol->min=min;
  xvol->max=max;
  xvol->evaluator = evaluator;
  xvol->ambient = tex->ambient;
  xvol->diffuse = tex->diffuse;
  xvol->opacity = tex->opacity;  
  xvol->samples = samples;

  xvoltex = malloc(sizeof(standard_texture));
  memset(xvoltex, 0, sizeof(standard_texture));

  xvoltex->ctr.x = 0.0;
  xvoltex->ctr.y = 0.0;
  xvoltex->ctr.z = 0.0;
  xvoltex->rot = xvoltex->ctr;
  xvoltex->scale = xvoltex->ctr;
  xvoltex->uaxs = xvoltex->ctr;
  xvoltex->vaxs = xvoltex->ctr;
  xvoltex->flags = RT_TEXTURE_NOFLAGS;

  xvoltex->col=tex->col;
  xvoltex->ambient=1.0;
  xvoltex->diffuse=0.0;
  xvoltex->specular=0.0;
  xvoltex->opacity=1.0;
  xvoltex->img=NULL;
  xvoltex->texfunc=(color(*)(const void *, const void *, void *))(ext_volume_texture);
  xvoltex->obj = (void *) xvol; /* XXX hack! */

  xvol->tex = (texture *) xvoltex;

  return xvol;
}

color ExtVoxelColor(flt scalar) {
  color col;

  if (scalar > 1.0) 
    scalar = 1.0;

  if (scalar < 0.0)
    scalar = 0.0;

  if (scalar < 0.5) {
    col.g = 0.0;
  }
  else {
    col.g = (scalar - 0.5) * 2.0;
  }

  col.r = scalar;
  col.b = 1.0 - (scalar / 2.0);

  return col;
} 

color ext_volume_texture(const vector * hit, const texture * tx, ray * ry) {
  color col, col2;
  box * bx;
  extvol * xvol;
  flt a, tx1, tx2, ty1, ty2, tz1, tz2;
  flt tnear, tfar;
  flt t, tdist, dt, ddt, sum, tt; 
  vector pnt, bln;
  flt scalar, transval; 
  point_light * li;
  color diffint; 
  vector N, L;
  flt inten;
  standard_texture * tex = (standard_texture *) tx;

    bx = (box *) tex->obj;
  xvol = (extvol *) tex->obj;

  col.r = 0.0;
  col.g = 0.0;
  col.b = 0.0;
 
  tnear= -FHUGE;
  tfar= FHUGE;
 
  if (ry->d.x == 0.0) {
    if ((ry->o.x < bx->min.x) || (ry->o.x > bx->max.x)) return col;
  }
  else {
    tx1 = (bx->min.x - ry->o.x) / ry->d.x;
    tx2 = (bx->max.x - ry->o.x) / ry->d.x;
    if (tx1 > tx2) { a=tx1; tx1=tx2; tx2=a; }
    if (tx1 > tnear) tnear=tx1;
    if (tx2 < tfar)   tfar=tx2;
  }
  if (tnear > tfar) return col;
  if (tfar < 0.0) return col;
 
 if (ry->d.y == 0.0) {
    if ((ry->o.y < bx->min.y) || (ry->o.y > bx->max.y)) return col;
  }
  else {
    ty1 = (bx->min.y - ry->o.y) / ry->d.y;
    ty2 = (bx->max.y - ry->o.y) / ry->d.y;
    if (ty1 > ty2) { a=ty1; ty1=ty2; ty2=a; }
    if (ty1 > tnear) tnear=ty1;
    if (ty2 < tfar)   tfar=ty2;
  }
  if (tnear > tfar) return col;
  if (tfar < 0.0) return col;
 
  if (ry->d.z == 0.0) {
    if ((ry->o.z < bx->min.z) || (ry->o.z > bx->max.z)) return col;
  }
  else {
    tz1 = (bx->min.z - ry->o.z) / ry->d.z;
    tz2 = (bx->max.z - ry->o.z) / ry->d.z;
    if (tz1 > tz2) { a=tz1; tz1=tz2; tz2=a; }
    if (tz1 > tnear) tnear=tz1;
    if (tz2 < tfar)   tfar=tz2;
  }
  if (tnear > tfar) return col;
  if (tfar < 0.0) return col;
 
  if (tnear < 0.0) tnear=0.0;
 
  tdist = xvol->samples;

  tt = (xvol->opacity / tdist); 

  bln.x=FABS(bx->min.x - bx->max.x);
  bln.y=FABS(bx->min.y - bx->max.y);
  bln.z=FABS(bx->min.z - bx->max.z);
  
     dt = 1.0 / tdist; 
    sum = 0.0;

/* Accumulate color as the ray passes through the voxels */
  for (t=tnear; t<=tfar; t+=dt) {
    if (sum < 1.0) {
      pnt.x=((ry->o.x + (ry->d.x * t)) - bx->min.x) / bln.x;
      pnt.y=((ry->o.y + (ry->d.y * t)) - bx->min.y) / bln.y;
      pnt.z=((ry->o.z + (ry->d.z * t)) - bx->min.z) / bln.z;

      /* call external evaluator assume 0.0 -> 1.0 range.. */ 
      scalar = xvol->evaluator(pnt.x, pnt.y, pnt.z);  

      transval = tt * scalar; 
      sum += transval; 

      col2 = ExtVoxelColor(scalar);

      col.r += transval * col2.r * xvol->ambient;
      col.g += transval * col2.g * xvol->ambient;
      col.b += transval * col2.b * xvol->ambient;

      ddt = dt;

      /* Add in diffuse shaded light sources (no shadows) */
      if (xvol->diffuse > 0.0) {
  
        /* Calculate the Volume gradient at the voxel */
        N.x = (xvol->evaluator(pnt.x - ddt, pnt.y, pnt.z)  -  
              xvol->evaluator(pnt.x + ddt, pnt.y, pnt.z)) *  8.0 * tt; 
  
        N.y = (xvol->evaluator(pnt.x, pnt.y - ddt, pnt.z)  -  
              xvol->evaluator(pnt.x, pnt.y + ddt, pnt.z)) *  8.0 * tt; 
  
        N.z = (xvol->evaluator(pnt.x, pnt.y, pnt.z - ddt)  -  
              xvol->evaluator(pnt.x, pnt.y, pnt.z + ddt)) *  8.0 * tt; 
 
        /* only light surfaces with enough of a normal.. */
        if ((N.x*N.x + N.y*N.y + N.z*N.z) > 0.0) { 
          list * cur;

          diffint.r = 0.0; 
          diffint.g = 0.0; 
          diffint.b = 0.0; 


          /* add the contribution of each of the lights.. */
          cur = ry->scene->lightlist;
          while (cur != NULL) {           /* loop for light contributions */
            li=(point_light *) cur->item; /* set li=to the current light  */
            VSUB(li->ctr, (*hit), L)
            VNorm(&L);
            VDOT(inten, N, L)
    
            /* only add light if its from the front of the surface */
            /* could add back-lighting if we wanted to later.. */
            if (inten > 0.0) {
              standard_texture * litex = (standard_texture *) li->tex;

              diffint.r += inten * litex->col.r;
              diffint.g += inten * litex->col.g;
              diffint.b += inten * litex->col.b;
            }

            cur = cur->next;
          }   

          col.r += col2.r * diffint.r * xvol->diffuse;
          col.g += col2.g * diffint.g * xvol->diffuse;
          col.b += col2.b * diffint.b * xvol->diffuse;
        }
      }
    }   
    else { 
      sum=1.0;
    }  
  }

  /* Add in transmitted ray from outside environment */
  if (sum < 1.0) {      /* spawn transmission rays / refraction */
    color transcol;
    shadedata shadevars;

    shadevars.hit=*hit;

    /* XXX this ought to be done in shade.c rather than here  */
    /*     if done in shade.c, we could do volumetric objects */
    /*     after solids are already known and handle          */
    /*     object-volume intersections better.                */
    transcol = shade_transmission(ry, &shadevars, 1.0 - sum);

    col.r += transcol.r; /* add the transmitted ray  */
    col.g += transcol.g; /* to the diffuse and       */
    col.b += transcol.b; /* transmission total..     */
  }

  return col;
}



