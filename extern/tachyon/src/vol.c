/*
 * vol.c - Volume rendering helper routines etc.
 *
 *
 *  $Id: vol.c,v 1.52 2013/04/21 02:30:46 johns Exp $
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
#include "vol.h"
#include "box.h"
#include "trace.h"
#include "ui.h"
#include "shade.h"
#include "texture.h"

int scalarvol_bbox(void * obj, vector * min, vector * max) {
  box * b = (box *) obj;

  *min = b->min;
  *max = b->max;

  return 1;
}

#if 0  /* not yet... */
static object_methods scalarvol_methods = {
  (void (*)(void *, void *))(box_intersect),
  (void (*)(void *, void *, void *, void *))(box_normal),
  scalarvol_bbox, 
  free 
};
#endif

void * newscalarvol(void * voidtex, vector min, vector max, 
                    int xs, int ys, int zs, const char * fname, 
                    scalarvol * invol) {
  standard_texture * tx, * tex;
  scalarvol * vol;

  tex=(standard_texture *) voidtex;
  tex->flags = RT_TEXTURE_NOFLAGS; /* doesn't cast a shadow */

  tx=malloc(sizeof(standard_texture));

  /* is the volume data already loaded? */
  if (invol==NULL) {
    vol=malloc(sizeof(scalarvol));
    vol->loaded=0;
    vol->data=NULL;
  } else {
    vol=invol;
  }

  vol->opacity=tex->opacity;
  vol->xres=xs;
  vol->yres=ys;
  vol->zres=zs;
  strcpy(vol->name, fname);

  tx->ctr.x = 0.0;
  tx->ctr.y = 0.0;
  tx->ctr.z = 0.0;
  tx->rot   = tx->ctr;
  tx->scale = tx->ctr;
  tx->uaxs  = tx->ctr;
  tx->vaxs  = tx->ctr;

  tx->flags = RT_TEXTURE_NOFLAGS;

  tx->col = tex->col;
  tx->ambient   = 1.0;
  tx->diffuse   = 0.0;
  tx->phong     = 0.0;
  tx->phongexp  = 0.0;
  tx->phongtype = 0;
  tx->specular  = 0.0;
  tx->opacity   = 1.0;
  tx->outline   = 0.0;
  tx->outlinewidth = 0.0;
  tx->img = vol;
  tx->texfunc = (color(*)(const void *, const void *, void *))(scalar_volume_texture);

  tx->obj = (void *) newbox(tx, min, max); /* XXX hack!! */

  /* Force load of volume data so that we don't have to do mutex locks */
  /* inside the rendering threads                                      */
  if (!vol->loaded) {
    LoadVol(vol);
  }

  /* check if loading succeeded */
  if (!vol->loaded) {
    tx->texfunc = (color(*)(const void *, const void *, void *))(constant_texture);
    tx->img = NULL;
    free(vol);
  }

  return (void *) tx->obj;
}


color VoxelColor(flt scalar) {
  color col;

  if (scalar > 1.0) 
    scalar = 1.0;

  if (scalar < 0.0)
    scalar = 0.0;

  if (scalar < 0.25) {
    col.r = scalar * 4.0;
    col.g = 0.0;
    col.b = 0.0;
  }
  else {
    if (scalar < 0.75) {
      col.r = 1.0;
      col.g = (scalar - 0.25) * 2.0;
      col.b = 0.0;
    }
    else {
      col.r = 1.0;
      col.g = 1.0;
      col.b = (scalar - 0.75) * 4.0;
    }
  }

  return col;
} 

color scalar_volume_texture(const vector * hit, const texture * tx, ray * ry) {
  color col, col2;
  box * bx;
  flt a, tx1, tx2, ty1, ty2, tz1, tz2;
  flt tnear, tfar;
  flt t, tdist, dt, sum, tt; 
  vector pnt, bln;
  scalarvol * vol;
  flt scalar, transval; 
  int x, y, z;
  unsigned char * ptr;
  standard_texture * tex = (standard_texture *) tx;

  bx=(box *) tex->obj;
  vol=(scalarvol *) ((standard_texture *) bx->tex)->img;
   
  col.r=0.0;
  col.g=0.0;
  col.b=0.0;
 
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

#if 0
  /* XXX this is where we cause early exit if the volumetric */
  /*     object intersects other geometric objects           */

  /* stop at closest intersection from other objects */
  if (ry->maxdist < tfar) 
    tfar = ry->maxdist;
#endif

  if (tnear > tfar) return col;
  if (tfar < 0.0) return col;
 
  if (tnear < 0.0) tnear=0.0;
 
  tdist=SQRT(vol->xres*vol->xres + vol->yres*vol->yres + vol->zres*vol->zres);
  tt = (vol->opacity / tdist); 

  bln.x=FABS(bx->min.x - bx->max.x);
  bln.y=FABS(bx->min.y - bx->max.y);
  bln.z=FABS(bx->min.z - bx->max.z);
  
  dt=SQRT(bln.x*bln.x + bln.y*bln.y + bln.z*bln.z) / tdist; 
  sum=0.0;

  for (t=tnear; t<=tfar; t+=dt) {
    pnt.x=((ry->o.x + (ry->d.x * t)) - bx->min.x) / bln.x;
    pnt.y=((ry->o.y + (ry->d.y * t)) - bx->min.y) / bln.y;
    pnt.z=((ry->o.z + (ry->d.z * t)) - bx->min.z) / bln.z;
 
    x=(int) ((vol->xres - 1.5) * pnt.x + 0.5);
    y=(int) ((vol->yres - 1.5) * pnt.y + 0.5);
    z=(int) ((vol->zres - 1.5) * pnt.z + 0.5);
   
    ptr = vol->data + ((vol->xres * vol->yres * z) + (vol->xres * y) + x);
   
    scalar = (flt) ((flt) 1.0 * ((int) ptr[0])) / 255.0;

    sum += tt * scalar; 

    transval = tt * scalar; 

    col2 = VoxelColor(scalar);

    if (sum < 1.0) {
      col.r += transval * col2.r;
      col.g += transval * col2.g;
      col.b += transval * col2.b;
      if (sum < 0.0) sum=0.0;
    }  
    else { 
      sum=1.0;
    }
  }


  /* XXX this has to be changed in order to allow volumetric objects */
  /*     to intersect with geometric objects                         */

  if (sum < 1.0) {      /* spawn transmission rays / refraction */    
    color transcol;
    shadedata shadevars;
    shadevars.hit=*hit;
    transcol = shade_transmission(ry, &shadevars, 1.0 - sum);

    col.r += transcol.r; /* add the transmitted ray  */    
    col.g += transcol.g; /* to the diffuse and       */
    col.b += transcol.b; /* transmission total..     */  
  }

  return col;
}

void LoadVol(scalarvol * vol) { 
  FILE * dfile;
 
  dfile=fopen(vol->name, "r");
  if (dfile==NULL) {
    char msgtxt[2048];
    sprintf(msgtxt, "Can't load volume %s, using object color", vol->name); 
    rt_ui_message(MSG_ERR, msgtxt);
    return;
  }  
 
  if (rt_mynode()==0) {
    char msgtxt[2048];
    sprintf(msgtxt, "Loading %dx%dx%d volume set from %s",
	vol->xres, vol->yres, vol->zres, vol->name);
    rt_ui_message(MSG_0, msgtxt);
  } 
  vol->data = malloc(vol->xres * vol->yres * vol->zres);

  if (fread(vol->data, (vol->xres * vol->yres * vol->zres), 1, dfile) == 1) {
    vol->loaded=1;
  } else {
    char msgtxt[2048];
    sprintf(msgtxt, "Can't load volume %s, using object color", vol->name); 
    rt_ui_message(MSG_ERR, msgtxt);
  }

  fclose(dfile);
}


